import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

!pip install pandas openpyxl requests

!pip install gdown

import gdown
import pandas as pd

# Google Drive File ID
file_id = "1zMqLpkxJTMmnxG4d-v_u31QdyUGQAZRn"
output = "downloaded_file.xlsx"

# Download file using gdown
gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", output, quiet=False)

df = pd.read_excel(output, engine="openpyxl", nrows=10000)

print("Full dataset shape:", df.shape)
df = df.sample(n=1000, random_state=42)

df.columns = df.columns.str.strip().str.lower()

pressure_mean = df['total-pressure'].mean()
pressure_std = df['total-pressure'].std()
velocity_mean = df['velocity-magnitude'].mean()
velocity_std = df['velocity-magnitude'].std()
temperature_mean = df['total-temperature'].mean()
temperature_std = df['total-temperature'].std()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size=256, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.output_dim = mapping_size
        self.scale = scale
        self.B = torch.randn(input_dim, mapping_size//2) * scale
        
    def forward(self, x):
        x_proj = (2 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FluidFlowPINN(nn.Module):
    def __init__(self, inner_radius, middle_radius, outer_radius, channel_length,
                 inner_inlet_velocity, middle_inlet_velocity, outer_inlet_velocity,
                 inner_inlet_species, middle_inlet_species, outer_inlet_species,
                 hidden_layers=[256, 256, 256, 256, 256]):
        super().__init__()
        
        # Geometry parameters
        self.r_i = inner_radius
        self.r_m = middle_radius
        self.r_o = outer_radius
        self.L = channel_length
        
        # Inlet conditions
        self.v_i = inner_inlet_velocity
        self.v_m = middle_inlet_velocity
        self.v_o = outer_inlet_velocity
        self.species_i = inner_inlet_species
        self.species_m = middle_inlet_species
        self.species_o = outer_inlet_species

        # Network architecture
        self.input_size = 2
        self.output_size = 12  # pressure, velocity, temperature, 9 species
        
        # Fourier feature encoding
        self.fourier_features = FourierFeatures(self.input_size, mapping_size=256)
        
        # Build network layers
        layers = []
        layer_sizes = [256] + hidden_layers
        
        for i in range(len(layer_sizes)-1):
            layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.LayerNorm(layer_sizes[i+1]),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
        
        layers.append(nn.Linear(layer_sizes[-1], self.output_size))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, y):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
            
        xy = torch.cat([x, y], dim=1)
        xy = self.fourier_features(xy)
        output = self.network(xy)
        
        pressure = output[:, 0:1]
        velocity = output[:, 1:2]
        temperature = output[:, 2:3]
        species = torch.sigmoid(output[:, 3:])
        
        return pressure, velocity, temperature, species

    def compute_boundary_loss(self, x_batch, y_batch):
        r = torch.sqrt(x_batch**2 + y_batch**2)
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Inner region (r < r_i)
        inner_mask = (r < self.r_i) & (x_batch < 1e-4)
        if inner_mask.any():
            inner_points = torch.stack([x_batch[inner_mask], y_batch[inner_mask]], dim=1)
            p_i, v_i, t_i, species_i = self(inner_points[:, 0], inner_points[:, 1])
            inner_loss = (
                torch.mean((v_i - self.v_i)**2) +
                torch.mean((species_i - self.species_i)**2)
            )
            total_loss = total_loss + inner_loss
        
        # Middle region (r_i < r < r_m)
        middle_mask = (r > self.r_i) & (r < self.r_m) & (x_batch < 1e-4)
        if middle_mask.any():
            middle_points = torch.stack([x_batch[middle_mask], y_batch[middle_mask]], dim=1)
            p_m, v_m, t_m, species_m = self(middle_points[:, 0], middle_points[:, 1])
            middle_loss = (
                torch.mean((v_m - self.v_m)**2) +
                torch.mean((species_m - self.species_m)**2)
            )
            total_loss = total_loss + middle_loss
        
        # Outer region (r_m < r < r_o)
        outer_mask = (r > self.r_m) & (r < self.r_o) & (x_batch < 1e-4)
        if outer_mask.any():
            outer_points = torch.stack([x_batch[outer_mask], y_batch[outer_mask]], dim=1)
            p_o, v_o, t_o, species_o = self(outer_points[:, 0], outer_points[:, 1])
            outer_loss = (
                torch.mean((v_o - self.v_o)**2) +
                torch.mean((species_o - self.species_o)**2)
            )
            total_loss = total_loss + outer_loss
        
        # Wall boundary conditions
        wall_masks = [
            torch.abs(r - self.r_i) < 1e-4,
            torch.abs(r - self.r_m) < 1e-4,
            torch.abs(r - self.r_o) < 1e-4
        ]
        
        for wall_mask in wall_masks:
            if wall_mask.any():
                wall_points = torch.stack([x_batch[wall_mask], y_batch[wall_mask]], dim=1)
                _, v_wall, _, _ = self(wall_points[:, 0], wall_points[:, 1])
                wall_loss = torch.mean(v_wall**2)  # No-slip condition
                total_loss = total_loss + wall_loss
        
        return total_loss
    
def safe_grad(loss, inputs, create_graph=True):
    grads = torch.autograd.grad(loss, inputs, create_graph=create_graph, 
                               allow_unused=True, retain_graph=True)
    grads = [torch.zeros_like(inp) if g is None else g for g, inp in zip(grads, inputs)]
    grads = [torch.nan_to_num(g, nan=0.0, posinf=1e-1, neginf=-1e-1) for g in grads]
    grads = [torch.clamp(g, min=-1.0, max=1.0) for g in grads]
    return grads

def compute_navier_stokes_loss(model, x, y, scale=1e-3):
    x_var = x.clone().detach().requires_grad_(True)
    y_var = y.clone().detach().requires_grad_(True)
    
    p, v, t, species = model(x_var, y_var)
    dp_dx, dp_dy = safe_grad(p.sum(), (x_var, y_var))
    dv_dx, dv_dy = safe_grad(v.sum(), (x_var, y_var))
    
    r = torch.sqrt(x_var**2 + y_var**2 + 1e-6)
    
    continuity = torch.clamp(scale * (dv_dx + dv_dy + v/(r + 1e-6)), -1.0, 1.0)
    momentum_r = torch.clamp(scale * (v * dv_dx - v**2/(r + 1e-6) + dp_dx), -1.0, 1.0)
    momentum_theta = torch.clamp(scale * (v * dv_dy + dp_dy), -1.0, 1.0)

    return torch.mean(continuity**2 + momentum_r**2 + momentum_theta**2)

def compute_reaction_source(species_conc, temperature, temperature_mean, temperature_std):
    temperature = temperature * temperature_std + temperature_mean
    
    Y_NH3 = species_conc[:, -1]
    Y_O2 = species_conc[:, 2]
    
    A = 1.1e7
    Ta = 10064
    
    Q = torch.clamp(
        A * Y_NH3 * (Y_O2**0.5) * torch.exp(-Ta/temperature.squeeze()),
        min=-1.0, max=1.0
    )
    
    source_terms = torch.zeros_like(species_conc)
    source_terms[:, -1] = -1.0 * Q    # NH3
    source_terms[:, 2] = -0.75 * Q    # O2
    source_terms[:, 3] = 1.5 * Q      # H2O
    source_terms[:, 5] = 1.0 * Q      # NO
    
    return source_terms

def compute_species_transport_loss(model, x, y, temperature_mean, temperature_std, scale=1e-3):
    x_var = x.clone().detach().requires_grad_(True)
    y_var = y.clone().detach().requires_grad_(True)
    
    _, v, t, species = model(x_var, y_var)
    r = torch.sqrt(x_var**2 + y_var**2 + 1e-6)
    
    species_loss = 0
    for i in range(species.shape[1]):
        ds_dx, ds_dy = safe_grad(species[:, i].sum(), (x_var, y_var))
        ds_dr = torch.clamp((x_var * ds_dx + y_var * ds_dy)/(r + 1e-6), -1.0, 1.0)
        convection = torch.clamp(scale * (v * ds_dr), -1.0, 1.0)
        diffusion = torch.clamp(scale * (ds_dr/(r + 1e-6) + 
                              torch.gradient(r * ds_dr, dim=0)[0]/(r + 1e-6)), -1.0, 1.0)
        
        reaction_source = torch.clamp(
            scale * compute_reaction_source(species, t, temperature_mean, temperature_std)[:, i],
            -1.0, 1.0
        )
        transport = convection - diffusion - reaction_source
        species_loss += torch.mean(transport**2)
    
    return species_loss


# Load and preprocess data
def preprocess_data(df, channel_length, outer_radius):
    # Calculate normalization parameters
    pressure_mean = df['total-pressure'].mean()
    pressure_std = df['total-pressure'].std()
    velocity_mean = df['velocity-magnitude'].mean()
    velocity_std = df['velocity-magnitude'].std()
    temperature_mean = df['total-temperature'].mean()
    temperature_std = df['total-temperature'].std()
    
    df_norm = df.copy()
    
    # Normalize spatial coordinates
    df_norm['x-coordinate'] = df['x-coordinate'] / channel_length
    df_norm['y-coordinate'] = df['y-coordinate'] / outer_radius
    
    # Normalize physics quantities
    df_norm['total-pressure'] = (df['total-pressure'] - pressure_mean) / pressure_std
    df_norm['velocity-magnitude'] = (df['velocity-magnitude'] - velocity_mean) / velocity_std
    df_norm['total-temperature'] = (df['total-temperature'] - temperature_mean) / temperature_std
    
    # Process species data
    species_columns = ['n2', 'h2', 'o2', 'h2o', 'oh', 'no', 'no2', 'n2o', 'nh3']
    for col in species_columns:
        df_norm[col] = np.clip(df[col], 0, 1)
    
    return df_norm, {
        'pressure': (pressure_mean, pressure_std),
        'velocity': (velocity_mean, velocity_std),
        'temperature': (temperature_mean, temperature_std)
    }


def initialize_model():
    # Define geometry parameters
    inner_radius = 0.005
    middle_radius = 0.0075
    outer_radius = 0.01
    channel_length = 0.550

    # Define inlet velocities
    inner_inlet_velocity = 18.13
    middle_inlet_velocity = 15.0
    outer_inlet_velocity = 10.0

    # Define species concentrations
    inner_inlet_species = torch.tensor([3.231520e-08, 0.05, 1.639588e-09, 1.686688e-09,
                                      2.563035e-16, 2.780497e-31, 2.326587e-10,
                                      1.270694e-11, 0.950000], dtype=torch.float32)
    
    middle_inlet_species = torch.tensor([0.79, 0.0, 0.18, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      dtype=torch.float32)
    
    outer_inlet_species = torch.tensor([0.79, 0.0, 0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     dtype=torch.float32)

    model = FluidFlowPINN(
        inner_radius=inner_radius,
        middle_radius=middle_radius,
        outer_radius=outer_radius,
        channel_length=channel_length,
        inner_inlet_velocity=inner_inlet_velocity,
        middle_inlet_velocity=middle_inlet_velocity,
        outer_inlet_velocity=outer_inlet_velocity,
        inner_inlet_species=inner_inlet_species,
        middle_inlet_species=middle_inlet_species,
        outer_inlet_species=outer_inlet_species
    )
    
    return model

def train_model(model, x_train, y_train, data_train, norm_params, epochs=1000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-8
    )
    
    temperature_mean, temperature_std = norm_params['temperature']
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(x_train, y_train)
        outputs_concat = torch.cat(outputs, dim=1)
        
        data_loss = torch.mean((outputs_concat - data_train)**2)
        physics_loss = (
            compute_navier_stokes_loss(model, x_train, y_train) +
            compute_species_transport_loss(model, x_train, y_train,
                                        temperature_mean, temperature_std)
        )
        bc_loss = model.compute_boundary_loss(x_train, y_train)
        
        loss = data_loss + physics_loss + bc_loss
        
        if torch.isnan(loss):
            print(f"NaN loss at epoch {epoch}")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.6f}')
    
    return model


import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def setup_model_parameters():
    # 1. Define geometry parameters
    inner_radius = 0.005  # meters (fluid region)
    middle_radius = 0.0075  # meters (hot co-flow region)
    outer_radius = 0.01  # meters (air region)
    channel_length = 0.550  # meters

    # 2. Define inlet velocities
    inner_inlet_velocity = 18.13  # m/s (fuel stream)
    middle_inlet_velocity = 15.0  # m/s (hot co-flow)
    outer_inlet_velocity = 10.0  # m/s (air)

    # 3. Define species concentrations
    inner_inlet_species = torch.tensor([
        3.231520e-08,  # N2
        0.05,          # H2
        1.639588e-09,  # O2
        1.686688e-09,  # H2O
        2.563035e-16,  # OH
        2.780497e-31,  # NO
        2.326587e-10,  # NO2
        1.270694e-11,  # N2O
        0.950000       # NH3
    ], dtype=torch.float32)

    middle_inlet_species = torch.tensor([
        0.79,  # N2
        0.0,   # H2
        0.18,  # O2
        0.03,  # H2O
        0.0,   # OH
        0.0,   # NO
        0.0,   # NO2
        0.0,   # N2O
        0.0    # NH3
    ], dtype=torch.float32)

    outer_inlet_species = torch.tensor([
        0.79,  # N2
        0.0,   # H2
        0.21,  # O2
        0.0,   # H2O
        0.0,   # OH
        0.0,   # NO
        0.0,   # NO2
        0.0,   # N2O
        0.0    # NH3
    ], dtype=torch.float32)

    return {
        'geometry': {
            'inner_radius': inner_radius,
            'middle_radius': middle_radius,
            'outer_radius': outer_radius,
            'channel_length': channel_length
        },
        'velocities': {
            'inner': inner_inlet_velocity,
            'middle': middle_inlet_velocity,
            'outer': outer_inlet_velocity
        },
        'species': {
            'inner': inner_inlet_species,
            'middle': middle_inlet_species,
            'outer': outer_inlet_species
        }
    }

def prepare_training_data(df, params):
    """Prepare training data from DataFrame"""
    # Normalize spatial coordinates
    x = torch.tensor(df['x-coordinate'].values / params['geometry']['channel_length'], 
                    dtype=torch.float32)
    y = torch.tensor(df['y-coordinate'].values / params['geometry']['outer_radius'], 
                    dtype=torch.float32)
    
    # Prepare target data
    pressure = torch.tensor(df['total-pressure'].values, dtype=torch.float32)
    velocity = torch.tensor(df['velocity-magnitude'].values, dtype=torch.float32)
    temperature = torch.tensor(df['total-temperature'].values, dtype=torch.float32)
    
    # Prepare species data
    species_columns = ['n2', 'h2', 'o2', 'h2o', 'oh', 'no', 'no2', 'n2o', 'nh3']
    species = torch.tensor(df[species_columns].values, dtype=torch.float32)
    
    # Combine all target variables
    targets = torch.cat([
        pressure.unsqueeze(1),
        velocity.unsqueeze(1),
        temperature.unsqueeze(1),
        species
    ], dim=1)
    
    return x, y, targets

def train_pinn_model(df, num_epochs=1000):
    # Setup model parameters
    params = setup_model_parameters()
    
    # Initialize model
    model = FluidFlowPINN(
        inner_radius=params['geometry']['inner_radius'],
        middle_radius=params['geometry']['middle_radius'],
        outer_radius=params['geometry']['outer_radius'],
        channel_length=params['geometry']['channel_length'],
        inner_inlet_velocity=params['velocities']['inner'],
        middle_inlet_velocity=params['velocities']['middle'],
        outer_inlet_velocity=params['velocities']['outer'],
        inner_inlet_species=params['species']['inner'],
        middle_inlet_species=params['species']['middle'],
        outer_inlet_species=params['species']['outer']
    )
    
    # Prepare training data
    x_train, y_train, targets = prepare_training_data(df, params)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-8
    )
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        p_pred, v_pred, t_pred, s_pred = model(x_train, y_train)
        pred = torch.cat([p_pred, v_pred, t_pred, s_pred], dim=1)
        
        # Compute losses
        data_loss = torch.mean((pred - targets)**2)
        physics_loss = compute_navier_stokes_loss(model, x_train, y_train)
        species_loss = compute_species_transport_loss(
            model, x_train, y_train,
            targets[:, 2].mean(), targets[:, 2].std()
        )
        bc_loss = model.compute_boundary_loss(x_train, y_train)
        
        # Total loss
        loss = data_loss + physics_loss + species_loss + bc_loss
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Print progress
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{num_epochs}:')
            print(f'  Total Loss: {loss.item():.6f}')
            print(f'  Data Loss: {data_loss.item():.6f}')
            print(f'  Physics Loss: {physics_loss.item():.6f}')
            print(f'  Species Loss: {species_loss.item():.6f}')
            print(f'  BC Loss: {bc_loss.item():.6f}')
    
    return model, losses

# Train the model
model, training_losses = train_pinn_model(df, num_epochs=1000)

# Save the trained model
torch.save(model.state_dict(), "trained_pinn_model.pt")

# Plot training losses
plt.figure(figsize=(10, 6))
plt.plot(training_losses)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.grid(True)
plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def generate_prediction_grid(geometry_params, num_points=100):
    """Generate a grid of points for prediction"""
    x = np.linspace(0, geometry_params['channel_length'], num_points)
    y = np.linspace(-geometry_params['outer_radius'], 
                    geometry_params['outer_radius'], num_points)
    X, Y = np.meshgrid(x, y)
    
    # Convert to torch tensors
    x_grid = torch.tensor(X.flatten(), dtype=torch.float32)
    y_grid = torch.tensor(Y.flatten(), dtype=torch.float32)
    
    return x_grid, y_grid, X, Y

def predict_flow_field(model, x_grid, y_grid):
    """Generate predictions for the entire domain"""
    model.eval()
    with torch.no_grad():
        pressure, velocity, temperature, species = model(x_grid, y_grid)
        
    # Reshape predictions
    n = int(np.sqrt(len(x_grid)))
    pressure = pressure.numpy().reshape(n, n)
    velocity = velocity.numpy().reshape(n, n)
    temperature = temperature.numpy().reshape(n, n)
    species = species.numpy().reshape(n, n, -1)
    
    return pressure, velocity, temperature, species

def plot_flow_field(X, Y, pressure, velocity, temperature, species, geometry_params):
    """Create visualizations of the predicted flow field"""
    # Custom colormap
    colors = ['navy', 'blue', 'cyan', 'yellow', 'red', 'darkred']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Predicted Flow Field', fontsize=16)
    
    # Plot pressure
    im0 = axes[0, 0].contourf(X, Y, pressure, levels=50, cmap=cmap)
    axes[0, 0].set_title('Pressure Distribution')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Plot velocity
    im1 = axes[0, 1].contourf(X, Y, velocity, levels=50, cmap=cmap)
    axes[0, 1].set_title('Velocity Magnitude')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot temperature
    im2 = axes[1, 0].contourf(X, Y, temperature, levels=50, cmap=cmap)
    axes[1, 0].set_title('Temperature Distribution')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot NH3 concentration (last species)
    im3 = axes[1, 1].contourf(X, Y, species[:, :, -1], levels=50, cmap=cmap)
    axes[1, 1].set_title('NH3 Concentration')
    plt.colorbar(im3, ax=axes[1, 1])
    
    # Add geometry markers
    for ax in axes.flat:
        # Draw channel boundaries
        ax.axhline(y=geometry_params['outer_radius'], color='k', linestyle='--')
        ax.axhline(y=-geometry_params['outer_radius'], color='k', linestyle='--')
        ax.axhline(y=geometry_params['middle_radius'], color='k', linestyle=':')
        ax.axhline(y=-geometry_params['middle_radius'], color='k', linestyle=':')
        ax.axhline(y=geometry_params['inner_radius'], color='k', linestyle=':')
        ax.axhline(y=-geometry_params['inner_radius'], color='k', linestyle=':')
        
        ax.set_xlabel('Axial Distance (m)')
        ax.set_ylabel('Radial Distance (m)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_species_distributions(X, Y, species, species_names):
    """Create visualizations of all species distributions"""
    n_species = species.shape[2]
    n_rows = (n_species + 2) // 3  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    fig.suptitle('Species Distributions', fontsize=16)
    
    # Custom colormap
    colors = ['navy', 'blue', 'cyan', 'yellow', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)
    
    for i in range(n_species):
        row = i // 3
        col = i % 3
        
        im = axes[row, col].contourf(X, Y, species[:, :, i], levels=50, cmap=cmap)
        axes[row, col].set_title(f'{species_names[i]} Distribution')
        plt.colorbar(im, ax=axes[row, col])
        
        axes[row, col].set_xlabel('Axial Distance (m)')
        axes[row, col].set_ylabel('Radial Distance (m)')
        axes[row, col].grid(True, alpha=0.3)
    
    # Remove empty subplots if any
    for i in range(n_species, n_rows * 3):
        row = i // 3
        col = i % 3
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    return fig

def predict_and_visualize(model_path, new_boundary_conditions=None):
    """Main function to load model, predict, and visualize results"""
    # Load the trained model
    model = FluidFlowPINN(
        # Geometry parameters
        inner_radius=0.005,    # 5mm - fuel stream radius
        middle_radius=0.0075,  # 7.5mm - hot co-flow radius
        outer_radius=0.01,     # 10mm - outer air stream radius
        channel_length=0.550,  # 550mm length
        
        # Inlet velocities
        inner_inlet_velocity=18.13,  # Fuel stream velocity (m/s)
        middle_inlet_velocity=15.0,  # Hot co-flow velocity (m/s)
        outer_inlet_velocity=10.0,   # Air stream velocity (m/s)
        
        # Inner inlet species (fuel stream - NH3 rich)
        inner_inlet_species=torch.tensor([
            3.231520e-08,  # N2  - Almost zero nitrogen
            0.05,          # H2  - 5% hydrogen
            1.639588e-09,  # O2  - Trace oxygen
            1.686688e-09,  # H2O - Trace water vapor
            2.563035e-16,  # OH  - Trace hydroxyl radical
            2.780497e-31,  # NO  - Negligible nitric oxide
            2.326587e-10,  # NO2 - Trace nitrogen dioxide
            1.270694e-11,  # N2O - Trace nitrous oxide
            0.950000       # NH3 - 95% ammonia (fuel)
        ]),
        
        # Middle inlet species (hot co-flow - vitiated air)
        middle_inlet_species=torch.tensor([
            0.79,  # N2  - 79% nitrogen
            0.0,   # H2  - No hydrogen
            0.18,  # O2  - 18% oxygen (slightly depleted)
            0.03,  # H2O - 3% water vapor (combustion product)
            0.0,   # OH  - No hydroxyl radical
            0.0,   # NO  - No nitric oxide
            0.0,   # NO2 - No nitrogen dioxide
            0.0,   # N2O - No nitrous oxide
            0.0    # NH3 - No ammonia
        ]),
        
        # Outer inlet species (pure air)
        outer_inlet_species=torch.tensor([
            0.79,  # N2  - 79% nitrogen (air composition)
            0.0,   # H2  - No hydrogen
            0.21,  # O2  - 21% oxygen (air composition)
            0.0,   # H2O - No water vapor
            0.0,   # OH  - No hydroxyl radical
            0.0,   # NO  - No nitric oxide
            0.0,   # NO2 - No nitrogen dioxide
            0.0,   # N2O - No nitrous oxide
            0.0    # NH3 - No 
    ])
    )    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # If new boundary conditions are provided, update the model
    if new_boundary_conditions:
        for attr, value in new_boundary_conditions.items():
            setattr(model, attr, value)
    
    # Generate prediction grid
    geometry_params = {
        'channel_length': 0.550,
        'outer_radius': 0.01,
        'middle_radius': 0.0075,
        'inner_radius': 0.005
    }
    
    x_grid, y_grid, X, Y = generate_prediction_grid(geometry_params)
    
    # Generate predictions
    pressure, velocity, temperature, species = predict_flow_field(model, x_grid, y_grid)
    
    # Create visualizations
    flow_field_fig = plot_flow_field(X, Y, pressure, velocity, temperature, species, 
                                   geometry_params)
    
    species_names = ['N2', 'H2', 'O2', 'H2O', 'OH', 'NO', 'NO2', 'N2O', 'NH3']
    species_fig = plot_species_distributions(X, Y, species, species_names)
    
    return flow_field_fig, species_fig



new_conditions = {
    'inner_inlet_velocity': 20.0,  # Increased inner velocity (fuel stream)
    'middle_inlet_velocity': 12.0,  # Decreased middle velocity (hot co-flow)
    'outer_inlet_velocity': 8.0,    # Decreased outer velocity (air)
    
    # Inner inlet species (fuel stream - rich in NH3)
    'inner_inlet_species': torch.tensor([
        3.231520e-08,  # N2  - Almost zero nitrogen
        0.07,          # H2  - 7% hydrogen
        1.639588e-09,  # O2  - Trace oxygen
        1.686688e-09,  # H2O - Trace water vapor
        2.563035e-16,  # OH  - Trace hydroxyl radical
        2.780497e-31,  # NO  - Negligible nitric oxide
        2.326587e-10,  # NO2 - Trace nitrogen dioxide
        1.270694e-11,  # N2O - Trace nitrous oxide
        0.930000       # NH3 - 93% ammonia (fuel)
    ])
}

# Generate predictions and plots
flow_fig, species_fig = predict_and_visualize(
    model_path="trained_pinn_model.pt",
    new_boundary_conditions=new_conditions
)

# Save the figures
flow_fig.savefig('flow_field_predictions.png', dpi=300, bbox_inches='tight')
species_fig.savefig('species_distributions.png', dpi=300, bbox_inches='tight')