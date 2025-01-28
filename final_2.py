!pip install pandas openpyxl requests

pip install gdown

import gdown
import pandas as pd

# Google Drive File ID
file_id = "1zMqLpkxJTMmnxG4d-v_u31QdyUGQAZRn"
output = "downloaded_file.xlsx"

# Download file using gdown
gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", output, quiet=False)

# Read the first 10 rows
df = pd.read_excel(output, engine="openpyxl", nrows=10)

import torch
import torch.nn as nn

class FluidFlowPINN(nn.Module):
    def __init__(self, inner_radius, outer_radius, channel_length, 
                 inner_inlet_velocity, outer_inlet_velocity,
                 inner_inlet_species, outer_inlet_species,
                 hidden_layers=[128, 128, 128, 128]):
        super().__init__()

        # modify layers
        
        # Geometry parameters
        self.r_i = inner_radius
        self.r_o = outer_radius
        self.L = channel_length
        
        # Inlet conditions
        self.v_i = inner_inlet_velocity
        self.v_o = outer_inlet_velocity
        self.species_i = inner_inlet_species  # Species concentrations for inner flow
        self.species_o = outer_inlet_species  # Species concentrations for outer flow
        

        # Input features: x, y coordinates
        self.input_size = 2
        
        # Output features: pressure, velocity, temperature, and 9 species concentrations
        self.output_size = 12  # 3 physical + 9 chemical species
        
        # Build the network
        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_size, hidden_layers[0]))
        layers.append(nn.Tanh()) 
        
        # Hidden layers
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.Tanh())
            
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], self.output_size))
        
        self.network = nn.Sequential(*layers)

    def compute_boundary_loss(self, x_batch, y_batch):
      """
      Compute losses for all boundary conditions
      """
      # Convert to polar coordinates for easier boundary handling
      r = torch.sqrt(x_batch**2 + y_batch**2)
      
      total_loss = torch.tensor(0.0, requires_grad=True)
      
      # 1. Inner wall boundary conditions (r = r_i)
      inner_wall_mask = torch.abs(r - self.r_i) < 1e-4
      if inner_wall_mask.any():
          inner_wall_points = torch.stack([
              x_batch[inner_wall_mask],
              y_batch[inner_wall_mask]
          ], dim=1)
          
          p_wall_i, v_wall_i, t_wall_i, species_wall_i = self(
              inner_wall_points[:, 0],
              inner_wall_points[:, 1]
          )
          inner_wall_loss = torch.mean(v_wall_i**2)
          total_loss = total_loss + inner_wall_loss
      
      # 2. Outer wall boundary conditions (r = r_o)
      outer_wall_mask = torch.abs(r - self.r_o) < 1e-4
      if outer_wall_mask.any():
          outer_wall_points = torch.stack([
              x_batch[outer_wall_mask],
              y_batch[outer_wall_mask]
          ], dim=1)
          
          p_wall_o, v_wall_o, t_wall_o, species_wall_o = self(
              outer_wall_points[:, 0],
              outer_wall_points[:, 1]
          )
          outer_wall_loss = torch.mean(v_wall_o**2)
          total_loss = total_loss + outer_wall_loss
      
      # 3. Inner inlet conditions (x = 0, r < r_i)
      inner_inlet_mask = (x_batch < 1e-4) & (r < self.r_i)
      if inner_inlet_mask.any():
          inner_inlet_points = torch.stack([
              x_batch[inner_inlet_mask],
              y_batch[inner_inlet_mask]
          ], dim=1)
          
          p_in_i, v_in_i, t_in_i, species_in_i = self(
              inner_inlet_points[:, 0],
              inner_inlet_points[:, 1]
          )
          
          inner_inlet_loss = (
              torch.mean((v_in_i - self.v_i)**2) +
              torch.mean((species_in_i - self.species_i)**2)
          )
          total_loss = total_loss + inner_inlet_loss

      # presssure and temperature at boundary include
      
      # 4. Outer inlet conditions (x = L, r_i < r < r_o)
      outer_inlet_mask = (torch.abs(x_batch - self.L) < 1e-4) & (r > self.r_i) & (r < self.r_o)
      if outer_inlet_mask.any():
          outer_inlet_points = torch.stack([
              x_batch[outer_inlet_mask],
              y_batch[outer_inlet_mask]
          ], dim=1)
          
          p_in_o, v_in_o, t_in_o, species_in_o = self(
              outer_inlet_points[:, 0],
              outer_inlet_points[:, 1]
          )
          
          outer_inlet_loss = (
              torch.mean((v_in_o - self.v_o)**2) +
              torch.mean((species_in_o - self.species_o)**2)
          )
          total_loss = total_loss + outer_inlet_loss
      
      # 5. Zero gradient outlet conditions
      # For inner flow outlet (x = L, r < r_i)
      inner_outlet_mask = (torch.abs(x_batch - self.L) < 1e-4) & (r < self.r_i)
      
      # For outer flow outlet (x = 0, r_i < r < r_o)
      outer_outlet_mask = (x_batch < 1e-4) & (r > self.r_i) & (r < self.r_o)
      
      if inner_outlet_mask.any() or outer_outlet_mask.any():
          # Compute gradients at outlets
          x_var = torch.tensor(x_batch, requires_grad=True)
          y_var = torch.tensor(y_batch, requires_grad=True)
          
          p, v, t, species = self(x_var, y_var)
          
          dp_dx = torch.autograd.grad(p.sum(), x_var, create_graph=True)[0]
          dv_dx = torch.autograd.grad(v.sum(), x_var, create_graph=True)[0]
          
          outlet_loss = torch.tensor(0.0, requires_grad=True)
          if inner_outlet_mask.any():
              outlet_loss = outlet_loss + torch.mean(dp_dx[inner_outlet_mask]**2) + torch.mean(dv_dx[inner_outlet_mask]**2)
          if outer_outlet_mask.any():
              outlet_loss = outlet_loss + torch.mean(dp_dx[outer_outlet_mask]**2) + torch.mean(dv_dx[outer_outlet_mask]**2)
              
          total_loss = total_loss + outlet_loss
      
      return total_loss

    def forward(self, x, y):
        # Ensure inputs are 2D tensors
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
            
        xy = torch.cat([x, y], dim=1)
        output = self.network(xy)
        
        # Split outputs into physical and chemical components
        pressure = output[:, 0:1]
        velocity = output[:, 1:2]
        temperature = output[:, 2:3]
        species = torch.sigmoid(output[:, 3:])  # Ensure species fractions are between 0 and 1
        
        return pressure, velocity, temperature, species
    
def navier_stokes_loss(model, x, y):
    # Create variables requiring gradient
    x_var = x.clone().detach().requires_grad_(True)
    y_var = y.clone().detach().requires_grad_(True)
    
    # Get model predictions
    p, v, t, species = model(x_var, y_var)
    
    # Calculate derivatives
    dp_dx = torch.autograd.grad(p.sum(), x_var, create_graph=True)[0]
    dp_dy = torch.autograd.grad(p.sum(), y_var, create_graph=True)[0]
    
    dv_dx = torch.autograd.grad(v.sum(), x_var, create_graph=True)[0]
    dv_dy = torch.autograd.grad(v.sum(), y_var, create_graph=True)[0]
    
    # Continuity equation
    continuity = dv_dx + dv_dy
    
    # Momentum equations 
    momentum_x = v * dv_dx + dp_dx
    momentum_y = v * dv_dy + dp_dy

    # add energy equation also
    
    return torch.mean(continuity**2 + momentum_x**2 + momentum_y**2)

def species_transport_loss(model, x, y):
    x_var = x.clone().detach().requires_grad_(True)
    y_var = y.clone().detach().requires_grad_(True)

    _, v, _, species = model(x_var, y_var)
    
    # Calculate species transport equations
    species_loss = 0
    for i in range(species.shape[1]):
        ds_dx = torch.autograd.grad(species[:, i].sum(), x_var, create_graph=True)[0]
        ds_dy = torch.autograd.grad(species[:, i].sum(), y_var, create_graph=True)[0]
        
        convection = v * ds_dx  # Simplified convection term
        diffusion = ds_dx**2 + ds_dy**2  # Simplified diffusion term

        # check from papers
        # jet in hot co-flow problem
        
        species_loss += torch.mean(convection**2 + diffusion**2)
    
    return species_loss


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 2
df = df[:100]
df.columns = df.columns.str.strip()
print(df.columns)

# Convert DataFrame to PyTorch tensors and normalize data
x_train = torch.tensor(df['x-coordinate'].values[:, None], dtype=torch.float32)
y_train = torch.tensor(df['y-coordinate'].values[:, None], dtype=torch.float32)

data_train = torch.tensor(df[[
    'total-pressure', 'velocity-magnitude', 'total-temperature',
    'n2', 'h2', 'o2', 'h2o', 'oh', 'no', 'no2', 'n2o', 'nh3'
]].values, dtype=torch.float32)

data_mean = data_train.mean(dim=0)
data_std = data_train.std(dim=0)
data_train_normalized = (data_train - data_mean) / data_std


def predict_and_plot_corrected(model, inner_radius, outer_radius, channel_length, parameter='temperature'):
    # Create a grid of points
    x = np.linspace(0, channel_length, 100)
    y = np.linspace(-outer_radius, outer_radius, 100)
    X, Y = np.meshgrid(x, y)
    
    # Convert to torch tensors
    X_torch = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    Y_torch = torch.tensor(Y.flatten()[:, None], dtype=torch.float32)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        pressure, velocity, temperature, species = model(X_torch, Y_torch)
    
    # Convert predictions to numpy arrays
    pressure = pressure.numpy().reshape(X.shape)
    velocity = velocity.numpy().reshape(X.shape)
    temperature = temperature.numpy().reshape(X.shape)
    species = species.numpy().reshape((X.shape[0], X.shape[1], -1))
    
    # Create mask for ONLY the region outside the outer cylinder
    R = np.sqrt(Y**2)
    mask = (R <= outer_radius)  # Now including the inner region!
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot based on parameter choice
    if parameter == 'temperature':
        data = temperature
        title = 'Temperature Distribution (K)'
        cmap = 'RdYlBu_r'
    elif parameter == 'velocity':
        data = velocity
        title = 'Velocity Magnitude (m/s)'
        cmap = 'viridis'
    elif parameter == 'pressure':
        data = pressure
        title = 'Pressure Distribution (Pa)'
        cmap = 'jet'
    elif parameter.startswith('species_'):
        species_idx = int(parameter.split('_')[1])
        data = species[:, :, species_idx]
        title = f'Species {species_idx} Mass Fraction'
        cmap = 'YlOrRd'
    
    # Apply mask (only masking outside outer cylinder)
    data = np.ma.masked_array(data, ~mask)
    
    # Create contour plot
    im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto')
    plt.colorbar(im, ax=ax, label=title)
    
    # Plot cylinder boundaries (now just as reference lines)
    ax.plot(x, np.ones_like(x)*inner_radius, 'k--', linewidth=1, label='Inner cylinder wall')
    ax.plot(x, -np.ones_like(x)*inner_radius, 'k--', linewidth=1)
    ax.plot(x, np.ones_like(x)*outer_radius, 'k-', linewidth=1, label='Outer cylinder wall')
    ax.plot(x, -np.ones_like(x)*outer_radius, 'k-', linewidth=1)
    
    # Add flow direction arrows
    arrow_x = channel_length/2
    # Inner flow arrow
    ax.arrow(arrow_x-channel_length/8, 0, channel_length/4, 0, 
             head_width=inner_radius/2, head_length=channel_length/20, 
             fc='white', ec='black', alpha=0.5)
    # Outer flow arrows
    ax.arrow(arrow_x+channel_length/8, inner_radius*1.5, -channel_length/4, 0, 
             head_width=inner_radius/2, head_length=channel_length/20, 
             fc='white', ec='black', alpha=0.5)
    ax.arrow(arrow_x+channel_length/8, -inner_radius*1.5, -channel_length/4, 0, 
             head_width=inner_radius/2, head_length=channel_length/20, 
             fc='white', ec='black', alpha=0.5)
    
    ax.set_xlabel('Channel Length (m)')
    ax.set_ylabel('Radial Position (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def train_pinn(model, x_train, y_train, data_train, data_mean, data_std, epochs=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=500)
    
    # Loss weights
    w_data = 0.5
    w_physics = 0.25 
    w_bc = 0.25      
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Data loss
        # p_pred, v_pred, t_pred, species_pred = model(x_train, y_train)
        # data_loss = torch.mean((torch.cat([p_pred, v_pred, t_pred, species_pred], dim=1) - data_train)**2)

        outputs = model(x_train, y_train)
        outputs_concat = torch.cat(outputs, dim=1)
        outputs_normalized = (outputs_concat - data_mean) / data_std
        data_loss = torch.mean((outputs_normalized - data_train_normalized)**2)

        
        # Physics losses
        ns_loss = navier_stokes_loss(model, x_train, y_train)
        species_loss = species_transport_loss(model, x_train, y_train)
        
        # Boundary condition loss
        bc_loss = model.compute_boundary_loss(x_train, y_train)
        
        # Total loss
        total_loss = (
            w_data * data_loss + 
            w_physics * (ns_loss + species_loss) + 
            w_bc * bc_loss
        )
        
        total_loss.backward()
        optimizer.step()
        
        scheduler.step(total_loss)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Total Loss = {total_loss.item():.6f}, '
                  f'Data Loss = {data_loss.item():.6f}, '
                  f'Physics Loss = {(ns_loss + species_loss).item():.6f}, '
                  f'BC Loss = {bc_loss.item():.6f}')
            

# Define geometry and flow parameters
inner_radius = 0.005  # meters
outer_radius = 0.01   # meters
channel_length = 0.550  # meters

# Define inlet conditions
inner_inlet_velocity = 18.13  # m/s (from your sample data)
outer_inlet_velocity = 10.0  # m/s 

# Define species concentrations at inlets
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
])

outer_inlet_species = torch.tensor([
    0.79,  # N2 (air composition)
    0.0,   # H2
    0.21,  # O2
    0.0,   # H2O
    0.0,   # OH
    0.0,   # NO
    0.0,   # NO2
    0.0,   # N2O
    0.0    # NH3
])

# Get the inlet conditions from your data
inner_inlet_data = df[df['x-coordinate'] == df['x-coordinate'].min()].iloc[0]

inner_inlet_species = torch.tensor([
    inner_inlet_data['n2'],
    inner_inlet_data['h2'],
    inner_inlet_data['o2'],
    inner_inlet_data['h2o'],
    inner_inlet_data['oh'],
    inner_inlet_data['no'],
    inner_inlet_data['no2'],
    inner_inlet_data['n2o'],
    inner_inlet_data['nh3']
], dtype=torch.float32)

# Initialize model with actual data
model = FluidFlowPINN(
    inner_radius=0.005,  # Update with your actual geometry 
    outer_radius=0.01,
    channel_length=df['x-coordinate'].max(),
    inner_inlet_velocity=inner_inlet_data['velocity-magnitude'],
    outer_inlet_velocity=-10.0,  # Update with your actual outer flow velocity
    inner_inlet_species=inner_inlet_species,
    outer_inlet_species=outer_inlet_species
)


# Add the training call here:
train_pinn(
    model=model,
    x_train=x_train,
    y_train=y_train,
    data_train=data_train,
    data_mean=data_mean,
    data_std=data_std,
    epochs=10000
)

## increase epochs


# Create a function that takes boundary conditions as parameters
def predict_flow_distribution(
    trained_model,
    new_inner_radius,
    new_outer_radius,
    new_channel_length,
    new_inner_velocity,
    new_outer_velocity,
    new_inner_species_concentrations,
    new_outer_species_concentrations
):
    # Update the model with new boundary conditions
    trained_model.r_i = new_inner_radius
    trained_model.r_o = new_outer_radius
    trained_model.L = new_channel_length
    trained_model.v_i = new_inner_velocity
    trained_model.v_o = new_outer_velocity
    trained_model.species_i = new_inner_species_concentrations
    trained_model.species_o = new_outer_species_concentrations
    
    # Create visualization grid
    x = np.linspace(0, new_channel_length, 100)
    y = np.linspace(-new_outer_radius, new_outer_radius, 100)
    X, Y = np.meshgrid(x, y)
    
    # Convert to torch tensors
    X_torch = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    Y_torch = torch.tensor(Y.flatten()[:, None], dtype=torch.float32)
    
    # Get predictions for new conditions
    trained_model.eval()
    with torch.no_grad():
        pressure, velocity, temperature, species = trained_model(X_torch, Y_torch)
    
    return X, Y, pressure, velocity, temperature, species

# Example usage with new conditions:
new_conditions = {
    'inner_radius': 0.006,  # Different from training conditions
    'outer_radius': 0.012,
    'channel_length': 0.15,
    'inner_velocity': 25.0,  # Different inlet velocity
    'outer_velocity': -12.0,
    'inner_species': torch.tensor([
        3.0e-08,  # N2
        0.06,     # H2 (different concentration)
        1.6e-09,  # O2
        1.7e-09,  # H2O
        2.5e-16,  # OH
        2.7e-31,  # NO
        2.3e-10,  # NO2
        1.2e-11,  # N2O
        0.94      # NH3
    ]),
    'outer_species': torch.tensor([0.79, 0.0, 0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

# Get predictions for new conditions
X, Y, pressure, velocity, temperature, species = predict_flow_distribution(
    trained_model=model,
    new_inner_radius=new_conditions['inner_radius'],
    new_outer_radius=new_conditions['outer_radius'],
    new_channel_length=new_conditions['channel_length'],
    new_inner_velocity=new_conditions['inner_velocity'],
    new_outer_velocity=new_conditions['outer_velocity'],
    new_inner_species_concentrations=new_conditions['inner_species'],
    new_outer_species_concentrations=new_conditions['outer_species']
)

# Plot the results using the plotting function we created earlier
predict_and_plot_corrected(
    model=model,
    inner_radius=new_conditions['inner_radius'],
    outer_radius=new_conditions['outer_radius'],
    channel_length=new_conditions['channel_length'],
    parameter="temperature"
)


## Todo:


# how to import boundary conditions (2d or 3d)
# governing equations (jet in hot coflow combusiton problem, related equations and hw to import)
# loss functions
# 3 layers -> fuel -> hot-coflow -> air inlet
# how to import dataset
# axis symmetry condition
# model architecture optimizations