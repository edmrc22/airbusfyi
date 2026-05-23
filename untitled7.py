import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 1. DEEPONET DEFINITION (MATCHING TRAINING ARCHITECTURE)
# =========================================================================
class DeepONetPlatePINN(nn.Module):
    def __init__(self, num_sensors=26, hidden_dim=256, basis_functions=128):
        super().__init__()
        self.num_sensors = num_sensors
        self.basis_functions = basis_functions
        branch_input_dim = num_sensors * 5  # 2 coordinates + 3 stress values per sensor = 5
        
        # Consolidates layers into 'branch' to align with saved training weights
        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, basis_functions * 3)
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, basis_functions * 3)
        )
        
    def forward(self, coords, sensor_state):
        batch_size = sensor_state.shape[0]
        
        if len(coords.shape) == 2:
            num_collo = coords.shape[0]
            coords_input = coords.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            num_collo = coords.shape[1]
            coords_input = coords

        b_out = self.branch(sensor_state).view(batch_size, 3, self.basis_functions)
        t_out = self.trunk(coords_input.reshape(-1, 2)).view(batch_size, num_collo, 3, self.basis_functions)
        
        # Uses fast batched matrix multiplication trick
        out = torch.einsum('bsf,bcsf->bcs', b_out, t_out)
        return out

# =========================================================================
# 2. BOUNDARY CONDITION HELPER
# =========================================================================
def draw_boundary_conditions(ax, nodes, clamped_edge, loaded_edge, pressure):
    bounds = {
        'left':    {'x': np.min(nodes[:, 0]), 'y': 0, 'dx': -1, 'dy': 0},
        'right':   {'x': np.max(nodes[:, 0]), 'y': 0, 'dx': 1,  'dy': 0},
        'bottom': {'x': np.mean(nodes[:, 0]), 'y': np.min(nodes[:, 1]), 'dx': 0, 'dy': -1},
        'top':    {'x': np.mean(nodes[:, 0]), 'y': np.max(nodes[:, 1]), 'dx': 0, 'dy': 1}
    }

    if clamped_edge in bounds:
        b = bounds[clamped_edge]
        if clamped_edge in ['left', 'right']:
            ax.axvline(x=b['x'], color='red', linewidth=6, alpha=0.5, label='Clamped')
        else:
            ax.axhline(y=b['y'], color='red', linewidth=6, alpha=0.5, label='Clamped')

    if loaded_edge in bounds:
        b = bounds[loaded_edge]
        direction_multiplier = -1 if pressure > 0 else 1 
        if loaded_edge in ['left', 'right']:
            y_vals = np.linspace(np.min(nodes[:, 1]), np.max(nodes[:, 1]), 6)
            x_vals = np.full_like(y_vals, b['x'])
            ax.quiver(x_vals, y_vals, b['dx'] * direction_multiplier, b['dy'], 
                      color='blue', scale=10, width=0.008, zorder=10, label=f'Load ({pressure:.1f})')
        else:
            x_vals = np.linspace(np.min(nodes[:, 0]), np.max(nodes[:, 0]), 10)
            y_vals = np.full_like(x_vals, b['y'])
            ax.quiver(x_vals, y_vals, b['dx'], b['dy'] * direction_multiplier, 
                      color='blue', scale=10, width=0.008, zorder=10, label=f'Load ({pressure:.1f})')

# =========================================================================
# 3. FULL FEM-MAPPED VISUALIZATION
# =========================================================================
def plot_pinn_on_fem_mesh(processed_idx=122, target_component=0, 
                          raw_filepath="pinn_fem_ground_truth.npz", 
                          processed_filepath="processed_pinn_training_data.npz",
                          augment_factor=5):
    
    # --- THE AUTO-MAPPING FIX ---
    # Automatically convert the processed row index (0-999) to the original raw FEM index (0-199)
    sample_idx = processed_idx // augment_factor  # Integer division
    
    print(f"Mapping Processed Row {processed_idx} to Raw FEM Simulation Sample {sample_idx}...")
    
    # 1. Load Raw FEM Data
    raw_data = np.load(raw_filepath, allow_pickle=True)
    nodes = raw_data['nodes']
    elements = raw_data['elements']
    true_stresses = raw_data['stresses'][sample_idx]
    
    metadata = raw_data['metadata'][sample_idx]
    clamped_edge = metadata['clamped_edge']
    loaded_edge = metadata['loaded_edge']
    pressure = metadata['applied_pressure']

    # 2. Convert FEM Quads to Triangles
    triangles = []
    for cell in elements:
        triangles.append([cell[0], cell[1], cell[2]])
        triangles.append([cell[0], cell[2], cell[3]])
    triangles = np.array(triangles)

    # 3. Load Processed PINN Data using the exact processed row index
    proc_data = np.load(processed_filepath)
    c_min, c_max = np.min(proc_data['collo_coords'], axis=0), np.max(proc_data['collo_coords'], axis=0)
    target_min, target_max = proc_data['collo_stresses'].min(), proc_data['collo_stresses'].max()
    
    # Grab the exact sensor augmentation layout used during training
    sensor_state_raw = proc_data['sensor_stresses'][processed_idx] 
    sensor_coords_raw = proc_data['sensor_coords'][processed_idx]

    # Normalize inputs
    def scale_c(c):
        return 2.0 * (c - c_min) / (c_max - c_min + 1e-8) - 1.0

    norm_nodes = torch.tensor(scale_c(nodes), dtype=torch.float32).unsqueeze(0)
    
    s_min, s_max = proc_data['sensor_stresses'].min(), proc_data['sensor_stresses'].max()
    norm_sensor_stresses = (sensor_state_raw - s_min) / (s_max - s_min + 1e-8)
    norm_sensor_coords = scale_c(sensor_coords_raw)
    
    combined_sensor_state = np.concatenate([norm_sensor_coords, norm_sensor_stresses], axis=-1)
    sensor_tensor = torch.tensor(combined_sensor_state, dtype=torch.float32).view(1, -1)

    # 4. Load Model & Predict
    model = DeepONetPlatePINN(num_sensors=26, hidden_dim=256, basis_functions=128)
    model.load_state_dict(torch.load("deeponet_pinn_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    
    with torch.no_grad():
        pred_norm = model(norm_nodes, sensor_tensor)
        
    # Un-normalize predictions back to physical MPa
    pred_phys = pred_norm[0].numpy() * (target_max - target_min + 1e-8) + target_min
    
    # Extract structural components (0=XX, 1=YY, 2=XY)
    actual_comp = true_stresses[:, target_component]
    pred_comp = pred_phys[:, target_component]
    error_comp = np.abs(actual_comp - pred_comp)

    # =========================================================================
    # PLOTTING RENDERING
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    comp_names = ["SIGMA_XX", "SIGMA_YY", "SIGMA_XY"]
    
    vmin = min(actual_comp.min(), pred_comp.min())
    vmax = max(actual_comp.max(), pred_comp.max())
    levels = np.linspace(vmin, vmax, 50)

    # Plot 1: Ground Truth
    tc1 = axes[0].tricontourf(nodes[:, 0], nodes[:, 1], triangles, actual_comp, levels=levels, cmap='jet')
    axes[0].triplot(nodes[:, 0], nodes[:, 1], triangles, color='black', alpha=0.15, linewidth=0.5)
    axes[0].scatter(sensor_coords_raw[:, 0], sensor_coords_raw[:, 1], color='white', edgecolor='black', s=80, marker='^', zorder=5, label='Sensors')
    axes[0].set_title(f"Ground Truth: {comp_names[target_component]}\nSample {sample_idx} (Proc {processed_idx})", fontweight='bold')
    
    # Plot 2: PINN Prediction
    tc2 = axes[1].tricontourf(nodes[:, 0], nodes[:, 1], triangles, pred_comp, levels=levels, cmap='jet')
    axes[1].triplot(nodes[:, 0], nodes[:, 1], triangles, color='black', alpha=0.15, linewidth=0.5)
    axes[1].scatter(sensor_coords_raw[:, 0], sensor_coords_raw[:, 1], color='white', edgecolor='black', s=80, marker='^', zorder=5)
    axes[1].set_title(f"PINN Prediction: {comp_names[target_component]}", fontweight='bold')

    # Plot 3: Absolute Deviation Error
    tc3 = axes[2].tricontourf(nodes[:, 0], nodes[:, 1], triangles, error_comp, levels=50, cmap='Reds')
    axes[2].triplot(nodes[:, 0], nodes[:, 1], triangles, color='black', alpha=0.15, linewidth=0.5)
    axes[2].set_title(f"Absolute Error (|True - Pred|)", fontweight='bold')

    for ax in axes:
        draw_boundary_conditions(ax, nodes, clamped_edge, loaded_edge, pressure)
        ax.set_aspect('equal')
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.legend(loc='upper right', fontsize=9)

    fig.colorbar(tc1, ax=axes[0], label='Stress (MPa)')
    fig.colorbar(tc2, ax=axes[1], label='Stress (MPa)')
    fig.colorbar(tc3, ax=axes[2], label='Error (MPa)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # You can now cleanly pass any index from 0 to 999!
    plot_pinn_on_fem_mesh(processed_idx=203, target_component=0)

