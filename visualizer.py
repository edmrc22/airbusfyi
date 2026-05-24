import numpy as np
import matplotlib.pyplot as plt

def visualize_sensors_and_full_stress(filepath="pinn_fem_ground_truth.npz", sample_idx=0):
    print(f"Loading data from {filepath}...")
    
    try:
        data = np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"ERROR: '{filepath}' not found. You must run 01_generate_fem_data.py first.")
        return

    nodes = data['nodes']
    elements = data['elements']
    stresses = data['stresses'][sample_idx]
    metadata = data['metadata'][sample_idx]
    sensor_indices = data['sensor_indices']
    
    pressure = metadata['applied_pressure']

    print(f"--- Sample {sample_idx} Configuration ---")
    print(f"Clamped Edge: {metadata['clamped_edge'].upper()} (Fixed)")
    print(f"Loaded Edge:  {metadata['loaded_edge'].upper()} (Fixed)")
    print(f"Applied Load: {pressure:.2f} MPa")

    # Convert quads to triangles for matplotlib compatibility
    triangles = []
    for cell in elements:
        triangles.append([cell[0], cell[1], cell[2]])
        triangles.append([cell[0], cell[2], cell[3]])
    triangles = np.array(triangles)

    # 1. Setup a 1x3 Visualization Matrix
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"Sample {sample_idx} Physical State | P = {pressure:.2f} MPa", fontsize=18)
    
    stress_components = [
        (stresses[:, 0], r'Normal Stress $\sigma_{xx}$ (MPa)'),
        (stresses[:, 1], r'Normal Stress $\sigma_{yy}$ (MPa)'),
        (stresses[:, 2], r'Shear Stress $\tau_{xy}$ (MPa)')
    ]

    for i, ax in enumerate(axes):
        sig_comp, title = stress_components[i]
        
        # Plot Stress Field
        tc = ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, sig_comp, levels=50, cmap='jet')
        plt.colorbar(tc, ax=ax, label=title)
        ax.triplot(nodes[:, 0], nodes[:, 1], triangles, color='black', alpha=0.15, linewidth=0.5)

        # Draw Clamped Edge (Left)
        ax.axvline(x=np.min(nodes[:, 0]), color='red', linewidth=6, label='Clamped Root')

        # Draw Loaded Edge (Top)
        direction_multiplier = -1 if pressure > 0 else 1 
        x_vals = np.linspace(np.min(nodes[:, 0]), np.max(nodes[:, 0]), 10)
        y_vals = np.full_like(x_vals, np.max(nodes[:, 1]))
        ax.quiver(x_vals, y_vals, 0, 1 * direction_multiplier, 
                  color='blue', scale=10, width=0.005, label='Applied Load')

        # Draw Fixed Sensors
        sensor_nodes = nodes[sensor_indices]
        ax.scatter(sensor_nodes[:, 0], sensor_nodes[:, 1], 
                   color='lime', edgecolor='black', s=60, zorder=5, 
                   label='Hardware Sensors')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_aspect('equal')
        if i == 0:
            ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    filename = f"sample_{sample_idx}_full_tensor_visualization.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Success. Comprehensive plot saved to: {filename}")

if __name__ == "__main__":
    visualize_sensors_and_full_stress(filepath="pinn_fem_ground_truth.npz", sample_idx=450)