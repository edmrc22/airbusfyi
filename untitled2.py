import numpy as np
import matplotlib.pyplot as plt

def visualize_sample(filepath="pinn_fem_ground_truth.npz", sample_idx=0):
    print(f"Loading data from {filepath}...")
    
    # 1. Load the dataset
    data = np.load(filepath, allow_pickle=True)
    nodes = data['nodes']
    elements = data['elements']
    stresses = data['stresses'][sample_idx]
    
    # Extract metadata (allow_pickle=True is required to read arrays of dictionaries)
    metadata = data['metadata'][sample_idx]
    
    clamped_edge = metadata['clamped_edge']
    loaded_edge = metadata['loaded_edge']
    pressure = metadata['applied_pressure']

    print(f"--- Sample {sample_idx} Configuration ---")
    print(f"Clamped Edge: {clamped_edge.upper()}")
    print(f"Loaded Edge:  {loaded_edge.upper()}")
    print(f"Applied Load: {pressure:.2f} MPa")

    # 2. Convert quad elements to triangles for matplotlib compatibility
    triangles = []
    for cell in elements:
        triangles.append([cell[0], cell[1], cell[2]])
        triangles.append([cell[0], cell[2], cell[3]])
    triangles = np.array(triangles)

    # 3. Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot the Stress Field (sigma_xx)
    sig_xx = stresses[:, 0]
    tc = ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, sig_xx, levels=50, cmap='jet')
    plt.colorbar(tc, ax=ax, label=r'Ground Truth $\sigma_{xx}$ (MPa)')

    # Overlay the mesh grid faintly
    ax.triplot(nodes[:, 0], nodes[:, 1], triangles, color='black', alpha=0.15, linewidth=0.5)

    # 4. Helper logic to draw BCs and Loads visually
    bounds = {
        'left':   {'x': np.min(nodes[:, 0]), 'y': 0, 'dx': -1, 'dy': 0, 'align': 'right'},
        'right':  {'x': np.max(nodes[:, 0]), 'y': 0, 'dx': 1,  'dy': 0, 'align': 'left'},
        'bottom': {'x': np.mean(nodes[:, 0]), 'y': np.min(nodes[:, 1]), 'dx': 0, 'dy': -1, 'align': 'top'},
        'top':    {'x': np.mean(nodes[:, 0]), 'y': np.max(nodes[:, 1]), 'dx': 0, 'dy': 1, 'align': 'bottom'}
    }

    # Draw Clamped Edge (Thick Red Line)
    if clamped_edge in bounds:
        b = bounds[clamped_edge]
        if clamped_edge in ['left', 'right']:
            ax.axvline(x=b['x'], color='red', linewidth=6, label='Clamped Boundary')
        else:
            ax.axhline(y=b['y'], color='red', linewidth=6, label='Clamped Boundary')

    # Draw Loaded Edge (Blue Arrows)
    if loaded_edge in bounds:
        b = bounds[loaded_edge]
        # Determine arrow direction (positive pressure pushes IN, negative pulls OUT)
        direction_multiplier = -1 if pressure > 0 else 1 
        
        # Draw a set of arrows along the loaded edge
        if loaded_edge in ['left', 'right']:
            y_vals = np.linspace(np.min(nodes[:, 1]), np.max(nodes[:, 1]), 6)
            x_vals = np.full_like(y_vals, b['x'])
            ax.quiver(x_vals, y_vals, b['dx'] * direction_multiplier, b['dy'], 
                      color='blue', scale=10, width=0.005, label=f'Load ({pressure:.1f})')
        else:
            x_vals = np.linspace(np.min(nodes[:, 0]), np.max(nodes[:, 0]), 10)
            y_vals = np.full_like(x_vals, b['y'])
            ax.quiver(x_vals, y_vals, b['dx'], b['dy'] * direction_multiplier, 
                      color='blue', scale=10, width=0.005, label=f'Load ({pressure:.1f})')

    # Formatting
    ax.set_title(f"Sample {sample_idx} Visualization\nClamped: {clamped_edge.upper()} | Loaded: {loaded_edge.upper()} | P = {pressure:.1f}", fontsize=14)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change the sample_idx here to look at different cases in your dataset (0 to 49)
    visualize_sample(filepath="pinn_fem_ground_truth.npz", sample_idx=134)