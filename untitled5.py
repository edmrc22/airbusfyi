import numpy as np
import matplotlib.pyplot as plt

def get_grid_sensor_indices(nodes, nx=10, ny=5, distance_threshold=None):
    """
    Generates a regular grid of sensor locations, snaps them to the closest real mesh nodes,
    and strips away any nodes lying on the outermost edges of the geometry.
    """
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    sensor_indices = []
    for point in grid_points:
        distances = np.linalg.norm(nodes - point, axis=1)
        closest_idx = np.argmin(distances)
        if distance_threshold is not None and distances[closest_idx] >= distance_threshold:
            continue
        sensor_indices.append(closest_idx)
            
    unique_indices = np.unique(sensor_indices)
    tol = 1e-4 
    
    interior_sensor_indices = []
    for idx in unique_indices:
        x, y = nodes[idx]
        on_outer_edge = (
            abs(x - x_min) < tol or 
            abs(x - x_max) < tol or 
            abs(y - y_min) < tol or 
            abs(y - y_max) < tol
        )
        if not on_outer_edge:
            interior_sensor_indices.append(idx)
            
    return np.array(interior_sensor_indices)


def get_internal_pool_indices(nodes, hole_radius=0.3):
    """
    Helper function to filter the entire global mesh down to an internal pool of nodes
    where physical hardware can safely be placed away from boundaries and the hole cutout.
    """
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    tol = 1e-4
    
    valid_interior_indices = []
    for idx, (x, y) in enumerate(nodes):
        on_outer_edge = (
            abs(x - x_min) < tol or abs(x - x_max) < tol or 
            abs(y - y_min) < tol or abs(y - y_max) < tol
        )
        inside_hole_buffer = np.sqrt(x**2 + y**2) < (hole_radius * 1.1)
        
        if not on_outer_edge and not inside_hole_buffer:
            valid_interior_indices.append(idx)
            
    return valid_interior_indices


def get_collocation_indices(nodes, hole_radius=0.3, num_rings=3, points_per_ring=20, margin_offset=0.08, points_per_margin=10):
    """
    Intelligently targets high-importance physical zones by placing points around 
    the central hole and slightly inside the margins of the outer plate boundary.
    """
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0]) 
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1]) 
    
    target_coords = []
    
    # 1. Hole rings
    ring_radii = hole_radius * np.linspace(1.15, 2, num_rings)
    for radius in ring_radii:
        angles = np.linspace(0, 2 * np.pi, points_per_ring, endpoint=False)
        for theta in angles:
            cx = radius * np.cos(theta)
            cy = radius * np.sin(theta)
            target_coords.append([cx, cy])
            
    # 2. Inset margin boundaries
    left_x   = x_min + margin_offset
    right_x  = x_max - margin_offset
    bottom_y = y_min + margin_offset
    top_y    = y_max - margin_offset
    
    y_span = np.linspace(y_min + margin_offset, y_max - margin_offset, points_per_margin)
    for y_val in y_span:
        target_coords.append([left_x, y_val])
        target_coords.append([right_x, y_val])
        
    points_per_margin_x = int(points_per_margin * (x_max - x_min) / (y_max - y_min))
    x_span = np.linspace(x_min + margin_offset, x_max - margin_offset, points_per_margin_x)
    for x_val in x_span:
        target_coords.append([x_val, bottom_y])
        target_coords.append([x_val, top_y])
        
    collo_indices = []
    for target in target_coords:
        distances = np.linalg.norm(nodes - target, axis=1)
        closest_idx = np.argmin(distances)
        collo_indices.append(closest_idx)
    
    # NEW: Add a dense background pool
    pool = get_internal_pool_indices(nodes, hole_radius=hole_radius)
    
    # Aim for a total of ~1,500 to 2,000 collocation points
    # This doesn't make the "experiment" less sparse because these 
    # points DON'T have ground truth stresses; they only have PDE residuals.
    num_background = 50 
    bg_indices = np.random.choice(pool, size=num_background, replace=False)
    
    combined_indices = np.concatenate([collo_indices, bg_indices])
    return np.unique(combined_indices)


def process_dataset_and_visualize(filepath="pinn_fem_ground_truth.npz", sample_idx=12, 
                                  sensor_mode="random_dynamic", num_random_sensors=12, 
                                  augment_factor=5, nx_sensors=15, ny_sensors=8):
    """
    1. Routes data extraction through 3 user-selected configuration pipelines.
    2. Packages dynamic spatial matrices matching augmented and un-augmented structures.
    3. Outputs tensor geometries ready for automated neural model reading and visualizes layout.
    """
    print(f"Loading raw dataset from {filepath}...")
    raw_data = np.load(filepath, allow_pickle=True)
    nodes = raw_data['nodes']
    elements = raw_data['elements']
    all_stresses = raw_data['stresses']  # Shape: (200, Num_Nodes, 3)
    num_simulations = len(all_stresses)
    
    avg_mesh_spacing = (np.max(nodes[:, 0]) - np.min(nodes[:, 0])) / 40 
    collo_indices = get_collocation_indices(nodes)
    collo_coords = nodes[collo_indices]
    
    # Initialize explicit layout routing variables
    mode = sensor_mode.lower()
    np.random.seed(42)  # Lock global seed for generation consistency
    
    # =========================================================================
    # STEP 1: CONFIGURATION ROUTING PIPELINES
    # =========================================================================
    if mode == "grid":
        print(f"Routing Pipeline: Extracting UNIFORM GRID tracking layout.")
        sensor_indices = get_grid_sensor_indices(nodes, nx=nx_sensors, ny=ny_sensors, distance_threshold=avg_mesh_spacing)
        
        # Broadcast single coordinate mask over simulation count dimensions
        sensor_coords_all = np.repeat(nodes[sensor_indices][np.newaxis, :, :], num_simulations, axis=0)
        sensor_stresses_all = all_stresses[:, sensor_indices, :]
        collo_stresses_all = all_stresses[:, collo_indices, :]
        
        # For visualization extraction:
        visual_sensor_coords = nodes[sensor_indices]
        
    elif mode == "random_static":
        print(f"Routing Pipeline: Extracting {num_random_sensors} STATIC RANDOM interior sensors.")
        pool = get_internal_pool_indices(nodes)
        sensor_indices = np.random.choice(pool, size=num_random_sensors, replace=False)
        
        # Broadcast chosen static indices layout uniformly
        sensor_coords_all = np.repeat(nodes[sensor_indices][np.newaxis, :, :], num_simulations, axis=0)
        sensor_stresses_all = all_stresses[:, sensor_indices, :]
        collo_stresses_all = all_stresses[:, collo_indices, :]
        
        # For visualization extraction:
        visual_sensor_coords = nodes[sensor_indices]
        
    elif mode == "random_dynamic":
        print(f"Routing Pipeline: Generating {num_random_sensors} DYNAMIC RANDOM sensors with Augment Factor = {augment_factor}.")
        pool = get_internal_pool_indices(nodes)
        
        dynamic_coords = []
        dynamic_sensor_stresses = []
        dynamic_collo_stresses = []
        
        for i in range(num_simulations):
            for aug in range(augment_factor):
                chosen_idx = np.random.choice(pool, size=num_random_sensors, replace=False)
                
                dynamic_coords.append(nodes[chosen_idx])
                dynamic_sensor_stresses.append(all_stresses[i, chosen_idx, :])
                dynamic_collo_stresses.append(all_stresses[i, collo_indices, :])
                
        sensor_coords_all = np.array(dynamic_coords)
        sensor_stresses_all = np.array(dynamic_sensor_stresses)
        collo_stresses_all = np.array(dynamic_collo_stresses)
        
        # For visualization extraction: Find the first augmented sub-sample row corresponding to sample_idx
        visual_row_idx = sample_idx * augment_factor
        visual_sensor_coords = sensor_coords_all[visual_row_idx]
        
    else:
        raise ValueError("Invalid sensor_mode! Choose from: 'grid', 'random_static', or 'random_dynamic'.")
        
    print(f"\n--- Processed Array Matrix Dimension Statistics ---")
    print(f"Input Sensor Coordinates Matrix Shape: {sensor_coords_all.shape}")
    print(f"Input Sensor Stresses Matrix Shape:    {sensor_stresses_all.shape}")
    print(f"Target Collocation Stresses Shape:     {collo_stresses_all.shape}")
    
    # Save processed components directly to disk
    np.savez(
        "processed_pinn_training_data.npz",
        sensor_coords=sensor_coords_all,
        sensor_stresses=sensor_stresses_all,
        collo_coords=collo_coords,
        collo_stresses=collo_stresses_all
    )
    print("Saved output components safely to 'processed_pinn_training_data.npz'!")
    
    # =========================================================================
    # STEP 2: PLOT THE VERIFICATION CASE (FIXED RESOLUTION)
    # =========================================================================
    sample_stresses = all_stresses[sample_idx]
    metadata = raw_data['metadata'][sample_idx]
    clamped_edge = metadata['clamped_edge']
    loaded_edge = metadata['loaded_edge']
    pressure = metadata['applied_pressure']

    print(f"\nRendering Visual Layout Verification for Sample {sample_idx}...")
    
    triangles = []
    for cell in elements:
        triangles.append([cell[0], cell[1], cell[2]])
        triangles.append([cell[0], cell[2], cell[3]])
    triangles = np.array(triangles)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    sig_xx = sample_stresses[:, 0]
    tc = ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, sig_xx, levels=50, cmap='jet')
    plt.colorbar(tc, ax=ax, label=r'Ground Truth $\sigma_{xx}$ (MPa)')
    ax.triplot(nodes[:, 0], nodes[:, 1], triangles, color='black', alpha=0.15, linewidth=0.5)

    # FIXED: Plotting specific sliced coordinates derived safely inside the pipelines above
    ax.scatter(visual_sensor_coords[:, 0], visual_sensor_coords[:, 1], color='black', s=45, zorder=5, 
               label=f'Saved Sensors ({len(visual_sensor_coords)} - Mode: {sensor_mode})')
    
    ax.scatter(collo_coords[:, 0], collo_coords[:, 1], color='black', marker='x', s=50, lw=1.5, zorder=6, 
               label=f'Saved Collocation Tensors ({len(collo_coords)})')

    # Visual Boundary Condition Helpers
    bounds = {
        'left':   {'x': np.min(nodes[:, 0]), 'y': 0, 'dx': -1, 'dy': 0},
        'right':  {'x': np.max(nodes[:, 0]), 'y': 0, 'dx': 1,  'dy': 0},
        'bottom': {'x': np.mean(nodes[:, 0]), 'y': np.min(nodes[:, 1]), 'dx': 0, 'dy': -1},
        'top':    {'x': np.mean(nodes[:, 0]), 'y': np.max(nodes[:, 1]), 'dx': 0, 'dy': 1}
    }

    if clamped_edge in bounds:
        b = bounds[clamped_edge]
        if clamped_edge in ['left', 'right']:
            ax.axvline(x=b['x'], color='red', linewidth=6, label='Clamped Edge')
        else:
            ax.axhline(y=b['y'], color='red', linewidth=6, label='Clamped Edge')

    if loaded_edge in bounds:
        b = bounds[loaded_edge]
        direction_multiplier = -1 if pressure > 0 else 1 
        if loaded_edge in ['left', 'right']:
            y_vals = np.linspace(np.min(nodes[:, 1]), np.max(nodes[:, 1]), 6)
            x_vals = np.full_like(y_vals, b['x'])
            ax.quiver(x_vals, y_vals, b['dx'] * direction_multiplier, b['dy'], color='blue', scale=10, width=0.005, label=f'Load ({pressure:.1f} MPa)')
        else:
            x_vals = np.linspace(np.min(nodes[:, 0]), np.max(nodes[:, 0]), 10)
            y_vals = np.full_like(x_vals, b['y'])
            ax.quiver(x_vals, y_vals, b['dx'], b['dy'] * direction_multiplier, color='blue', scale=10, width=0.005, label=f'Load ({pressure:.1f} MPa)')

    ax.set_title(f"Sample {sample_idx} Layout Verification ({sensor_mode.upper()} Mode)\nClamped: {clamped_edge.upper()} | Loaded: {loaded_edge.upper()} | P = {pressure:.1f} MPa", fontsize=13)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_dataset_and_visualize(
        filepath="pinn_fem_ground_truth.npz", 
        sample_idx=32, 
        sensor_mode="random_static",       # Options: "grid", "random_static", or "random_dynamic"
        num_random_sensors=26,              # Active for random modes
        augment_factor=5,                   # Active for random_dynamic mode
        nx_sensors=10,                      # Active for grid mode
        ny_sensors=6                        # Active for grid mode
    )