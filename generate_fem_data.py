import os
os.environ["FELUPE_VERBOSE"] = "false"  # Kills the ASCII logo and global verbosity

import numpy as np
import felupe as fem
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

# --- Configuration & Material Properties ---
NUM_SAMPLES = 500           # Required volume for DeepONet training
E = 210000.0                # Young's Modulus for Steel/Aluminum (MPa)
NU = 0.3                    # Poisson's ratio
NUM_SENSORS = 50            # The number of fixed physical sensors

# Plate dimensions for off-center hole
h = 1.0                    
L_left = 1.0               
L_right = 3.0              
r = 0.3                    

OUTPUT_FILE = "pinn_fem_ground_truth.npz"

def create_offcenter_hole_mesh():
    """Generates a 2D plate with an off-center hole using Felupe quad elements."""
    print("Generating asymmetric geometry and quad mesh natively with Felupe...")
    phi = np.linspace(1, 0.5, 21) * np.pi / 2
    line = fem.mesh.Line(n=21)
    curve = line.copy(points=r * np.vstack([np.cos(phi), np.sin(phi)]).T)
    top = line.copy(points=np.vstack([np.linspace(0, h, 21), np.linspace(h, h, 21)]).T)
    
    face = curve.fill_between(top, n=np.linspace(0, 1, 21) ** 1.3 * 2 - 1)
    quarter_hole = fem.mesh.concatenate([face, face.mirror(normal=[-1, 1, 0])])
    
    rect_right = fem.mesh.Rectangle(a=(h, 0), b=(L_right, h), n=21)
    quarter_right = fem.mesh.concatenate([quarter_hole, rect_right])
    half_right = fem.mesh.concatenate([quarter_right, quarter_right.mirror(normal=[0, 1, 0])])
    
    half_left = fem.mesh.concatenate([quarter_hole, quarter_hole.mirror(normal=[0, 1, 0])])
    half_left_mirrored = half_left.mirror(normal=[1, 0, 0])
    
    full_mesh = fem.mesh.concatenate([half_left_mirrored, half_right])
    return full_mesh.sweep(decimals=5)

def main():
    mesh = create_offcenter_hole_mesh()
    region = fem.RegionQuad(mesh)
    displacement = fem.Field(region, dim=2)
    field = fem.FieldContainer([displacement])
    
    umat = fem.LinearElasticPlaneStress(E=E, nu=NU)
    solid = fem.SolidBody(umat, field)
    
    n_nodes = len(mesh.points)
    all_stresses = np.zeros((NUM_SAMPLES, n_nodes, 3))
    metadata = []
    
    # -------------------------------------------------------------------------
    # CRITICAL UPGRADE: HYBRID SENSOR ARRAY (RING + ROOT ANCHORS + FIELD)
    # -------------------------------------------------------------------------
    print("Designing Hybrid Sensor Array (Ring + Root Anchors + Field)...")
    
    # 1. THE PERIMETER RING (Hole Gradient Capture)
    NUM_RING = 15
    ring_radius = r + 0.05  
    
    theta = np.linspace(0, 2 * np.pi, NUM_RING, endpoint=False)
    ideal_ring_x = ring_radius * np.cos(theta)
    ideal_ring_y = ring_radius * np.sin(theta)
    ideal_ring_coords = np.column_stack([ideal_ring_x, ideal_ring_y])
    
    tree = cKDTree(mesh.points)
    _, ring_indices = tree.query(ideal_ring_coords)
    ring_indices = np.unique(ring_indices)
    
    # 1.5 THE ROOT ANCHORS (Max Bending Moment Capture)
    # We place 4 sensors specifically to monitor the extreme top/bottom left corners.
    # We stay 0.05 units away from the exact corner to avoid the mathematical singularity.
    ideal_root_coords = np.array([
        [-0.95,  0.95], [-0.95,  0.80],  # Top-left zone
        [-0.95, -0.95], [-0.95, -0.80]   # Bottom-left zone
    ])
    _, root_indices = tree.query(ideal_root_coords)
    root_indices = np.unique(root_indices)

    # 2. THE GLOBAL FIELD (Low Gradient Capture)
    NUM_FIELD = NUM_SENSORS - len(ring_indices) - len(root_indices)
    
    # Strict safe zone for K-Means: away from edges, outside the ring, and away from root anchors
    field_mask = (
        (mesh.points[:, 0] > -L_left + 0.1) & 
        (mesh.points[:, 0] < L_right - 0.1) &
        (mesh.points[:, 1] > -h + 0.1) & 
        (mesh.points[:, 1] < h - 0.1) &
        ((mesh.points[:, 0]**2 + mesh.points[:, 1]**2) > (ring_radius + 0.1)**2)
    )
    
    valid_field_indices = np.where(field_mask)[0]
    valid_field_points = mesh.points[valid_field_indices]
    
    kmeans = KMeans(n_clusters=NUM_FIELD, n_init=10, random_state=42)
    kmeans.fit(valid_field_points)
    ideal_field_coords = kmeans.cluster_centers_
    
    field_tree = cKDTree(valid_field_points)
    _, closest_local_indices = field_tree.query(ideal_field_coords)
    field_indices = valid_field_indices[closest_local_indices]
    field_indices = np.unique(field_indices)
    
    # 3. ASSEMBLY & VERIFICATION
    sensor_indices = np.concatenate([ring_indices, root_indices, field_indices])
    sensor_indices = np.unique(sensor_indices)
    
    while len(sensor_indices) < NUM_SENSORS:
        available = np.setdiff1d(valid_field_indices, sensor_indices)
        fillers = np.random.choice(available, size=(NUM_SENSORS - len(sensor_indices)), replace=False)
        sensor_indices = np.concatenate([sensor_indices, fillers])
        sensor_indices = np.unique(sensor_indices)

    # -------------------------------------------------------------------------
    # BOUNDARY CONDITIONS & FEM SOLVER
    # -------------------------------------------------------------------------
    eps = 1e-3
    clamped_mask = mesh.points[:, 0] < -L_left + eps  # Left Edge is ALWAYS clamped
    loaded_mask = mesh.points[:, 1] > h - eps         # Top Edge is ALWAYS loaded
    
    boundaries = {"clamp": fem.Boundary(displacement, mask=clamped_mask)}
    region_boundary = fem.RegionQuadBoundary(mesh, mask=loaded_mask)
    field_boundary = fem.FieldContainer([fem.Field(region_boundary, dim=2)])
    
    print(f"Mesh generated with {n_nodes} nodes.")
    print(f"{NUM_SENSORS} permanent sensor locations chosen ({len(ring_indices)} in ring, {NUM_SENSORS - len(ring_indices)} in field).")
    print(f"Assembling and solving {NUM_SAMPLES} load cases...")
    
    for i in range(NUM_SAMPLES):
        displacement.values[:] = 0.0
        
        load_pressure = np.random.uniform(-150.0, 150.0)
        
        load = fem.SolidBodyPressure(field_boundary, pressure=load_pressure)
        step = fem.Step(items=[solid, load], boundaries=boundaries)
        
        # STRICTLY REQUIRED: verbose=False to prevent terminal flood and slowdown
        fem.Job(steps=[step]).evaluate(verbose=False)
        
        stress_nodes = fem.tools.project(solid.evaluate.stress(), region)
        
        all_stresses[i] = np.column_stack([stress_nodes[:, 0, 0], stress_nodes[:, 1, 1], stress_nodes[:, 0, 1]])
        
        meta = {
            'sample_id': i, 
            'clamped_edge': 'left',          
            'loaded_edge': 'top',            
            'applied_pressure': load_pressure
        }
        metadata.append(meta)
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{NUM_SAMPLES} load cases...")

    # --- CRITICAL AUDIT: EMPIRICAL BOUNDS ---
    all_pressures = [m['applied_pressure'] for m in metadata]
    max_p = np.max(all_pressures)
    min_p = np.min(all_pressures)
    
    print("\n" + "="*50)
    print("DATASET GENERATION SUMMARY")
    print("="*50)
    print(f"Total Simulations: {NUM_SAMPLES}")
    print(f"Maximum Applied Load: {max_p:.2f} MPa")
    print(f"Minimum Applied Load: {min_p:.2f} MPa")
    print("="*50 + "\n")

    print(f"Saving dataset to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        nodes=mesh.points, 
        elements=mesh.cells, 
        stresses=all_stresses, 
        metadata=metadata,
        sensor_indices=sensor_indices 
    )
    print("Generation complete!")

if __name__ == "__main__":
    main()