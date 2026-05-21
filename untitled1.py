import numpy as np
import matplotlib.pyplot as plt
import felupe as fem

# --- Configuration & Material Properties ---
NUM_SAMPLES = 200           
E = 210000.0               # Young's Modulus (MPa)
NU = 0.3                   # Poisson's ratio

# Plate dimensions for off-center hole
h = 1.0                    # Half-height of the plate (y goes from -1 to 1)
L_left = 1.0               # Left edge distance from hole center
L_right = 3.0              # Right edge distance from hole center
r = 0.3                    # Hole radius

OUTPUT_FILE = "pinn_fem_ground_truth.npz"

# COLLOCATION POINTS -> DONT FEED MODEL 5000 NODES
# WE PICK 50 COLLOCATION POINTS FOR PROBLEM

# TRAINING: FEED MODEL N MANY RANDOM SENSOR POINTS X,Y,STRESS

# TRAINING: REAL STRESS RESUTLS AT COLLOCATION POINTS
# LOSS FUNCTION BASED ON PHYSICS 


# SENSOR POINTS -> RANDOM SAMPLE 

def create_offcenter_hole_mesh():
    """Generates a 2D plate with an off-center hole using purely Felupe quad elements."""
    print("Generating asymmetric geometry and quad mesh natively with Felupe...")
    
    # 1. Base hole quadrant (Top-Right of the hole)
    phi = np.linspace(1, 0.5, 21) * np.pi / 2
    line = fem.mesh.Line(n=21)
    curve = line.copy(points=r * np.vstack([np.cos(phi), np.sin(phi)]).T)
    top = line.copy(points=np.vstack([np.linspace(0, h, 21), np.linspace(h, h, 21)]).T)

    face = curve.fill_between(top, n=np.linspace(0, 1, 21) ** 1.3 * 2 - 1)
    
    # Create the complete 1x1 quadrant containing the hole arc
    quarter_hole = fem.mesh.concatenate([face, face.mirror(normal=[-1, 1, 0])])
    
    # 2. Build the right half (extends to L_right = 3.0)
    rect_right = fem.mesh.Rectangle(a=(h, 0), b=(L_right, h), n=21)
    quarter_right = fem.mesh.concatenate([quarter_hole, rect_right])
    half_right = fem.mesh.concatenate([quarter_right, quarter_right.mirror(normal=[0, 1, 0])])
    
    # 3. Build the left half (extends to L_left = 1.0, so no extra rectangle needed)
    half_left = fem.mesh.concatenate([quarter_hole, quarter_hole.mirror(normal=[0, 1, 0])])
    half_left_mirrored = half_left.mirror(normal=[1, 0, 0])
    
    # 4. Concatenate the asymmetric halves and sweep to merge coincident nodes
    full_mesh = fem.mesh.concatenate([half_left_mirrored, half_right])
    return full_mesh.sweep(decimals=5)

def main():
    # 1. Generate Mesh and Setup Solid Mechanics model
    mesh = create_offcenter_hole_mesh()
    region = fem.RegionQuad(mesh)
    displacement = fem.Field(region, dim=2)
    
    field = fem.FieldContainer([displacement])
    
    umat = fem.LinearElasticPlaneStress(E=E, nu=NU)
    solid = fem.SolidBody(umat, field)
    
    # Data storage arrays (ONLY Stresses and Metadata)
    n_nodes = len(mesh.points)
    all_stresses = np.zeros((NUM_SAMPLES, n_nodes, 3))
    metadata = []
    
    edges = ['left', 'right', 'top', 'bottom']
    print(f"Mesh generated with {n_nodes} nodes and {len(mesh.cells)} quad elements.")
    print(f"Assembling and solving {NUM_SAMPLES} load cases for stress fields...")
    
    for i in range(NUM_SAMPLES):
        # Reset the displacement field to zero for each iteration
        displacement.values[:] = 0.0
        
        # Randomize edges and load
        clamped_edge = np.random.choice(edges)
        remaining_edges = [e for e in edges if e != clamped_edge]
        loaded_edge = np.random.choice(remaining_edges)
        
        # Pressure: Negative is tension, Positive is compression
        load_pressure = np.random.uniform(-100.0, 100.0)
        
        # Define Boundary Masks for the asymmetric plate
        eps = 1e-3
        masks = {
            'left': mesh.points[:, 0] < -L_left + eps,
            'right': mesh.points[:, 0] > L_right - eps,
            'bottom': mesh.points[:, 1] < -h + eps,
            'top': mesh.points[:, 1] > h - eps
        }
        
        # 3. Apply Boundary Conditions (Clamped)
        clamped_mask = masks[clamped_edge]
        boundaries = {"clamp": fem.Boundary(displacement, mask=clamped_mask)}
        
        # 4. Apply Pressure Load
        loaded_mask = masks[loaded_edge]
        region_boundary = fem.RegionQuadBoundary(mesh, mask=loaded_mask)
        field_boundary = fem.FieldContainer([fem.Field(region_boundary, dim=2)])
        
        load = fem.SolidBodyPressure(field_boundary, pressure=load_pressure)
        
        # 5. Solve the linear step
        step = fem.Step(items=[solid, load], boundaries=boundaries)
        fem.Job(steps=[step]).evaluate()
        
        # 6. Post-Processing (Project quadrature point stress to nodes)
        stress_nodes = fem.tools.project(solid.evaluate.stress(), region)
        
        # Store components [sigma_xx, sigma_yy, sigma_xy]
        all_stresses[i] = np.column_stack([stress_nodes[:, 0, 0], stress_nodes[:, 1, 1], stress_nodes[:, 0, 1]])
        
        meta = {
            'sample_id': i, 'clamped_edge': clamped_edge,
            'loaded_edge': loaded_edge, 'applied_pressure': load_pressure
        }
        metadata.append(meta)
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{NUM_SAMPLES} load cases...")

    # Save the master grid dataset (Strictly structural/stress data)
    print(f"Saving dataset to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        nodes=mesh.points, 
        elements=mesh.cells, 
        stresses=all_stresses, 
        metadata=metadata
    )
    print("Generation complete!")

    # --- Visual Verification ---
    print("Plotting verification for last sample...")
    
    # Convert quads to triangles just for matplotlib visualization
    triangles = []
    for cell in mesh.cells:
        triangles.append([cell[0], cell[1], cell[2]])
        triangles.append([cell[0], cell[2], cell[3]])
    triangles = np.array(triangles)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    sig_xx_last = all_stresses[-1, :, 0]
    
    tc = ax.tricontourf(mesh.points[:, 0], mesh.points[:, 1], triangles, sig_xx_last, levels=50, cmap='jet')
    plt.colorbar(tc, ax=ax, label=r'$\sigma_{xx}$ (MPa)')
    
    meta_last = metadata[-1]
    ax.set_title(f"Ground Truth Stress $\sigma_{{xx}}$ (Off-Center Hole)\nClamped: {meta_last['clamped_edge'].upper()} | Loaded: {meta_last['loaded_edge'].upper()} (P={meta_last['applied_pressure']:.1f})")
    ax.set_aspect('equal')
    plt.savefig("verification_plot.png", dpi=150, bbox_inches='tight')
    print("Saved 'verification_plot.png'. Ready for the PINN dataset splitter!")

if __name__ == "__main__":
    main()