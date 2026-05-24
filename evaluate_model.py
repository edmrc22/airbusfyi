import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from train_pi_deeponet import PIDeepONet, WingRibDataset

def evaluate_model_full_tensor(sample_idx=450):
    print(f"--- Launching DeepONet Full Tensor Validation (Sample {sample_idx}) ---")
    
    dataset = WingRibDataset("pinn_fem_ground_truth.npz")
    device = torch.device("cpu")
    
    model = PIDeepONet(num_sensors=dataset.num_sensors).to(device)
    
    weight_path = "pi_deeponet_weights.pth"
    if not os.path.exists(weight_path):
        print(f"FATAL ERROR: '{weight_path}' not found. Train the model first.")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print("Model weights loaded securely.")

    # Modified unpack: dataset no longer returns redundant grid
    branch_in, _ = dataset[sample_idx]
    
    branch_batch = branch_in.unsqueeze(0).to(device)
    nodes_batch = dataset.nodes_norm.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_norm = model(branch_batch, nodes_batch)
    
    pred_norm = pred_norm.squeeze(0).cpu().numpy()
    
    # Retrieve scaling properties
    s_min = dataset.stress_min.numpy()
    s_sig = dataset.S_sig.numpy()
    
    # Un-scale predictions cleanly
    pred_stresses = (pred_norm + 1.0) * s_sig + s_min
    
    raw_data = np.load("pinn_fem_ground_truth.npz", allow_pickle=True)
    true_stresses = raw_data['stresses'][sample_idx]
    nodes = raw_data['nodes']
    elements = raw_data['elements']
    
    triangles = np.array([[c[0], c[1], c[2]] for c in elements] + [[c[0], c[2], c[3]] for c in elements])
    stress_names = [r'$\sigma_{xx}$ (Normal X)', r'$\sigma_{yy}$ (Normal Y)', r'$\tau_{xy}$ (Shear)']
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f"DeepONet Full Tensor Autopsy - Unseen Sample {sample_idx}", fontsize=20, y=0.98)
    
    print("\n--- ERROR METRICS ---")
    for i in range(3):
        true_comp = true_stresses[:, i]
        pred_comp = pred_stresses[:, i]
        error_comp = np.abs(true_comp - pred_comp)
        
        print(f"{stress_names[i]}:")
        print(f"  Max Absolute Error:  {np.max(error_comp):.2f} MPa")
        print(f"  Mean Absolute Error: {np.mean(error_comp):.2f} MPa")
        
        tc0 = axes[i, 0].tricontourf(nodes[:, 0], nodes[:, 1], triangles, true_comp, levels=50, cmap='jet')
        axes[i, 0].set_title(f"FEM Ground Truth: {stress_names[i]}")
        plt.colorbar(tc0, ax=axes[i, 0])
        
        tc1 = axes[i, 1].tricontourf(nodes[:, 0], nodes[:, 1], triangles, pred_comp, levels=tc0.levels, cmap='jet')
        axes[i, 1].set_title(f"DeepONet Prediction: {stress_names[i]}")
        plt.colorbar(tc1, ax=axes[i, 1])
        
        tc2 = axes[i, 2].tricontourf(nodes[:, 0], nodes[:, 1], triangles, error_comp, levels=50, cmap='Reds')
        axes[i, 2].set_title(f"Absolute Error Map: {stress_names[i]}")
        plt.colorbar(tc2, ax=axes[i, 2])
        
        for j in range(3):
            axes[i, j].set_aspect('equal')
            axes[i, j].set_xlabel("X")
            axes[i, j].set_ylabel("Y")
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"validation_autopsy_full_sample_{sample_idx}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved massive 3x3 validation autopsy to {filename}")

if __name__ == "__main__":
    evaluate_model_full_tensor(sample_idx=450)