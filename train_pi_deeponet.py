import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. OPTIMIZED DATA LOADER 
# ==========================================
class WingRibDataset(Dataset):
    def __init__(self, filepath="pinn_fem_ground_truth.npz"):
        print(f"Loading FEM dataset from {filepath}...")
        data = np.load(filepath, allow_pickle=True)
        
        self.nodes = torch.tensor(data['nodes'], dtype=torch.float32)
        self.stresses = torch.tensor(data['stresses'], dtype=torch.float32)
        self.sensor_indices = data['sensor_indices']
        
        self.num_samples = self.stresses.shape[0]
        self.num_nodes = self.nodes.shape[0]
        self.num_sensors = len(self.sensor_indices)
        
        self.x_min, self.x_max = self.nodes[:, 0].min(), self.nodes[:, 0].max()
        self.y_min, self.y_max = self.nodes[:, 1].min(), self.nodes[:, 1].max()
        
        self.stress_min = self.stresses.amin(dim=(0, 1))
        self.stress_max = self.stresses.amax(dim=(0, 1))
        
        self.S_x = (self.x_max - self.x_min) / 2.0
        self.S_y = (self.y_max - self.y_min) / 2.0
        self.S_sig = (self.stress_max - self.stress_min) / 2.0
        
        self.nodes_norm = self.nodes.clone()
        self.nodes_norm[:, 0] = (self.nodes[:, 0] - self.x_min) / self.S_x - 1.0
        self.nodes_norm[:, 1] = (self.nodes[:, 1] - self.y_min) / self.S_y - 1.0
        
        self.stresses_norm = (self.stresses - self.stress_min) / self.S_sig - 1.0
        self.branch_inputs = self.stresses_norm[:, self.sensor_indices, :].view(self.num_samples, -1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.branch_inputs[idx], self.stresses_norm[idx]

# ==========================================
# 2. STRICT HIGH-FREQUENCY ARCHITECTURE
# ==========================================
class FourierFeatureTransform(nn.Module):
    """Forces the network into the high-frequency domain to prevent flat-field collapse."""
    def __init__(self, in_features, mapping_size, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
        
    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh()) 
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class PIDeepONet(nn.Module):
    def __init__(self, num_sensors, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.branch = MLP(input_dim=num_sensors * 3, hidden_layers=[256, 256, 256], output_dim=3 * latent_dim)
        
        self.fourier = FourierFeatureTransform(in_features=2, mapping_size=128, scale=10.0) 
        self.trunk = MLP(input_dim=256, hidden_layers=[256, 128, 128], output_dim=latent_dim)
        
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, branch_in, trunk_in):
        b_out = self.branch(branch_in).view(branch_in.shape[0], 3, self.latent_dim)
        t_encoded = self.fourier(trunk_in)
        t_out = self.trunk(t_encoded)
        
        pred_stresses = torch.einsum('bvd, bpd -> bpv', b_out, t_out)
        return pred_stresses + self.bias

# ==========================================
# 3. STRICT SPLIT-VARIABLE PHYSICS ENGINE 
# ==========================================
def compute_physics_loss(model, branch_in, colloc_coords, S_x, S_y, S_sig):
    x = colloc_coords[..., 0:1].clone().requires_grad_(True)
    y = colloc_coords[..., 1:2].clone().requires_grad_(True)
    coords = torch.cat([x, y], dim=-1)
    
    stresses = model(branch_in, coords)
    sigma_xx_norm = stresses[..., 0:1]
    sigma_yy_norm = stresses[..., 1:2]
    tau_xy_norm   = stresses[..., 2:3]
    
    d_sxx_dx = torch.autograd.grad(sigma_xx_norm, x, grad_outputs=torch.ones_like(sigma_xx_norm), create_graph=True)[0]
    d_sxx_dy = torch.autograd.grad(sigma_xx_norm, y, grad_outputs=torch.ones_like(sigma_xx_norm), create_graph=True)[0]
    
    d_syy_dx = torch.autograd.grad(sigma_yy_norm, x, grad_outputs=torch.ones_like(sigma_yy_norm), create_graph=True)[0]
    d_syy_dy = torch.autograd.grad(sigma_yy_norm, y, grad_outputs=torch.ones_like(sigma_yy_norm), create_graph=True)[0]
    
    d_txy_dx = torch.autograd.grad(tau_xy_norm, x, grad_outputs=torch.ones_like(tau_xy_norm), create_graph=True)[0]
    d_txy_dy = torch.autograd.grad(tau_xy_norm, y, grad_outputs=torch.ones_like(tau_xy_norm), create_graph=True)[0]
    
    S_xx, S_yy, S_xy = S_sig[0], S_sig[1], S_sig[2]
    
    coeff_1 = (S_xy * S_x) / (S_xx * S_y)
    residual_eq_x = d_sxx_dx + coeff_1 * d_txy_dy
    
    coeff_2 = (S_xy * S_y) / (S_yy * S_x)
    residual_eq_y = coeff_2 * d_txy_dx + d_syy_dy
    
    d2_sxx_dx2 = torch.autograd.grad(d_sxx_dx, x, grad_outputs=torch.ones_like(d_sxx_dx), create_graph=True)[0]
    d2_syy_dx2 = torch.autograd.grad(d_syy_dx, x, grad_outputs=torch.ones_like(d_syy_dx), create_graph=True)[0]
    
    d2_sxx_dy2 = torch.autograd.grad(d_sxx_dy, y, grad_outputs=torch.ones_like(d_sxx_dy), create_graph=True)[0]
    d2_syy_dy2 = torch.autograd.grad(d_syy_dy, y, grad_outputs=torch.ones_like(d_syy_dy), create_graph=True)[0]
    
    ratio_syy_sxx = S_yy / S_xx
    ratio_y_x = (S_x**2) / (S_y**2)
    
    laplacian_x_term = d2_sxx_dx2 + ratio_syy_sxx * d2_syy_dx2
    laplacian_y_term = d2_sxx_dy2 + ratio_syy_sxx * d2_syy_dy2
    
    residual_comp = laplacian_x_term + ratio_y_x * laplacian_y_term
    
    return torch.mean(residual_eq_x**2) + torch.mean(residual_eq_y**2) + torch.mean(residual_comp**2)

# ==========================================
# 4. STREAMLINED OPTIMIZATION LOOP
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    dataset = WingRibDataset("pinn_fem_ground_truth.npz")
    global_nodes_norm = dataset.nodes_norm.to(device)
    
    S_x, S_y = dataset.S_x, dataset.S_y
    S_sig = dataset.S_sig.to(device)

    indices = list(range(dataset.num_samples))
    train_dataset = torch.utils.data.Subset(dataset, indices[:400])
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    model = PIDeepONet(num_sensors=dataset.num_sensors).to(device)
    
    # ---------------------------------------
    # PHASE 1: ADAM GLOBAL OPTIMIZATION
    # ---------------------------------------
    print("\n--- PHASE 1: Adam Global Search (With Capped Annealing) ---")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=300, gamma=0.5)

    ADAM_EPOCHS = 1000
    N_DATA_POINTS = 200      
    N_COLLOC_POINTS = 400   
    
    sensor_idx_tensor = torch.tensor(dataset.sensor_indices, dtype=torch.long, device=device)
    
    # Adaptive Weighting Initialization
    lambda_phys = 1.0  
    alpha_anneal = 0.9 
    trunk_last_weight = model.trunk.network[-1].weight  
    
    for epoch in range(ADAM_EPOCHS):
        total_data_loss = 0.0
        total_phys_loss = 0.0
        
        for branch_batch, targets_batch in dataloader:
            batch_size = branch_batch.shape[0]
            branch_batch = branch_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer_adam.zero_grad()
            
            # --- A. SENSOR-ANCHORED DATA LOSS ---
            trunk_in_sensors = global_nodes_norm[sensor_idx_tensor].unsqueeze(0).expand(batch_size, -1, -1)
            target_sensors = targets_batch[:, sensor_idx_tensor, :]
            pred_sensors = model(branch_batch, trunk_in_sensors)
            loss_data_sensors = nn.functional.mse_loss(pred_sensors, target_sensors)
            
            rand_indices = torch.randperm(dataset.num_nodes, device=device)[:(N_DATA_POINTS - dataset.num_sensors)]
            trunk_in_field = global_nodes_norm[rand_indices].unsqueeze(0).expand(batch_size, -1, -1)
            target_field = targets_batch[:, rand_indices, :]
            pred_field = model(branch_batch, trunk_in_field)
            loss_data_field = nn.functional.mse_loss(pred_field, target_field)
            
            loss_data = (20.0 * loss_data_sensors) + loss_data_field
            
            # --- B. PHYSICS LOSS ---
            colloc_idx = torch.randperm(dataset.num_nodes, device=device)[:N_COLLOC_POINTS]
            colloc_coords = global_nodes_norm[colloc_idx].unsqueeze(0).expand(batch_size, -1, -1)
            loss_phys = compute_physics_loss(model, branch_batch, colloc_coords, S_x, S_y, S_sig)
            
            # --- C. PROTECTED ANNEALING & WARMUP ---
            if epoch < 200:
                # WARMUP: 0.0 physics weight forces the network to map the hole first
                lambda_phys = 0.0 
            else:
                grad_data = torch.autograd.grad(loss_data, trunk_last_weight, retain_graph=True)[0]
                grad_phys = torch.autograd.grad(loss_phys, trunk_last_weight, retain_graph=True)[0]
                
                max_grad_data = torch.max(torch.abs(grad_data))
                mean_grad_phys = torch.mean(torch.abs(grad_phys))
                
                target_lambda = max_grad_data / (mean_grad_phys + 1e-8)
                
                # THE LEASH: Prevent Lambda from collapsing the network
                target_lambda = torch.clamp(target_lambda, min=0.1, max=10.0)
                
                lambda_phys = (alpha_anneal * lambda_phys) + ((1.0 - alpha_anneal) * target_lambda.item())
            
            # --- D. OPTIMIZATION STEP ---
            total_loss = loss_data + (lambda_phys * loss_phys)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_adam.step()
            
            total_data_loss += loss_data.item()
            total_phys_loss += loss_phys.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_data = total_data_loss / len(dataloader)
            avg_phys = total_phys_loss / len(dataloader)
            print(f"Adam Epoch {epoch+1:04d}/{ADAM_EPOCHS} | Anchored Data: {avg_data:.5f} | Phys: {avg_phys:.5f} | Lambda: {lambda_phys:.3f}")

    # ---------------------------------------
    # PHASE 2: L-BFGS PDE POLISHING
    # ---------------------------------------
    # Default to 1.0 if warmup somehow bypassed annealing
    final_lambda = lambda_phys if lambda_phys > 0.0 else 1.0 
    print(f"\n--- PHASE 2: L-BFGS PDE Polishing (Locked at Lambda: {final_lambda:.3f}) ---")
    
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), 
        lr=0.1,             
        max_iter=20,        
        history_size=50,    
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )
    
    LBFGS_EPOCHS = 30 
    nan_detected = False
    
    for epoch in range(LBFGS_EPOCHS):
        if nan_detected:
            break
            
        total_data_loss = 0.0
        
        for branch_batch, targets_batch in dataloader:
            batch_size = branch_batch.shape[0]
            branch_batch = branch_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            rand_indices = torch.randperm(dataset.num_nodes, device=device)[:(N_DATA_POINTS - dataset.num_sensors)]
            trunk_in_field = global_nodes_norm[rand_indices].unsqueeze(0).expand(batch_size, -1, -1)
            target_field = targets_batch[:, rand_indices, :]
            
            trunk_in_sensors = global_nodes_norm[sensor_idx_tensor].unsqueeze(0).expand(batch_size, -1, -1)
            target_sensors = targets_batch[:, sensor_idx_tensor, :]
            
            colloc_idx = torch.randperm(dataset.num_nodes, device=device)[:N_COLLOC_POINTS]
            colloc_coords = global_nodes_norm[colloc_idx].unsqueeze(0).expand(batch_size, -1, -1)
            
            def closure():
                optimizer_lbfgs.zero_grad()
                
                pred_sensors = model(branch_batch, trunk_in_sensors)
                loss_data_sensors = nn.functional.mse_loss(pred_sensors, target_sensors)
                
                pred_field = model(branch_batch, trunk_in_field)
                loss_data_field = nn.functional.mse_loss(pred_field, target_field)
                
                loss_data = (20.0 * loss_data_sensors) + loss_data_field
                loss_phys = compute_physics_loss(model, branch_batch, colloc_coords, S_x, S_y, S_sig)
                
                total_loss = loss_data + (final_lambda * loss_phys)
                total_loss.backward()
                
                return total_loss
                
            loss_tensor = optimizer_lbfgs.step(closure)
            current_loss = loss_tensor.item()
            
            if np.isnan(current_loss):
                print("\n[FATAL ERROR] L-BFGS encountered NaN. Halting training to preserve previous weights.")
                nan_detected = True
                break
                
            with torch.no_grad():
                total_data_loss += nn.functional.mse_loss(model(branch_batch, trunk_in_field), target_field).item()
                
        if not nan_detected and (epoch + 1) % 5 == 0:
            avg_data = total_data_loss / len(dataloader)
            print(f"L-BFGS Epoch {epoch+1:02d}/{LBFGS_EPOCHS} | PDE Polishing Complete for Epoch")

    if not nan_detected:
        print("\nTraining Complete. Saving cleanly polished model weights...")
        torch.save(model.state_dict(), "pi_deeponet_weights.pth")
    else:
        print("\nTraining aborted during Phase 2. The Adam weights from Phase 1 are still in memory.")
        torch.save(model.state_dict(), "pi_deeponet_weights_adam_only.pth")

if __name__ == "__main__":
    train_model()