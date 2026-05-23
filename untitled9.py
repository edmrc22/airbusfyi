import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

# =========================================================================
# 1. ROBUST DATASET WITH COORDINATE NORMALIZATION
# =========================================================================
class DynamicPINNDataset(Dataset):
    def __init__(self, processed_filepath="processed_pinn_training_data.npz"):
        data = np.load(processed_filepath)
        
        raw_sensor_coords = data['sensor_coords'] # Dimensions: (1000, 26, 2)
        raw_collo_coords = data['collo_coords']   # Dimensions: (164, 2) or (1000, 164, 2)
        
        # Safe handling of whether collocation coordinates are batched or static spatial points
        if len(raw_collo_coords.shape) == 3:
            base_collo = raw_collo_coords[0]
        else:
            base_collo = raw_collo_coords

        self.c_min = np.min(base_collo, axis=0)
        self.c_max = np.max(base_collo, axis=0)
        self.c_scale = (self.c_max - self.c_min) / 2.0 
        
        def scale_c(coords):
            return 2.0 * (coords - self.c_min) / (self.c_max - self.c_min + 1e-8) - 1.0

        norm_sensor_coords = scale_c(raw_sensor_coords)
        self.collo_coords = torch.tensor(scale_c(base_collo), dtype=torch.float32)
        
        raw_stresses = data['sensor_stresses']
        self.s_min, self.s_max = raw_stresses.min(), raw_stresses.max()
        norm_stresses = (raw_stresses - self.s_min) / (self.s_max - self.s_min + 1e-8)
        
        combined_inputs = np.concatenate([norm_sensor_coords, norm_stresses], axis=-1)
        self.num_samples = len(raw_stresses)
        self.inputs = torch.tensor(combined_inputs, dtype=torch.float32).view(self.num_samples, -1)
        
        self.target_min, self.target_max = data['collo_stresses'].min(), data['collo_stresses'].max()
        self.s_scale = float(self.target_max - self.target_min)
        
        norm_targets = (data['collo_stresses'] - self.target_min) / (self.target_max - self.target_min + 1e-8)
        self.targets = torch.tensor(norm_targets, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'sensor_state': self.inputs[idx],
            'true_stresses': self.targets[idx],
            'coords': self.collo_coords
        }

# =========================================================================
# 2. HIGH-PERFORMANCE DEEPONET ARCHITECTURE (UPDATED SENSORS COUNT)
# =========================================================================
class DeepONetPlatePINN(nn.Module):
    def __init__(self, num_sensors=26, hidden_dim=256, basis_functions=128):
        super().__init__()
        self.num_sensors = num_sensors
        self.basis_functions = basis_functions
        branch_input_dim = num_sensors * 5  # 2 coordinates + 3 stress values per sensor = 5
        
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
        
        # Handle whether coordinates are passed as static global items or per-sample batches
        if len(coords.shape) == 2:
            num_collo = coords.shape[0]
            # Add implicit batch dimension to match forward trunk expectations
            coords_input = coords.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            num_collo = coords.shape[1]
            coords_input = coords

        b_out = self.branch(sensor_state).view(batch_size, 3, self.basis_functions)
        t_out = self.trunk(coords_input.reshape(-1, 2)).view(batch_size, num_collo, 3, self.basis_functions)
        
        # High speed dot product optimization eliminating memory expansion overhead
        out = torch.einsum('bsf,bcsf->bcs', b_out, t_out)
        return out

# =========================================================================
# 3. OPTIMIZED PHYSICS LOSS
# =========================================================================
class PhysicsInformedLoss(nn.Module):
    def __init__(self, stress_scale, coord_scale, lambda_physics=1e-1):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()
        
        self.stress_scale = stress_scale
        self.coord_scale = coord_scale 
        
    def forward(self, pred, true, coords, compute_physics=True):
        loss_data = self.mse(pred, true)
        
        if not compute_physics or self.lambda_physics == 0:
            return loss_data, loss_data, torch.tensor(0.0, device=pred.device)

        sxx, syy, sxy = pred[..., 0], pred[..., 1], pred[..., 2]
        
        # Graph execution optimization: Passing pre-allocated unified weight matrices 
        # avoids computational graph collapsing errors across batch channels.
        grad_ones = torch.ones_like(sxx)
        dsxx_dxy = torch.autograd.grad(sxx, coords, grad_outputs=grad_ones, create_graph=True)[0]
        dsyy_dxy = torch.autograd.grad(syy, coords, grad_outputs=grad_ones, create_graph=True)[0]
        dsxy_dxy = torch.autograd.grad(sxy, coords, grad_outputs=grad_ones, create_graph=True)[0]
        
        dp_dx = self.stress_scale / self.coord_scale[0]
        dp_dy = self.stress_scale / self.coord_scale[1]
        
        res_x_phys = dsxx_dxy[..., 0] * dp_dx + dsxy_dxy[..., 1] * dp_dy
        res_y_phys = dsxy_dxy[..., 0] * dp_dx + dsyy_dxy[..., 1] * dp_dy
        
        char_scale = self.stress_scale / torch.mean(self.coord_scale)
        loss_phys = torch.mean((res_x_phys / char_scale)**2 + (res_y_phys / char_scale)**2)
        
        return loss_data + (self.lambda_physics * loss_phys), loss_data, loss_phys

# =========================================================================
# 4. RUNTIME SYSTEM EXECUTION ENGINE
# =========================================================================
if __name__ == "__main__":
    # Context-aware device routing optimization
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        use_pin_memory = True
        print("System Pipeline Status: Active Acceleration Found (CUDA).")
    else:
        device = torch.device("cpu")
        use_pin_memory = False
        print("System Pipeline Status: Standard CPU Engine Active.")
        
    full_dataset = DynamicPINNDataset()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, pin_memory=use_pin_memory)
    
    # Explicitly configure neural layout to match generated metadata array bounds
    model = DeepONetPlatePINN(num_sensors=26, hidden_dim=256, basis_functions=128).to(device)
    
    c_scale_tensor = torch.tensor(full_dataset.c_scale, dtype=torch.float32).to(device)
    criterion = PhysicsInformedLoss(stress_scale=full_dataset.s_scale, 
                                    coord_scale=c_scale_tensor).to(device)
                                    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    for epoch in range(1, 401):
        model.train()
        t_loss = 0
        for b in train_loader:
            optimizer.zero_grad(set_to_none=True)
            s_state = b['sensor_state'].to(device, non_blocking=True)
            t_stress = b['true_stresses'].to(device, non_blocking=True)
            
            # Enforce tracking requirements specifically onto the coordinate subset tensor
            coords = b['coords'].to(device, non_blocking=True).requires_grad_(True)
            
            pred = model(coords, s_state)
            loss, l_data, l_phys = criterion(pred, t_stress, coords, compute_physics=True)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()

        # VALIDATION PERFORMANCE ACCELERATION LOOP
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for b in val_loader:
                coords_val = b['coords'].to(device, non_blocking=True)
                s_state_val = b['sensor_state'].to(device, non_blocking=True)
                t_stress_val = b['true_stresses'].to(device, non_blocking=True)
                
                pred_val = model(coords_val, s_state_val)
                # Setting compute_physics=False skips the costly backpropagation graph compilation
                loss_val, _, _ = criterion(pred_val, t_stress_val, coords_val, compute_physics=False)
                v_loss += loss_val.item()
                
        v_loss /= len(val_loader)
        scheduler.step(v_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Val Loss: {v_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e}")

    torch.save(model.state_dict(), "deeponet_pinn_weights.pth")
    print("Clean model parameters saved successfully.")