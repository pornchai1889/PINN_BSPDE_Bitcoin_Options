import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import logging
from datetime import datetime

def main():
    """
    Main function to run the PINN training and evaluation process for BTCUSDT options.
    This version includes model saving functionality.
    """
    # --- 1. Dynamic and Structured Directory Setup ---
    base_output_dir = os.path.join("btcusdt_options_call_V2", "btcusdt_call_pricing_mlp")
    run_number = 1
    while True:
        result_dir = os.path.join(base_output_dir, f"result_{run_number}")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            break
        run_number += 1

    # --- 1.5. Setup Logging ---
    log_filename = os.path.join(result_dir, f"run_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"--- Created new result directory: {result_dir} ---")
    logging.info(f"--- Log file created at: {log_filename} ---\n")

    # --- 2. Setup and Configuration ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {DEVICE}")

    # --- 3. PINN Model Parameters for BTCUSDT ---
    K = 116000.0
    r = 0.05
    sigma = 0.321775
    T = 7 / 365.0
    #T = 0.01916496
    S_range = [90000.0, 140000.0]
    t_range = [0.0, T]
    
    S_min, S_max = S_range
    t_min, t_max = t_range

    payoff_func = lambda x: np.fmax(x - K, 0)

    # Normalization and Denormalization functions
    def normalize(t, S):
        t_norm = (t - t_min) / (t_max - t_min)
        S_norm = (S - S_min) / (S_max - S_min)
        return t_norm, S_norm

    def denormalize_S(S_norm):
        return S_norm * (S_max - S_min) + S_min

    # Data Generation Functions
    def get_diff_data(n):
        t_points = np.random.uniform(*t_range, (n, 1))
        S_points = np.random.uniform(*S_range, (n, 1))
        t_norm, S_norm = normalize(t_points, S_points)
        return np.concatenate([t_norm, S_norm], axis=1)

    def get_ivp_data(n):
        t_points = T * np.ones((n, 1))
        S_points = np.random.uniform(*S_range, (n, 1))
        t_norm, S_norm = normalize(t_points, S_points)
        X_norm = np.concatenate([t_norm, S_norm], axis=1)
        y_val = payoff_func(S_points)
        return X_norm, y_val / K

    def get_bvp_data(n):
        t_points = np.random.uniform(*t_range, (n, 1))
        S1_points = S_min * np.ones((n, 1))
        t1_norm, S1_norm = normalize(t_points, S1_points)
        X1_norm = np.concatenate([t1_norm, S1_norm], axis=1)
        y1_val = np.zeros((n, 1))
        
        S2_points = S_max * np.ones((n, 1))
        t2_norm, S2_norm = normalize(t_points, S2_points)
        X2_norm = np.concatenate([t2_norm, S2_norm], axis=1)
        y2_val = (S2_points - K * np.exp(-r * (T - t_points))).reshape(-1, 1)
        return X1_norm, y1_val / K, X2_norm, y2_val / K

    # --- 4. PINN Model Definition ---
    class EuropeanCallPINN(nn.Module):
        def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
            super().__init__()
            activation = nn.Tanh()
            self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation)
            self.fch = nn.Sequential(*[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation) for _ in range(N_LAYERS)])
            self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        def forward(self, x):
            return self.fce(self.fch(self.fcs(x)))

    # --- 5. Training Setup ---
    N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS = 2, 1, 100, 8
    EPOCHS = 80000
    LEARNING_RATE = 1e-4
    N_SAMPLE = 8000 # สำหรับ get_diff_data(4 * N_SAMPLE) จะเป็น 4 เท่า
    PHYSICS_LOSS_WEIGHT = 0.1

    logging.info("\n" + "="*50)
    logging.info("TRAINING PARAMETERS")
    logging.info("="*50)
    logging.info("Market Parameters:")
    logging.info(f"  - Strike Price (K): {K}")
    logging.info(f"  - Risk-free Rate (r): {r}")
    logging.info(f"  - Volatility (sigma): {sigma}")
    logging.info(f"  - Maturity (T): {T:.5f} years")
    logging.info(f"  - Price Range (S_range): {S_range}")
    logging.info("\nPINN & Training Hyperparameters:")
    logging.info(f"  - Input Neurons: {N_INPUT}")
    logging.info(f"  - Output Neurons: {N_OUTPUT}")
    logging.info(f"  - Hidden Neurons per Layer: {N_HIDDEN}")
    logging.info(f"  - Number of Hidden Layers: {N_LAYERS}")
    logging.info(f"  - Epochs: {EPOCHS}")
    logging.info(f"  - Learning Rate: {LEARNING_RATE}")
    logging.info(f"  - Samples per Batch (N_SAMPLE): {N_SAMPLE}")
    logging.info(f"  - Physics Loss Weight: {PHYSICS_LOSS_WEIGHT}")
    logging.info("="*50 + "\n")

    model = EuropeanCallPINN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/10**3:.2f}K")
    loss_history = {"total_loss": [], "loss_ivp_bvp": [], "loss_BS": [], "loss_bvp1": [], "loss_bvp2": [], "loss_ivp": []}

    # --- 6. Training Loop ---
    logging.info("\n--- Starting PINN Training ---")
    for i in range(EPOCHS):
        optimizer.zero_grad()
        
        ivp_x, ivp_y = get_ivp_data(N_SAMPLE)
        ivp_y_pred = model(torch.from_numpy(ivp_x).float().to(DEVICE))
        mse_ivp = loss_fn(torch.from_numpy(ivp_y).float().to(DEVICE), ivp_y_pred)

        bvp_x1, bvp_y1, bvp_x2, bvp_y2 = get_bvp_data(N_SAMPLE)
        mse_bvp1 = loss_fn(model(torch.from_numpy(bvp_x1).float().to(DEVICE)), torch.from_numpy(bvp_y1).float().to(DEVICE))
        mse_bvp2 = loss_fn(model(torch.from_numpy(bvp_x2).float().to(DEVICE)), torch.from_numpy(bvp_y2).float().to(DEVICE))
        data_loss = mse_ivp + mse_bvp1 + mse_bvp2

        X_pde_norm = get_diff_data(4 * N_SAMPLE)
        X_pde_tensor = torch.from_numpy(X_pde_norm).float().to(DEVICE).requires_grad_()
        v_norm = model(X_pde_tensor)

        grads = torch.autograd.grad(v_norm, X_pde_tensor, grad_outputs=torch.ones_like(v_norm), retain_graph=True, create_graph=True)[0]
        dv_dt_norm, dv_dS_norm = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)
        
        grads2nd = torch.autograd.grad(dv_dS_norm, X_pde_tensor, grad_outputs=torch.ones_like(dv_dS_norm), create_graph=True)[0]
        d2v_dS2_norm = grads2nd[:, 1].view(-1, 1)

        dV_dt = K * dv_dt_norm * (1 / (t_max - t_min)) # แปลงเป็นค่าจริง
        dV_dS = K * dv_dS_norm * (1 / (S_max - S_min))
        d2V_dS2 = K * d2v_dS2_norm * (1 / (S_max - S_min))**2
        
        S_pde = denormalize_S(X_pde_tensor[:, 1].view(-1, 1))
        V_pde = v_norm * K

        pde = dV_dt + 0.5 * (sigma**2 * S_pde**2) * d2V_dS2 + r * S_pde * dV_dS - r * V_pde
        pde_loss = PHYSICS_LOSS_WEIGHT * loss_fn(pde / K, torch.zeros_like(pde)) # pde / K ย่อขนาด

        loss = data_loss + pde_loss
        loss_history["total_loss"].append(loss.item())
        loss_history["loss_ivp_bvp"].append(data_loss.item())
        loss_history["loss_BS"].append(pde_loss.item())
        loss_history["loss_ivp"].append(mse_ivp.item())
        loss_history["loss_bvp1"].append(mse_bvp1.item())
        loss_history["loss_bvp2"].append(mse_bvp2.item())
        loss.backward(); optimizer.step()

        if (i + 1) % 1000 == 0:
            logging.info(f"Epoch {i+1}/{EPOCHS}, Loss: {loss.item():.8f}, Data: {data_loss.item():.8f}, PDE: {pde_loss.item():.8f}")
    
    logging.info("--- Training Finished ---\n")

    # <<< START: ADDED MODEL SAVING >>>
    # --- 6.5. Save the Trained Model ---
    model_save_path = os.path.join(result_dir, "pinn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"--- Model saved successfully to: {model_save_path} ---\n")
    # <<< END: ADDED MODEL SAVING >>>

    # --- 7. Analytical Solution Function ---
    def eur_call_analytical_price(S, t, K, r, sigma, T):
        t2m = T - t
        epsilon = 1e-8
        t2m = torch.clamp(t2m, min=epsilon); S = torch.clamp(S, min=epsilon)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * t2m) / (sigma * torch.sqrt(t2m))
        d2 = d1 - sigma * torch.sqrt(t2m)
        N0 = lambda value: 0.5 * (1 + torch.erf(value / (2**0.5)))
        return S * N0(d1) - K * N0(d2) * torch.exp(-r * t2m)

    # --- 8. Comprehensive Visualization ---
    logging.info(f"--- Generating and Saving Visualizations to {result_dir} ---")
    model.eval()

    # Plot 1: Data Sampling
    plt.figure("Data Sampling", figsize=(10, 7))
    t_bvp = np.random.uniform(t_range[0], t_range[1], 500)
    S_bvp_min = np.full_like(t_bvp, S_range[0])
    S_bvp_max = np.full_like(t_bvp, S_range[1])
    S_ivp = np.random.uniform(S_range[0], S_range[1], 500)
    t_ivp = np.full_like(S_ivp, t_range[1])
    t_pde = np.random.uniform(t_range[0], t_range[1], 2000)
    S_pde = np.random.uniform(S_range[0], S_range[1], 2000)
    plt.scatter(t_bvp, S_bvp_min, label=f"BVP (S={S_range[0]})", color="red", marker="x")
    plt.scatter(t_bvp, S_bvp_max, label=f"BVP (S={S_range[1]})", color="green", marker="x")
    plt.scatter(t_ivp, S_ivp, label="IVP (t=T)", color="blue", marker="o")
    plt.scatter(t_pde, S_pde, label="PDE Collocation Points", color="grey", alpha=0.3)
    plt.xlabel("Time (t)"); plt.ylabel("BTCUSDT Price (S)")
    plt.title("Data Sampling for BTCUSDT Call Option PINN")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(result_dir, "data_sampling.png")); plt.close("Data Sampling")

    # Plot 2: Detailed Training Curves
    loss_df = pd.DataFrame(loss_history)
    loss_df_ma = loss_df.rolling(window=100).mean()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    fig_loss, axes = plt.subplots(len(loss_df.columns), 1, figsize=(12, 15), sharex=True)
    fig_loss.suptitle('Detailed Training Curves (Moving Average)', fontsize=16)
    for i, col in enumerate(loss_df.columns):
        axes[i].plot(loss_df_ma.index, loss_df_ma[col], label=col, color=colors[i % len(colors)])
        axes[i].set_ylabel(col); axes[i].grid(True, linestyle='--', alpha=0.6); axes[i].set_yscale('log'); axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Epoch'); plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(result_dir, "detailed_training_curves.png")); plt.close(fig_loss)

    # Plot 2.5: Separate Total Loss Curve
    plt.figure("Total Loss", figsize=(10, 6))
    plt.plot(loss_df_ma.index, loss_df_ma['total_loss'], label='Total Loss (MA)', color='tab:red')
    plt.title('Total Loss Curve (Moving Average)')
    plt.xlabel('Epoch'); plt.ylabel('Total Loss')
    plt.yscale('log'); plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "total_loss_curve.png")); plt.close("Total Loss")

    # Prepare test data for remaining plots
    s_grid_vals = np.linspace(S_range[0], S_range[1], 100)
    t_grid_vals = np.linspace(t_range[0], t_range[1], 100)
    s_grid, t_grid = np.meshgrid(s_grid_vals, t_grid_vals)
    t_grid_norm, s_grid_norm = normalize(t_grid.flatten(), s_grid.flatten())
    X_test_norm = torch.tensor(np.column_stack((t_grid_norm, s_grid_norm)), dtype=torch.float).to(DEVICE)
    with torch.no_grad():
        y_pinn_test_norm = model(X_test_norm)
    y_pinn_test_np = y_pinn_test_norm.cpu().numpy() * K
    t_grid_tensor = torch.tensor(t_grid.flatten(), dtype=torch.float).to(DEVICE)
    s_grid_tensor = torch.tensor(s_grid.flatten(), dtype=torch.float).to(DEVICE)
    y_analytical_test = eur_call_analytical_price(s_grid_tensor, t_grid_tensor, K, r, sigma, T)
    y_analytical_test_np = y_analytical_test.cpu().numpy()

    # Plot 3: 3D Surface Comparison
    fig_3d = plt.figure("3D Surface Comparison", figsize=(16, 7))
    ax1 = fig_3d.add_subplot(121, projection='3d'); ax1.plot_surface(s_grid, t_grid, y_analytical_test_np.reshape(s_grid.shape), cmap="viridis")
    ax1.set_title("Analytical Solution"); ax1.set_xlabel("S"); ax1.set_ylabel("t"); ax1.set_zlabel("V"); ax1.view_init(elev=30, azim=-120)
    ax2 = fig_3d.add_subplot(122, projection='3d'); ax2.plot_surface(s_grid, t_grid, y_pinn_test_np.reshape(s_grid.shape), cmap="viridis")
    ax2.set_title("PINN Prediction"); ax2.set_xlabel("S"); ax2.set_ylabel("t"); ax2.set_zlabel("V"); ax2.view_init(elev=30, azim=-120)
    plt.savefig(os.path.join(result_dir, "3d_surface_comparison.png")); plt.close(fig_3d)

    # Plot 4: Scatter Plot Comparison
    
    # Calculate Metrics for Scatter Plot
    y_true_scatter = y_analytical_test_np.flatten()
    y_pred_scatter = y_pinn_test_np.flatten()
    rmse_scatter = np.sqrt(np.mean((y_true_scatter - y_pred_scatter)**2))
    corr_scatter = np.corrcoef(y_true_scatter, y_pred_scatter)[0, 1]
    title_scatter = (f"PINN vs. Analytical Predictions\n"
                     f"(RMSE: {rmse_scatter:.2f}, Correlation: {corr_scatter:.4f})")
    
    plt.figure("Scatter Plot Comparison", figsize=(8, 8)); plt.scatter(y_pred_scatter, y_true_scatter, marker="x", alpha=0.6)
    plt.plot([y_analytical_test_np.min(), y_analytical_test_np.max()], [y_analytical_test_np.min(), y_analytical_test_np.max()], 'r--', label='Ideal Match (y=x)')
    plt.xlabel("PINN Prediction"); plt.ylabel("Analytical Prediction")
    
    # <<< MODIFIED LINE >>>
    plt.title(title_scatter)
    
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend(); plt.axis('equal')
    plt.savefig(os.path.join(result_dir, "scatter_comparison.png")); plt.close("Scatter Plot Comparison")

    # Plot 5: Simulated Market Path Comparison
    n_steps = 250
    t_path = np.linspace(t_range[0], t_range[1], n_steps)
    S_path = K * np.exp(np.cumsum(np.random.randn(n_steps) * sigma * np.sqrt(T / n_steps)))
    t_path_norm, S_path_norm = normalize(t_path, S_path)
    X_path_test_norm = torch.tensor(np.column_stack((t_path_norm, S_path_norm)), dtype=torch.float).to(DEVICE)
    with torch.no_grad():
        y_pinn_path_norm = model(X_path_test_norm)
    y_pinn_path_np = y_pinn_path_norm.cpu().numpy() * K
    y_analytical_path = eur_call_analytical_price(torch.tensor(S_path, dtype=torch.float).to(DEVICE), torch.tensor(t_path, dtype=torch.float).to(DEVICE), K, r, sigma, T)
    y_analytical_path_np = y_analytical_path.cpu().numpy()
    
    y_true_path = y_analytical_path_np.flatten()
    y_pred_path = y_pinn_path_np.flatten()
    rmse_path = np.sqrt(np.mean((y_true_path - y_pred_path)**2))
    corr_path = np.corrcoef(y_true_path, y_pred_path)[0, 1]
    title_path = (f"Option Price Comparison on a Simulated BTCUSDT Path\n"
                  f"(RMSE: {rmse_path:.2f}, Correlation: {corr_path:.4f})")

    plt.figure("Simulated Path Comparison", figsize=(12, 7)); main_ax = plt.gca()
    main_ax.plot(t_path, y_analytical_path_np, label='Analytical Price', color='dodgerblue', linewidth=2)
    main_ax.plot(t_path, y_pinn_path_np, label='PINN Price', color='darkorange', linestyle='--', linewidth=2)
    
    main_ax.set_title(title_path)
    
    main_ax.set_xlabel('Time (t)'); main_ax.set_ylabel('Option Premium Price (V)')
    ax2 = main_ax.twinx()
    ax2.plot(t_path, S_path, label='BTCUSDT Path', color='green', alpha=0.4, linestyle=':')
    ax2.set_ylabel('BTCUSDT Price (S)', color='green'); ax2.tick_params(axis='y', labelcolor='green')
    lines, labels = main_ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    main_ax.legend(lines + lines2, labels + labels2, loc='upper left'); main_ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(result_dir, "simulated_path_comparison.png")); plt.close("Simulated Path Comparison")
    
    logging.info(f"--- All visualizations saved successfully to {result_dir} ---")

if __name__ == "__main__":
    main()