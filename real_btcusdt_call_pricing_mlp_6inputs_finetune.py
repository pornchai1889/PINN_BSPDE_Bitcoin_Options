import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import os
import logging
from datetime import datetime

# --- MODIFIED FUNCTION DEFINITION ---
def plot_candlestick_with_prediction(df, result_dir, btc_price, pinn_prediction, rmse, correlation):
    """
    Plots a candlestick chart for the option price, overlays the BTC price,
    and also overlays the PINN predicted option price with metrics in the title.
    """
    fig, ax1 = plt.subplots(figsize=(15, 8))
    # --- MODIFIED TITLE ---
    title_str = (
        f"Market vs. PINN Price Comparison\n"
        f"(RMSE: {rmse:.2f}, R: {correlation:.4f})"
    )
    fig.suptitle(title_str, fontsize=16)

    # Candlestick plotting logic
    for i, row in df.iterrows():
        t = row['current_time_t']
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = 'green' if c >= o else 'red'
        
        ax1.plot([t, t], [l, h], color=color, linewidth=1)
        width = (df['current_time_t'].iloc[1] - df['current_time_t'].iloc[0]) * 0.8
        ax1.add_patch(Rectangle((t - width/2, min(o, c)), width, abs(c - o), facecolor=color, edgecolor=color))
        
    ax1.plot(df['current_time_t'], df['close'], label='Market Price (Close)', color='purple', linestyle='-', alpha=0.7, linewidth=1.5)
    ax1.plot(df['current_time_t'], pinn_prediction, label='PINN Price', color='darkorange', linestyle='--', alpha=0.9, linewidth=1.5)

    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Option Price (V)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(df['current_time_t'], btc_price, label='BTCUSDT Price', color='lightgreen', alpha=0.9, linestyle=':')
    ax2.set_ylabel('BTCUSDT Price (S)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(result_dir, "candlestick_with_prediction.png"), dpi=300)
    plt.close()

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Option Pricing.
    """
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_input, n_hidden)] +
                                    [nn.Linear(n_hidden, n_hidden) for _ in range(n_layers - 1)] +
                                    [nn.Linear(n_hidden, n_output)])
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

def main():
    # --- 1. & 1.5. การตั้งค่า Directory และ Logging ---
    base_output_dir = os.path.join("btcusdt_options_call_V2", "real_btcusdt_call_pricing_mlp_6inputs")
    run_number = 1
    while True:
        result_dir = os.path.join(base_output_dir, f"result_{run_number}")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            break
        run_number += 1
    log_filename = os.path.join(result_dir, f"run_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info(f"Results will be saved in: {result_dir}")

    # --- 2. ค่าคอนฟิกและ Hyperparameters ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = {
        'n_layers': 8,
        'n_hidden': 128,
        'learning_rate': 1e-5, # <<< (ปรับค่าให้ละเอียดขึ้นจากต้นแบบ)
        'epochs': 40000,
        'warmup_epochs': 5000,
        'annealing_epochs': 10000,
        'pde_weight_target': 1e-6,
        'data_path': os.path.join("btcusdt_options_call_V2", "real_btcusdt_implied_params_mlp", "run_4", "btc_option_parameters_169.csv")
    }
    
    logging.info("--- Hyperparameters ---")
    for key, value in p.items(): logging.info(f"{key}: {value}")
    logging.info("-----------------------")
    
    # --- 3. โหลดและเตรียมข้อมูล ---
    try:
        df = pd.read_csv(p['data_path']).dropna()
        logging.info(f"Data loaded from {p['data_path']}. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Data file not found at {p['data_path']}"); return

    # (ส่วนของการเตรียมข้อมูลคงเดิมทั้งหมด)
    S_data = torch.tensor(df['btc_close_price'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    t_data = torch.tensor(df['current_time_t'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    r_data = torch.tensor(df['predicted_r'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    sigma_data = torch.tensor(df['predicted_sigma'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    K_data = torch.tensor(df['strike_price_K'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    T_data = torch.tensor(df['contract_duration_T'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    V_market_data = torch.tensor(df['close'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)

    S_mean, S_std = S_data.mean(), S_data.std()
    t_mean, t_std = t_data.mean(), t_data.std()
    r_mean, r_std = r_data.mean(), r_data.std()
    sigma_mean, sigma_std = sigma_data.mean(), sigma_data.std()
    K_mean, K_std = K_data.mean(), K_data.std()
    T_mean, T_std = T_data.mean(), T_data.std()
    V_mean, V_std = V_market_data.mean(), V_market_data.std()

    S_std = 1 if S_std == 0 else S_std
    t_std = 1 if t_std == 0 else t_std
    r_std = 1 if r_std == 0 else r_std
    sigma_std = 1 if sigma_std == 0 else sigma_std
    K_std = 1 if K_std == 0 else K_std
    T_std = 1 if T_std == 0 else T_std

    S_norm = (S_data - S_mean) / S_std
    t_norm = (t_data - t_mean) / t_std
    r_norm = (r_data - r_mean) / r_std
    sigma_norm = (sigma_data - sigma_mean) / sigma_std
    K_norm = (K_data - K_mean) / K_std
    T_norm = (T_data - T_mean) / T_std
    V_market_norm = (V_market_data - V_mean) / V_std
    
    X_train = torch.cat([S_norm, t_norm, r_norm, sigma_norm, K_norm, T_norm], dim=1)
    
    # --- 4. สร้างโมเดล, Optimizer, และ Loss Function (ปรับปรุงเพื่อ Fine-tune) ---
    
    # ปรับ n_input เป็น 6
    model = PINN(n_input=6, n_output=1, n_hidden=p['n_hidden'], n_layers=p['n_layers']).to(DEVICE)
    
    # <<< START: โค้ดที่เพิ่มเข้ามาสำหรับ Fine-tuning >>>
    pretrained_model_path = os.path.join("btcusdt_options_call_V2", "real_btcusdt_call_pricing_mlp_6inputs", "result_18", "pinn_model168.pth")
    
    if os.path.exists(pretrained_model_path):
        try:
            model.load_state_dict(torch.load(pretrained_model_path, map_location=DEVICE))
            logging.info(f"Successfully loaded pretrained model for fine-tuning from: {pretrained_model_path}")
        except Exception as e:
            logging.error(f"Error loading model state_dict: {e}. Training from scratch.")
    else:
        logging.warning(f"Pretrained model not found at '{pretrained_model_path}'. Training from scratch.")
    # <<< END: โค้ดที่เพิ่มเข้ามาสำหรับ Fine-tuning >>>

    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
    loss_fn = nn.MSELoss()
    logging.info(f"Model Architecture:\n{model}")

    # --- 5. Training Loop (เหมือนเดิม) ---
    loss_data_history, loss_pde_history, total_loss_history = [], [], []
    logging.info("Starting training (fine-tuning)...") # <<< ปรับ Log เล็กน้อย
    for epoch in range(p['epochs']):
        model.train()
        optimizer.zero_grad()
        V_pinn_norm = model(X_train)
        loss_data = loss_fn(V_pinn_norm, V_market_norm)
        current_pde_weight = 0.0
        loss_pde_item = 0.0
        if epoch > p['warmup_epochs']:
            S_pde = S_data.clone().requires_grad_(True)
            t_pde = t_data.clone().requires_grad_(True)
            r_pde, sigma_pde, K_pde, T_pde = r_data.clone(), sigma_data.clone(), K_data.clone(), T_data.clone()
            S_pde_norm = (S_pde - S_mean) / S_std
            t_pde_norm = (t_pde - t_mean) / t_std
            r_pde_norm = (r_pde - r_mean) / r_std
            sigma_pde_norm = (sigma_pde - sigma_mean) / sigma_std
            K_pde_norm = (K_pde - K_mean) / K_std
            T_pde_norm = (T_pde - T_mean) / T_std
            X_pde = torch.cat([S_pde_norm, t_pde_norm, r_pde_norm, sigma_pde_norm, K_pde_norm, T_pde_norm], dim=1)
            V_pde_norm = model(X_pde)
            V_pde = V_pde_norm * V_std + V_mean
            dV_dt = torch.autograd.grad(outputs=V_pde, inputs=t_pde, grad_outputs=torch.ones_like(V_pde), retain_graph=True, create_graph=True)[0]
            dV_dS = torch.autograd.grad(outputs=V_pde, inputs=S_pde, grad_outputs=torch.ones_like(V_pde), retain_graph=True, create_graph=True)[0]
            d2V_dS2 = torch.autograd.grad(outputs=dV_dS, inputs=S_pde, grad_outputs=torch.ones_like(dV_dS), retain_graph=True, create_graph=True)[0]
            pde_residual = dV_dt + 0.5 * (sigma_pde**2) * (S_pde**2) * d2V_dS2 + r_pde * S_pde * dV_dS - r_pde * V_pde
            loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
            loss_pde_item = loss_pde.item()
            annealing_progress = min(1.0, (epoch - p['warmup_epochs']) / p['annealing_epochs'])
            current_pde_weight = p['pde_weight_target'] * annealing_progress
        total_loss = loss_data + current_pde_weight * loss_pde_item
        if current_pde_weight > 0:
            total_loss = loss_data + current_pde_weight * loss_pde
        total_loss.backward()
        optimizer.step()
        loss_data_history.append(loss_data.item())
        loss_pde_history.append(loss_pde_item)
        total_loss_history.append(total_loss.item())
        if (epoch + 1) % 1000 == 0:
            logging.info(f'Epoch [{epoch+1}/{p["epochs"]}], Data Loss: {loss_data.item():.6f}, PDE Loss: {loss_pde_item:.6f}, Current Weight: {current_pde_weight:.2e}, Total Loss: {total_loss.item():.6f}')
    logging.info("Training finished.")

    # --- 6. บันทึกโมเดล ---
    model_save_path = os.path.join(result_dir, "pinn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    # --- 7. ทำนายผลและพล็อตกราฟ (เหมือนเดิม) ---
    model.eval()
    with torch.no_grad():
        V_pinn_final_norm = model(X_train)
        V_pinn_final = V_pinn_final_norm * V_std + V_mean
    
    V_pinn_final_np = V_pinn_final.cpu().numpy()
    V_market_np = V_market_data.cpu().numpy()

    logging.info("Generating plots...")
    
    # --- START: NEW METRICS CALCULATION ---
    actual_prices = V_market_np.flatten()
    predicted_prices = V_pinn_final_np.flatten()
    
    rmse = np.sqrt(np.mean((actual_prices - predicted_prices)**2))
    correlation = np.corrcoef(actual_prices, predicted_prices)[0, 1]
    logging.info(f"Final Metrics -> RMSE: {rmse:.4f}, Correlation: {correlation:.4f}")
    # --- END: NEW METRICS CALCULATION ---

    # กราฟ Loss Curves
    plt.figure("Loss Curves", figsize=(10, 6))
    plt.plot(total_loss_history, label='Total Loss')
    plt.plot(loss_data_history, label='Data Loss', linestyle='--', alpha=0.7)
    plt.plot(np.array(loss_pde_history) * p['pde_weight_target'], label=f'Raw PDE Loss (scaled for viz)', linestyle=':', alpha=0.5)
    plt.title('Loss Curves')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log')
    plt.axvline(x=p['warmup_epochs'], color='r', linestyle='--', label='End of Warm-up')
    plt.axvline(x=p['warmup_epochs'] + p['annealing_epochs'], color='g', linestyle='--', label='End of Annealing')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(result_dir, "total_loss_curve.png"), dpi=300)
    plt.close()

    # กราฟ Scatter Comparison
    plt.figure("Scatter Comparison", figsize=(8, 8))
    plt.scatter(predicted_prices, actual_prices, alpha=0.5, label='PINN vs Actual')
    min_val = min(actual_prices.min(), predicted_prices.min())
    max_val = max(actual_prices.max(), predicted_prices.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Match')
    title_str = (
        f"PINN Prediction vs Actual Market Price\n"
        f"(RMSE: {rmse:.2f}, Correlation: {correlation:.4f})"
    )
    plt.title(title_str)
    plt.xlabel('PINN Predicted Price'); plt.ylabel('Actual Market Price (close)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(result_dir, "scatter_comparison.png"), dpi=300)
    plt.close()
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    title_str = (
        f"Market vs. PINN Price Comparison\n"
        f"(RMSE: {rmse:.2f}, R: {correlation:.4f})"
    )
    fig.suptitle(title_str, fontsize=16)

    ax1.plot(df['current_time_t'], actual_prices, label='Market Price (Close)', color='purple', linestyle='-')
    ax1.plot(df['current_time_t'], predicted_prices, label='PINN Price', color='darkorange', linestyle='--')

    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Option Price (V)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(df['current_time_t'], df['btc_close_price'].values, label='BTCUSDT Price', color='lightgreen', linestyle=':')
    ax2.set_ylabel('BTCUSDT Price (S)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(result_dir, "market_vs_pinn_comparison.png"), dpi=300)
    plt.close()
    
    logging.info("Generated 'Market vs. PINN Price Comparison' plot.")

    plot_candlestick_with_prediction(df, result_dir, df['btc_close_price'].values, predicted_prices, rmse, correlation)
    
    logging.info("All plots generated successfully.")
    logging.info(f"Process completed. All artifacts are in {result_dir}")

if __name__ == '__main__':
    main()