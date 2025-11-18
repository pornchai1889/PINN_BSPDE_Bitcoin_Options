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

def plot_candlestick_with_prediction(df, result_dir, btc_price, pinn_prediction):
    """
    Plots a candlestick chart for the option price, overlays the BTC price,
    and also overlays the PINN predicted option price.
    """
    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.suptitle('Option Candlestick Chart with PINN Prediction vs. BTC Price', fontsize=16)

    # Candlestick plotting logic
    # สร้างกราฟแท่งเทียน
    for i, row in df.iterrows():
        t = row['current_time_t']
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = 'green' if c >= o else 'red'
        
        ax1.plot([t, t], [l, h], color=color, linewidth=1)
        width = (df['current_time_t'].iloc[1] - df['current_time_t'].iloc[0]) * 0.8
        ax1.add_patch(Rectangle((t - width/2, min(o, c)), width, abs(c - o), facecolor=color, edgecolor=color))

    # พล็อตเส้นราคาปิดจริงของออปชัน
    ax1.plot(df['current_time_t'], df['close'], label='Actual Option Close', color='darkorange', linestyle='-', alpha=0.7, linewidth=1.5)
    # พล็อตเส้นราคาที่ทำนายจาก PINN
    ax1.plot(df['current_time_t'], pinn_prediction, label='PINN Predicted Price', color='magenta', linestyle='--', alpha=0.9, linewidth=1.5)

    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Option Price (V)', color='darkorange')
    ax1.tick_params(axis='y', labelcolor='darkorange')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # สร้างแกน y ที่สองสำหรับราคา BTC
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
    """
    Main function to run the PINN training and evaluation process.
    This version uses dynamic r and sigma from a pre-calculated file.
    """
    # --- 1. & 1.5. การตั้งค่า Directory และ Logging (เหมือนเดิม) ---
    base_output_dir = os.path.join("btcusdt_options_call_V2", "real_btcusdt_call_pricing_mlp")
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
        'learning_rate': 1e-4,
        'epochs': 100000,
        'warmup_epochs': 10000,
        'pde_weight': 1e-10,
        'data_path': os.path.join("btcusdt_options_call_V2", "real_btcusdt_implied_params_mlp", "run_1", "btc_option_parameters.csv")
    }
    
    logging.info("--- Hyperparameters ---")
    for key, value in p.items(): logging.info(f"{key}: {value}")
    logging.info("-----------------------")
    
    # --- 3. & 3.5. โหลดและเตรียมข้อมูล (เหมือนเดิม) ---
    try:
        df = pd.read_csv(p['data_path']).dropna()
        logging.info(f"Data loaded from {p['data_path']}. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Data file not found at {p['data_path']}"); return
    S_data = torch.tensor(df['btc_close_price'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    t_data = torch.tensor(df['current_time_t'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    V_market_data = torch.tensor(df['close'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    r_data = torch.tensor(df['predicted_r'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    sigma_data = torch.tensor(df['predicted_sigma'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    S_mean, S_std = S_data.mean(), S_data.std()
    t_mean, t_std = t_data.mean(), t_data.std()
    V_mean, V_std = V_market_data.mean(), V_market_data.std()
    S_data_norm = (S_data - S_mean) / S_std
    t_data_norm = (t_data - t_mean) / t_std
    V_market_data_norm = (V_market_data - V_mean) / V_std
    X_train = torch.cat([S_data_norm, t_data_norm], dim=1)
    
    # --- 4. สร้างโมเดล (เหมือนเดิม) ---
    model = PINN(n_input=2, n_output=1, n_hidden=p['n_hidden'], n_layers=p['n_layers']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
    loss_fn = nn.MSELoss()
    logging.info(f"Model Architecture:\n{model}")

    # --- 5. Training Loop ---
    loss_data_history, loss_pde_history, total_loss_history = [], [], []
    logging.info("Starting training...")

    for epoch in range(p['epochs']):
        model.train()
        optimizer.zero_grad()
        
        V_pinn_norm = model(X_train)
        loss_data = loss_fn(V_pinn_norm, V_market_data_norm)
        
        loss_pde_item = 0.0
        
        if epoch >= p['warmup_epochs']:
            S_train = S_data.clone().requires_grad_(True)
            t_train = t_data.clone().requires_grad_(True)
            S_train_norm = (S_train - S_mean) / S_std
            t_train_norm = (t_train - t_mean) / t_std
            X_pde = torch.cat([S_train_norm, t_train_norm], dim=1)
            V_pde_norm = model(X_pde)
            V_pde = V_pde_norm * V_std + V_mean
            
            dV_dt = torch.autograd.grad(outputs=V_pde, inputs=t_train, grad_outputs=torch.ones_like(V_pde), retain_graph=True, create_graph=True)[0]
            dV_dS = torch.autograd.grad(outputs=V_pde, inputs=S_train, grad_outputs=torch.ones_like(V_pde), retain_graph=True, create_graph=True)[0]
            d2V_dS2 = torch.autograd.grad(outputs=dV_dS, inputs=S_train, grad_outputs=torch.ones_like(dV_dS), retain_graph=True, create_graph=True)[0]
            
            pde_residual = dV_dt + 0.5 * (sigma_data**2) * (S_train**2) * d2V_dS2 + r_data * S_train * dV_dS - r_data * V_pde
            loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
            
            total_loss = loss_data + p['pde_weight'] * loss_pde
            # --- MODIFIED LINE ---
            # บันทึกค่า PDE Loss ที่ถูกถ่วงน้ำหนักแล้วสำหรับแสดงผล
            loss_pde_item = (p['pde_weight'] * loss_pde).item()
        else:
            total_loss = loss_data

        total_loss.backward()
        optimizer.step()
        
        loss_data_history.append(loss_data.item())
        loss_pde_history.append(loss_pde_item)
        total_loss_history.append(total_loss.item())
        
        if (epoch + 1) % 1000 == 0:
            logging.info(f'Epoch [{epoch+1}/{p["epochs"]}], Data Loss: {loss_data.item():.6f}, Weighted PDE Loss: {loss_pde_item:.6f}, Total Loss: {total_loss.item():.6f}')

    logging.info("Training finished.")

    # --- 6. & 7. บันทึกโมเดลและพล็อตกราฟ (เหมือนเดิม) ---
    model_save_path = os.path.join(result_dir, "pinn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")
    model.eval()
    with torch.no_grad():
        V_pinn_final_norm = model(X_train)
        V_pinn_final = V_pinn_final_norm * V_std + V_mean
    V_pinn_final_np = V_pinn_final.cpu().numpy()
    V_market_np = V_market_data.cpu().numpy()
    logging.info("Generating plots...")
    
    plt.figure("Loss Curves", figsize=(10, 6))
    plt.plot(total_loss_history, label='Total Loss')
    plt.plot(loss_data_history, label='Data Loss', linestyle='--', alpha=0.7)
    plt.plot(loss_pde_history, label=f'Weighted PDE Loss', linestyle='--', alpha=0.7)
    plt.title('Loss Curves')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log')
    plt.axvline(x=p['warmup_epochs'], color='r', linestyle='--', label='End of Warm-up')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(result_dir, "total_loss_curve.png"), dpi=300)
    plt.close()

    plt.figure("Scatter Comparison", figsize=(8, 8))
    plt.scatter(V_pinn_final_np, V_market_np, alpha=0.5, label='PINN vs Actual')
    min_val = min(V_market_np.min(), V_pinn_final_np.min())
    max_val = max(V_market_np.max(), V_pinn_final_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Match')
    plt.title('PINN Prediction vs Actual Market Price')
    plt.xlabel('PINN Predicted Price'); plt.ylabel('Actual Market Price (close)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(result_dir, "scatter_comparison.png"), dpi=300)
    plt.close()
    
    plot_candlestick_with_prediction(df, result_dir, df['btc_close_price'].values, V_pinn_final_np)
    
    logging.info("All plots generated successfully.")
    logging.info(f"Process completed. All artifacts are in {result_dir}")

if __name__ == '__main__':
    main()