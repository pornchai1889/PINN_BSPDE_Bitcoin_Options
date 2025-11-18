import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import os
import logging
from datetime import datetime

def plot_candlestick(df, result_dir, btc_price):
    """
    Plots a candlestick chart for the option price and overlays the BTC price.
    """
    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.suptitle('Option Candlestick Chart vs. BTC Price', fontsize=16)

    # Candlestick plotting logic
    # สร้างกราฟแท่งเทียน
    for i, row in df.iterrows():
        t = row['current_time_t']
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = 'green' if c >= o else 'red'
        
        # วาดเส้น high-low
        ax1.plot([t, t], [l, h], color=color, linewidth=1)
        # วาดแท่ง open-close
        rect_height = abs(c - o)
        rect_bottom = min(o, c)
        width = (df['current_time_t'].iloc[1] - df['current_time_t'].iloc[0]) * 0.8
        ax1.add_patch(Rectangle((t - width/2, rect_bottom), width, rect_height, facecolor=color, edgecolor=color))

    # พล็อตเส้นราคาปิดของออปชัน
    ax1.plot(df['current_time_t'], df['close'], label='Option Close Price', color='darkorange', linestyle='-', alpha=0.7, linewidth=1.5)

    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Option Price (V)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # สร้างแกน y ที่สองสำหรับราคา BTC
    ax2 = ax1.twinx()
    ax2.plot(df['current_time_t'], btc_price, label='BTCUSDT Price', color='lightgreen', alpha=0.9, linestyle=':')
    ax2.set_ylabel('BTCUSDT Price (S)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(result_dir, "candlestick_chart.png"), dpi=300)
    plt.close()


def eur_call_analytical_price(S, t, K, r, sigma, T):
    """
    Analytical solution for European call option using PyTorch.
    
    S: Current price of the underlying asset
    t: Current time
    K: Strike price
    r: Risk-free rate
    sigma: Volatility
    T: Time to maturity (Total contract duration)
    """
    # จัดการกรณีที่ T-t เป็นศูนย์หรือน้อยมากเพื่อหลีกเลี่ยงการหารด้วยศูนย์
    time_to_maturity = T - t
    # เพิ่มค่า epsilon เล็กน้อยเพื่อความเสถียร
    time_to_maturity = torch.clamp(time_to_maturity, min=1e-9)

    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * time_to_maturity) / (sigma * torch.sqrt(time_to_maturity))
    d2 = d1 - sigma * torch.sqrt(time_to_maturity)
    
    N = torch.distributions.Normal(0, 1).cdf
    
    V = S * N(d1) - K * torch.exp(-r * time_to_maturity) * N(d2)
    return V

class ImpliedParameterNet(nn.Module):
    """
    Neural Network to predict implied r and sigma.
    """
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_input, n_hidden))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, n_output))
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
        output = self.layers[-1](x)
        
        # ใช้ Softplus เพื่อให้แน่ใจว่าค่า r และ sigma เป็นบวกเสมอ
        output = torch.nn.functional.softplus(output)
        
        # แยก output ออกเป็น r และ sigma
        r, sigma = torch.split(output, 1, dim=1)
        return r, sigma

def main():
    """
    Main function to run the Implied Parameter Network training and evaluation.
    """
    # --- 1. ตั้งค่า Directory สำหรับบันทึกผล ---
    base_output_dir = os.path.join("btcusdt_options_call_V2", "real_btcusdt_implied_params_mlp")
    run_number = 1
    while True:
        result_dir = os.path.join(base_output_dir, f"run_{run_number}")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            break
        run_number += 1

    # --- 1.5. ตั้งค่า Logging ---
    log_filename = os.path.join(result_dir, f"run_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Results will be saved in: {result_dir}")

    # --- 2. ค่าคอนฟิกและ Hyperparameters ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 50000
    learning_rate = 1e-4
    
    p = {
        'n_layers': 8,
        'n_hidden': 128,
        'learning_rate': learning_rate,
        'epochs': n_epochs,
        'data_path': os.path.join("btcusdt_options_call_V2", "klines", "BTC-251024-110000-C_Weekly_1h.csv")
    }
    
    logging.info("--- Hyperparameters ---")
    for key, value in p.items():
        logging.info(f"{key}: {value}")
    logging.info("-----------------------")
    
    # --- 3. โหลดและเตรียมข้อมูล ---
    try:
        df = pd.read_csv(p['data_path'])
        df = df.dropna()
        logging.info(f"Data loaded successfully from {p['data_path']}. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Data file not found at {p['data_path']}")
        return

    # ดึงข้อมูลมาสร้างเป็น Tensor
    S_data = torch.tensor(df['btc_close_price'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    t_data = torch.tensor(df['current_time_t'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    K_data = torch.tensor(df['strike_price_K'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    T_data = torch.tensor(df['contract_duration_T'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    V_market_data = torch.tensor(df['close'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)

    # ทำ Normalization ข้อมูล
    S_mean, S_std = S_data.mean(), S_data.std()
    t_mean, t_std = t_data.mean(), t_data.std()
    V_market_mean, V_market_std = V_market_data.mean(), V_market_data.std()

    S_data_norm = (S_data - S_mean) / S_std
    t_data_norm = (t_data - t_mean) / t_std
    V_market_data_norm = (V_market_data - V_market_mean) / V_market_std
    
    X_train = torch.cat([S_data_norm, t_data_norm], dim=1)

    # --- 4. สร้างโมเดล, Optimizer, และ Loss Function ---
    model = ImpliedParameterNet(n_input=2, n_output=2, n_hidden=p['n_hidden'], n_layers=p['n_layers']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
    loss_fn = nn.MSELoss()

    logging.info(f"Model Architecture:\n{model}")

    # --- 5. Training Loop ---
    total_loss_history = []
    logging.info("Starting training...")
    
    for epoch in range(p['epochs']):
        model.train()
        
        r_pred, sigma_pred = model(X_train)
        V_predicted = eur_call_analytical_price(S_data, t_data, K_data, r_pred, sigma_pred, T_data)
        V_predicted_norm = (V_predicted - V_market_mean) / V_market_std
        loss = loss_fn(V_predicted_norm, V_market_data_norm)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss_history.append(loss.item())
        
        if (epoch + 1) % 1000 == 0:
            logging.info(f'Epoch [{epoch+1}/{p["epochs"]}], Loss: {loss.item():.8f}')

    logging.info("Training finished.")

    # --- 6. ทดสอบโมเดลและบันทึกผล ---
    model.eval()
    with torch.no_grad():
        final_r_pred, final_sigma_pred = model(X_train)

    result_df = df[[
        'open_time', 'open', 'high', 'low', 'close', 'btc_close_price', 
        'strike_price_K', 'time_to_maturity_t2m', 'current_time_t', 'contract_duration_T'
    ]].copy()
    result_df['predicted_r'] = final_r_pred.cpu().numpy()
    result_df['predicted_sigma'] = final_sigma_pred.cpu().numpy()

    csv_save_path = os.path.join(result_dir, "btc_option_parameters.csv")
    result_df.to_csv(csv_save_path, index=False)
    logging.info(f"Predicted parameters saved to {csv_save_path}")

    # --- 7. พล็อตกราฟแสดงผล ---
    logging.info("Generating plots...")
    
    # พล็อตกราฟ Total Loss
    plt.figure("Total Loss Curve", figsize=(10, 6))
    plt.plot(total_loss_history)
    plt.title('Total Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "total_loss_curve.png"), dpi=300)
    plt.close()

    # Predicted r vs ราคา BTC
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(result_df['current_time_t'], result_df['predicted_r'], label='Predicted Interest Rate (r)', color='darkorange')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Interest Rate (r)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(result_df['current_time_t'], result_df['btc_close_price'], label='BTCUSDT Price', color='lightgreen', alpha=0.9, linestyle=':')
    ax2.set_ylabel('BTCUSDT Price (S)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    plt.title('Implied Risk-free Rate (r) vs. BTC Price')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "implied_r_vs_btc_price.png"), dpi=300)
    plt.close()

    # Predicted sigma vs ราคา BTC
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(result_df['current_time_t'], result_df['predicted_sigma'], label='Predicted Volatility (sigma)', color='seagreen')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Volatility (sigma)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(result_df['current_time_t'], result_df['btc_close_price'], label='BTCUSDT Price', color='lightgreen', alpha=0.9, linestyle=':')
    ax2.set_ylabel('BTCUSDT Price (S)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    plt.title('Implied Volatility (sigma) vs. BTC Price')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "implied_sigma_vs_btc_price.png"), dpi=300)
    plt.close()
    
    # Predicted r vs ราคา Option
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(result_df['current_time_t'], result_df['predicted_r'], label='Predicted Interest Rate (r)', color='darkorange')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Interest Rate (r)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(result_df['current_time_t'], result_df['close'], label='Option Price (V)', color='blueviolet', alpha=0.9, linestyle=':')
    ax2.set_ylabel('Option Price (V)', color='blueviolet')
    ax2.tick_params(axis='y', labelcolor='blueviolet')
    plt.title('Implied Risk-free Rate (r) vs. Option Price')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "implied_r_vs_option_price.png"), dpi=300)
    plt.close()

    # Predicted sigma vs ราคา Option
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(result_df['current_time_t'], result_df['predicted_sigma'], label='Predicted Volatility (sigma)', color='seagreen')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Volatility (sigma)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(result_df['current_time_t'], result_df['close'], label='Option Price (V)', color='blueviolet', alpha=0.9, linestyle=':')
    ax2.set_ylabel('Option Price (V)', color='blueviolet')
    ax2.tick_params(axis='y', labelcolor='blueviolet')
    plt.title('Implied Volatility (sigma) vs. Option Price')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "implied_sigma_vs_option_price.png"), dpi=300)
    plt.close()

    # พล็อตกราฟแท่งเทียน
    plot_candlestick(df, result_dir, result_df['btc_close_price'])
    
    logging.info("All plots generated successfully.")
    logging.info(f"Process completed. All artifacts are in {result_dir}")


if __name__ == '__main__':
    main()