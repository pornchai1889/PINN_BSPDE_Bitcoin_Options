import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

def eur_call_analytical_price(S, t, K, r, sigma, T):
    """
    Analytical solution for European call option using PyTorch.
    """
    time_to_maturity = T - t
    time_to_maturity = torch.clamp(time_to_maturity, min=1e-9)
    sigma = torch.clamp(sigma, min=1e-7) # Ensure sigma is positive

    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * time_to_maturity) / (sigma * torch.sqrt(time_to_maturity))
    d2 = d1 - sigma * torch.sqrt(time_to_maturity)
    
    N = torch.distributions.Normal(0, 1).cdf
    
    V = S * N(d1) - K * torch.exp(-r * time_to_maturity) * N(d2)
    return V

def main():
    """
    Main function to find the single best-fit constant r and sigma for a dataset.
    """
    # --- 1. การตั้งค่า Directory และ Logging ---
    base_output_dir = os.path.join("btcusdt_options_call_V2", "real_find_best_fit_constant_params")
    run_number = 1
    while True:
        result_dir = os.path.join(base_output_dir, f"run_{run_number}")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            break
        run_number += 1

    log_filename = os.path.join(result_dir, f"run_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info(f"Results will be saved in: {result_dir}")

    # --- 2. ค่าคอนฟิกและ Hyperparameters ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = {
        'learning_rate': 1e-3, # อาจใช้ learning rate สูงขึ้นได้
        'epochs': 20000,
        'data_path': os.path.join("btcusdt_options_call_V2", "klines", "BTC-250919-116000-C_Weekly_1h.csv")
    }
    
    logging.info("--- Hyperparameters ---")
    for key, value in p.items():
        logging.info(f"{key}: {value}")
    logging.info("-----------------------")
    
    # --- 3. โหลดและเตรียมข้อมูล ---
    try:
        df = pd.read_csv(p['data_path']).dropna()
        logging.info(f"Data loaded successfully from {p['data_path']}. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Data file not found at {p['data_path']}")
        return

    S_data = torch.tensor(df['btc_close_price'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    t_data = torch.tensor(df['current_time_t'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    K_data = torch.tensor(df['strike_price_K'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    T_data = torch.tensor(df['contract_duration_T'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)
    V_market_data = torch.tensor(df['close'].values, dtype=torch.float).unsqueeze(1).to(DEVICE)

    # Normalization สำหรับ V เพื่อความเสถียรของ Loss
    V_market_mean, V_market_std = V_market_data.mean(), V_market_data.std()
    V_market_data_norm = (V_market_data - V_market_mean) / V_market_std
    
    # --- 4. สร้าง r และ sigma ให้เป็นพารามิเตอร์ที่เรียนรู้ได้ ---
    # ให้ค่าเริ่มต้นที่สมเหตุสมผล
    r_param = nn.Parameter(torch.tensor(0.05, device=DEVICE)) 
    sigma_param = nn.Parameter(torch.tensor(0.8, device=DEVICE))

    # สร้าง Optimizer ที่จะปรับค่าแค่ r และ sigma
    optimizer = torch.optim.Adam([r_param, sigma_param], lr=p['learning_rate'])
    loss_fn = nn.MSELoss()

    logging.info(f"Initial r: {r_param.item():.4f}, Initial sigma: {sigma_param.item():.4f}")

    # --- 5. Training Loop ---
    loss_history = []
    logging.info("Starting optimization...")
    
    for epoch in range(p['epochs']):
        optimizer.zero_grad()
        
        # คำนวณ V ทำนายโดยใช้ r และ sigma ค่าปัจจุบัน
        # ใช้ torch.abs(sigma_param) เพื่อบังคับให้ sigma เป็นบวกเสมอ
        V_predicted = eur_call_analytical_price(S_data, t_data, K_data, r_param, torch.abs(sigma_param), T_data)
        V_predicted_norm = (V_predicted - V_market_mean) / V_market_std
        
        loss = loss_fn(V_predicted_norm, V_market_data_norm)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 1000 == 0:
            logging.info(f'Epoch [{epoch+1}/{p["epochs"]}], Loss: {loss.item():.8f}, r: {r_param.item():.4f}, sigma: {torch.abs(sigma_param).item():.4f}')

    logging.info("Optimization finished.")
    final_r = r_param.item()
    final_sigma = torch.abs(sigma_param).item()
    logging.info(f"--- Best-fit Constant Parameters ---")
    logging.info(f"Final Best-fit r: {final_r:.6f}")
    logging.info(f"Final Best-fit sigma: {final_sigma:.6f}")
    logging.info("------------------------------------")

    # --- 6. พล็อตกราฟแสดงผล ---
    logging.info("Generating plots...")
    
    # คำนวณ V ทำนายสุดท้ายด้วยค่าที่ดีที่สุด
    with torch.no_grad():
        V_final_predicted = eur_call_analytical_price(S_data, t_data, K_data, r_param, torch.abs(sigma_param), T_data)

    V_final_predicted_np = V_final_predicted.cpu().numpy().flatten()
    V_market_np = V_market_data.cpu().numpy().flatten()

    # กราฟ Loss Curve
    plt.figure("Loss Curve", figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Loss Curve for Best-fit Parameters')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # กราฟ Scatter Comparison
    rmse = np.sqrt(np.mean((V_market_np - V_final_predicted_np)**2))
    correlation = np.corrcoef(V_market_np, V_final_predicted_np)[0, 1]

    plt.figure("Scatter Comparison", figsize=(8, 8))
    plt.scatter(V_final_predicted_np, V_market_np, alpha=0.5)
    min_val = min(V_market_np.min(), V_final_predicted_np.min())
    max_val = max(V_market_np.max(), V_final_predicted_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Match')
    title_str = (
        f"Constant Parameter Prediction vs Actual Market Price\n"
        f"(RMSE: {rmse:.2f}, Correlation: {correlation:.4f})"
    )
    plt.title(title_str)
    plt.xlabel('Predicted Price (with constant r, sigma)')
    plt.ylabel('Actual Market Price (close)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(result_dir, "scatter_comparison.png"), dpi=300)
    plt.close()

    logging.info(f"Process completed. All artifacts are in {result_dir}")

if __name__ == '__main__':
    main()