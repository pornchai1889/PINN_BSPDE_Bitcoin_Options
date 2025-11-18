import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import logging
from datetime import datetime

# --- 1. คลาส PINN (ต้องเหมือนกับตอนที่ฝึก) ---
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

# --- 2. ฟังก์ชันคำนวณสถิติย้อนหลัง (Retroactive) ---
def get_normalization_stats(df_slice, device):
    """
    คำนวณ Mean และ Std จาก DataFrame ที่กำหนด (ย้อนหลัง)
    """
    S_data = torch.tensor(df_slice['btc_close_price'].values, dtype=torch.float).unsqueeze(1).to(device)
    t_data = torch.tensor(df_slice['current_time_t'].values, dtype=torch.float).unsqueeze(1).to(device)
    r_data = torch.tensor(df_slice['predicted_r'].values, dtype=torch.float).unsqueeze(1).to(device)
    sigma_data = torch.tensor(df_slice['predicted_sigma'].values, dtype=torch.float).unsqueeze(1).to(device)
    K_data = torch.tensor(df_slice['strike_price_K'].values, dtype=torch.float).unsqueeze(1).to(device)
    T_data = torch.tensor(df_slice['contract_duration_T'].values, dtype=torch.float).unsqueeze(1).to(device)
    V_market_data = torch.tensor(df_slice['close'].values, dtype=torch.float).unsqueeze(1).to(device)

    stats = {
        'S_mean': S_data.mean(), 'S_std': S_data.std(),
        't_mean': t_data.mean(), 't_std': t_data.std(),
        'r_mean': r_data.mean(), 'r_std': r_data.std(),
        'sigma_mean': sigma_data.mean(), 'sigma_std': sigma_data.std(),
        'K_mean': K_data.mean(), 'K_std': K_data.std(),
        'T_mean': T_data.mean(), 'T_std': T_data.std(),
        'V_mean': V_market_data.mean(), 'V_std': V_market_data.std()
    }

    # จัดการกรณีที่ Std เป็น 0 (ถ้าข้อมูลมีแค่ 1 แถว)
    for key in stats.keys():
        if 'std' in key and stats[key] == 0:
            stats[key] = torch.tensor(1.0, device=device)
            
    return stats

# --- 3. ฟังก์ชัน Normalize Input ---
def normalize_input(S, t, r, sigma, K, T, stats):
    """
    Normalize input 6 ตัว โดยใช้สถิติที่คำนวณมา
    """
    S_norm = (S - stats['S_mean']) / stats['S_std']
    t_norm = (t - stats['t_mean']) / stats['t_std']
    r_norm = (r - stats['r_mean']) / stats['r_std']
    sigma_norm = (sigma - stats['sigma_mean']) / stats['sigma_std']
    K_norm = (K - stats['K_mean']) / stats['K_std']
    T_norm = (T - stats['T_mean']) / stats['T_std']
    
    # รวมเป็น Tensor [1, 6]
    X_norm = torch.cat([S_norm, t_norm, r_norm, sigma_norm, K_norm, T_norm], dim=1)
    return X_norm

# --- 4. Main Evaluation Script ---
def evaluate_walk_forward():
    
    # --- 4.1. ตั้งค่าพื้นฐาน ---
    mpl.rcParams['axes.unicode_minus'] = False # สำหรับ Matplotlib
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # พารามิเตอร์โมเดล (ต้องตรงกับตอนฝึก)
    N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS = 6, 1, 128, 8

    # ตำแหน่งไฟล์และโฟลเดอร์
    DATA_FILE = os.path.join("btcusdt_options_call_V2", "real_btcusdt_implied_params_mlp", "run_4", "btc_option_parameters_169.csv")
    MODEL_DIR = os.path.join("btcusdt_options_call_V2", "model_B")
    OUTPUT_DIR = MODEL_DIR # เซฟผลลัพธ์ไว้ที่เดียวกับโมเดล
    
    # สร้างโฟลเดอร์ Output (ถ้ายังไม่มี)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ตั้งค่า Logging
    log_filename = os.path.join(OUTPUT_DIR, f"evaluation_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info(f"--- Starting Walk-Forward Evaluation ---")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Loading data from: {DATA_FILE}")
    logging.info(f"Loading models from: {MODEL_DIR}")
    logging.info(f"Saving results to: {OUTPUT_DIR}")

    # --- 4.2. โหลดข้อมูลทั้งหมด ---
    try:
        df_all = pd.read_csv(DATA_FILE).dropna()
        if len(df_all) < 169:
            logging.error(f"Data file only contains {len(df_all)} rows. Expected 169 or more.")
            return
        # เราจะใช้ข้อมูลแค่ 169 แถวแรก (index 0-168)
        df_all = df_all.iloc[0:169].copy()
        logging.info(f"Loaded {len(df_all)} data points (Index 0-168).")
    except FileNotFoundError:
        logging.error(f"Data file not found at {DATA_FILE}"); return
    except Exception as e:
        logging.error(f"Error loading data: {e}"); return

    # --- 4.3. Walk-Forward Prediction Loop (10 จุด) ---
    forecast_predictions = []
    forecast_actuals = []
    forecast_indices = []
    forecast_times = []
    
    num_predictions = 10
    start_point_num = 159 # เริ่มจากโมเดลที่ฝึก 159 จุด
    
    logging.info("\n--- Starting 10-Step Walk-Forward Prediction ---")

    for i in range(num_predictions):
        model_train_points = start_point_num + i # เช่น 159, 160, ..., 168
        current_index = model_train_points - 1   # Index ของจุดปัจจุบัน (Input 5 ตัว) e.g., 158
        forecast_index = model_train_points      # Index ของจุดอนาคต (Input t, Actual V) e.g., 159
        
        model_name = f"pinn_model{model_train_points}.pth"
        model_path = os.path.join(MODEL_DIR, model_name)
        
        logging.info(f"Step {i+1}/{num_predictions}: Loading '{model_name}' to predict point {forecast_index + 1} (Index {forecast_index})...")

        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}"); break
            
        try:
            # 1. โหลดโมเดล
            model = PINN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            # 2. คำนวณสถิติย้อนหลัง
            df_train = df_all.iloc[0 : model_train_points] # e.g., [0:159] -> index 0-158
            stats = get_normalization_stats(df_train, DEVICE)

            # 3. เตรียม Input
            # ดึง 5 inputs จากจุดปัจจุบัน (current_index)
            S_in = torch.tensor(df_all.iloc[current_index]['btc_close_price'], dtype=torch.float).view(1, 1).to(DEVICE)
            r_in = torch.tensor(df_all.iloc[current_index]['predicted_r'], dtype=torch.float).view(1, 1).to(DEVICE)
            sigma_in = torch.tensor(df_all.iloc[current_index]['predicted_sigma'], dtype=torch.float).view(1, 1).to(DEVICE)
            K_in = torch.tensor(df_all.iloc[current_index]['strike_price_K'], dtype=torch.float).view(1, 1).to(DEVICE)
            T_in = torch.tensor(df_all.iloc[current_index]['contract_duration_T'], dtype=torch.float).view(1, 1).to(DEVICE)
            
            # ดึง 1 input (t) จากจุดอนาคต (forecast_index)
            t_in = torch.tensor(df_all.iloc[forecast_index]['current_time_t'], dtype=torch.float).view(1, 1).to(DEVICE)

            # 4. Normalize Input
            X_norm = normalize_input(S_in, t_in, r_in, sigma_in, K_in, T_in, stats)

            # 5. ทำนาย
            with torch.no_grad():
                V_norm_pred = model(X_norm)
            
            # 6. Denormalize Output
            V_pred = (V_norm_pred.item() * stats['V_std'].item()) + stats['V_mean'].item()
            
            # 7. เก็บผลลัพธ์
            V_actual = df_all.iloc[forecast_index]['close']
            
            forecast_predictions.append(V_pred)
            forecast_actuals.append(V_actual)
            forecast_indices.append(forecast_index) # เก็บ index (159-168)
            forecast_times.append(df_all.iloc[forecast_index]['current_time_t'])
            
            logging.info(f"  -> Predicted: {V_pred:.4f}, Actual: {V_actual:.4f}")

        except Exception as e:
            logging.error(f"Error during prediction step {i+1}: {e}"); break
            
    logging.info("--- Walk-Forward Prediction Finished ---")

    # --- 4.4. คำนวณ Metrics (เฉพาะ 10 จุด) ---
    if len(forecast_actuals) == num_predictions:
        actual_10 = np.array(forecast_actuals)
        predicted_10 = np.array(forecast_predictions)
        
        rmse = np.sqrt(np.mean((actual_10 - predicted_10)**2))
        # np.corrcoef อาจจะคืนค่า NaN ถ้าค่าคงที่, เพิ่มการตรวจสอบ
        if np.std(actual_10) > 0 and np.std(predicted_10) > 0:
            correlation = np.corrcoef(actual_10, predicted_10)[0, 1]
        else:
            correlation = np.nan
        
        logging.info("\n--- Forecast Performance (10 Points) ---")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"Correlation (R): {correlation:.4f}")
        logging.info("--------------------------------------")
    else:
        logging.warning(f"Could not calculate metrics. Only {len(forecast_actuals)} predictions were made.")
        rmse = np.nan
        correlation = np.nan

    # --- 4.5. บันทึก CSV ผลลัพธ์ 10 จุด ---
    try:
        df_forecast = pd.DataFrame({
            'Index': forecast_indices, # ลำดับแท่งเทียน (e.g., 159-168)
            'current_time_t': forecast_times,
            'Actual_V': forecast_actuals,
            'Predicted_V': forecast_predictions
        })
        csv_path = os.path.join(OUTPUT_DIR, "walk_forward_forecast_results.csv")
        df_forecast.to_csv(csv_path, index=False)
        logging.info(f"Forecast results CSV saved to: {csv_path}")
    except Exception as e:
        logging.error(f"Failed to save forecast CSV: {e}")


    # --- 4.6. ทำนายส่วน History (1-159) ---
    logging.info("\n--- Predicting Historical Part (Points 1-159) ---")
    history_predictions = [np.nan] * 159 # สร้าง list ว่าง 159 ช่อง
    
    # ใช้โมเดลที่ฝึกจาก 169 จุด (ตามที่คุณระบุ)
    model_final_name = "pinn_model169.pth" 
    model_final_path = os.path.join(MODEL_DIR, model_final_name)
    
    if os.path.exists(model_final_path):
        try:
            # 1. โหลดโมเดล
            model_final = PINN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS).to(DEVICE)
            model_final.load_state_dict(torch.load(model_final_path, map_location=DEVICE))
            model_final.eval()
            
            # 2. คำนวณสถิติ (จากข้อมูลทั้งหมด 169 จุด)
            stats_all = get_normalization_stats(df_all, DEVICE)
            
            # 3. เตรียม Input (ทั้งหมด 169 จุด)
            X_all_norm = normalize_input(
                torch.tensor(df_all['btc_close_price'].values, dtype=torch.float).view(-1, 1).to(DEVICE),
                torch.tensor(df_all['current_time_t'].values, dtype=torch.float).view(-1, 1).to(DEVICE),
                torch.tensor(df_all['predicted_r'].values, dtype=torch.float).view(-1, 1).to(DEVICE),
                torch.tensor(df_all['predicted_sigma'].values, dtype=torch.float).view(-1, 1).to(DEVICE),
                torch.tensor(df_all['strike_price_K'].values, dtype=torch.float).view(-1, 1).to(DEVICE),
                torch.tensor(df_all['contract_duration_T'].values, dtype=torch.float).view(-1, 1).to(DEVICE),
                stats_all
            )
            
            # 4. ทำนาย
            with torch.no_grad():
                # ทำนายเฉพาะ 159 จุดแรก (index 0-158)
                X_history_norm = X_all_norm[0:159]
                V_history_norm_pred = model_final(X_history_norm)
            
            # 5. Denormalize
            V_history_pred = (V_history_norm_pred * stats_all['V_std']) + stats_all['V_mean']
            history_predictions = V_history_pred.cpu().numpy().flatten().tolist()
            logging.info(f"Successfully predicted {len(history_predictions)} historical points using '{model_final_name}'.")

        except Exception as e:
            logging.error(f"Error predicting historical part: {e}")
    else:
        logging.warning(f"Final model '{model_final_name}' not found. Historical plot will be empty.")
        
    # --- 4.7. สร้างกราฟรวม ---
    logging.info("Generating final comparison plot...")
    try:
        # รวมเส้นทำนาย
        # V_combined_pred = [np.nan] * 159 + forecast_predictions # History ว่าง
        V_combined_pred = history_predictions + forecast_predictions # History + Forecast
        
        # ดึงข้อมูลสำหรับพล็อต
        t_all = df_all['current_time_t'].values
        V_actual_all = df_all['close'].values
        S_actual_all = df_all['btc_close_price'].values
        
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        title_str = (
            f"Walk-Forward Prediction vs. Market Price\n"
            f"Forecast (10 points) -> RMSE: {rmse:.2f}, R: {correlation:.4f}"
        )
        fig.suptitle(title_str, fontsize=16)

        # 1. พล็อตราคาจริง (Baseline)
        ax1.plot(t_all, V_actual_all, label='Market Price (Close)', color='purple', linestyle='-')
        
        # 2. พล็อตราคาทำนาย (เส้นรวม)
        ax1.plot(t_all, V_combined_pred, label='PINN Price (History + Forecast)', color='darkorange', linestyle='--')

        # 3. พล็อตเส้นแบ่งโซน
        # หาเวลาของจุดที่ 159 (index 158) ซึ่งเป็นจุดสุดท้ายของ History
        v_line_t = df_all.iloc[158]['current_time_t']
        ax1.axvline(x=v_line_t, color='red', linestyle='-', linewidth=2, label='Forecast Start (t=159)')

        ax1.set_xlabel('Time (t)')
        ax1.set_ylabel('Option Price (V)')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 4. พล็อตราคา BTC (แกน Y ที่สอง)
        ax2 = ax1.twinx()
        ax2.plot(t_all, S_actual_all, label='BTCUSDT Price', color='lightgreen', linestyle=':', alpha=0.7)
        ax2.set_ylabel('BTCUSDT Price (S)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # 5. รวม Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = os.path.join(OUTPUT_DIR, "walk_forward_comparison_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        logging.info(f"Comparison plot saved to: {plot_path}")
        
    except Exception as e:
        logging.error(f"Failed to generate plot: {e}")

    logging.info("--- Evaluation script finished ---")


# --- 5. Run Script ---
if __name__ == '__main__':
    evaluate_walk_forward()