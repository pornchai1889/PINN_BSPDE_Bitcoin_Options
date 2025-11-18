import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd

# --- 1. กำหนดพารามิเตอร์และโครงสร้างโมเดล (ต้องตรงกับตอนที่ฝึก) ---

# พารามิเตอร์โมเดล (PINN Hyperparameters)
N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS = 2, 1, 100, 8
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# พารามิเตอร์ของตลาด (Market Parameters)
r = 0.05
sigma = 0.321775
S_range = [90000.0, 140000.0]

# --- 2. นิยามฟังก์ชันและคลาสที่จำเป็น ---

# ตั้งค่า Font (ถ้ามี)
try:
    mpl.rcParams['axes.unicode_minus'] = False
    print("Font settings configured for plotting.")
except Exception as e:
    print(f"Warning: Could not configure font settings. Error: {e}")

# คลาสของโมเดล (ต้องเหมือนเดิม)
class EuropeanCallPINN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh()
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation)
        self.fch = nn.Sequential(*[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation) for _ in range(N_LAYERS)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
    def forward(self, x):
        return self.fce(self.fch(self.fcs(x)))

# ฟังก์ชันคำนวณราคาตามทฤษฎี (Analytical Solution)
def eur_call_analytical_price(S, t, K, r, sigma, T):
    t2m = T - t
    epsilon = 1e-8
    t2m = torch.clamp(t2m, min=epsilon); S = torch.clamp(S, min=epsilon)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * t2m) / (sigma * torch.sqrt(t2m))
    d2 = d1 - sigma * torch.sqrt(t2m)
    N0 = lambda value: 0.5 * (1 + torch.erf(value / (2**0.5)))
    return S * N0(d1) - K * N0(d2) * torch.exp(-r * t2m)


# --- 3. โหลดโมเดลที่ฝึกไว้ ---

model_path = os.path.join('btcusdt_options_call_V2', 'btcusdt_call_pricing_mlp', 'result_13', 'pinn_model.pth')
model = EuropeanCallPINN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS).to(DEVICE)

if not os.path.exists(model_path):
    print(f"Error: Model file not found at '{model_path}'")
else:
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded successfully from: {model_path}")

    # --- 4. โหลดข้อมูลตลาดจริงจากไฟล์ CSV ที่ระบุ ---
    data_file_path = os.path.join('btcusdt_options_call_V2', 'klines', 'BTC-250919-116000-C_Weekly_1h.csv')

    # ดึงชื่อไฟล์ออกมาจาก Path เพื่อใช้ในการตั้งชื่อแบบไดนามิก
    file_basename = os.path.splitext(os.path.basename(data_file_path))[0]
    print(f"Analyzing data from: {file_basename}")
    
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at '{data_file_path}'")
    else:
        print(f"Loading market data from: {data_file_path}")
        market_data_df = pd.read_csv(data_file_path)

        # อัปเดตพารามิเตอร์ K และ T จากข้อมูลในไฟล์ CSV
        K = market_data_df['strike_price_K'].iloc[0]
        T = market_data_df['contract_duration_T'].iloc[0]
        print(f"Parameters updated from data file: K={K}, T={T:.6f} (years)")

        # กำหนดช่วงของ S และ t สำหรับ Normalization
        S_min, S_max = S_range
        t_min, t_max = 0.0, T

        # ฟังก์ชัน Normalization (ต้องใช้พารามิเตอร์ที่อัปเดตแล้ว)
        def normalize(t, S):
            t_norm = (t - t_min) / (t_max - t_min)
            S_norm = (S - S_min) / (S_max - S_min)
            return t_norm, S_norm

        # ดึงข้อมูลจาก DataFrame มาใช้
        t_path = market_data_df['current_time_t'].values
        S_path = market_data_df['btc_close_price'].values
        market_price_path = market_data_df['close'].values

        # --- 5. ทำนายและคำนวณราคา ---
        
        t_path_norm, S_path_norm = normalize(t_path, S_path)
        X_path_test_norm = torch.tensor(np.column_stack((t_path_norm, S_path_norm)), dtype=torch.float).to(DEVICE)

        with torch.no_grad():
            y_pinn_path_norm = model(X_path_test_norm)
        
        y_pinn_path_np = y_pinn_path_norm.cpu().numpy() * K

        y_analytical_path = eur_call_analytical_price(torch.tensor(S_path, dtype=torch.float).to(DEVICE), torch.tensor(t_path, dtype=torch.float).to(DEVICE), K, r, sigma, T)
        y_analytical_path_np = y_analytical_path.cpu().numpy()
        
        # --- 6. ประเมินความแม่นยำของโมเดล ---
        actual_prices = market_price_path
        predicted_prices = y_pinn_path_np.flatten()
        rmse = np.sqrt(np.mean((actual_prices - predicted_prices)**2))
        correlation = np.corrcoef(actual_prices, predicted_prices)[0, 1]

        print("\n--- Model Performance Evaluation ---")
        print(f"Root Mean Squared Error (RMSE)    : {rmse:.4f}")
        print(f"Correlation Coefficient (R)       : {correlation:.4f}")
        print("------------------------------------\n")
        
        plot_title = (f'Market vs. Model vs. Analytical Price Comparison'
                      f'\n(RMSE: {rmse:.2f}, R: {correlation:.4f}) | {file_basename}')

        # --- 7. พล็อตกราฟเปรียบเทียบ (กราฟที่ 1: มี Analytical Price) ---
        
        plt.figure("Market Data Comparison", figsize=(12, 7))
        main_ax = plt.gca()

        main_ax.plot(t_path, market_price_path, label='Market Price (Close)', color='purple', linestyle='-', markersize=4, alpha=0.7)
        main_ax.plot(t_path, y_analytical_path_np, label='Analytical Price', color='dodgerblue', linewidth=2, linestyle='-.')
        main_ax.plot(t_path, y_pinn_path_np, label='PINN Price', color='darkorange', linestyle='--', linewidth=2)

        
        main_ax.set_title(plot_title)
        main_ax.set_xlabel('Time (t)')
        main_ax.set_ylabel('Option Price (V)')
        
        ax2 = main_ax.twinx()
        ax2.plot(t_path, S_path, label='BTCUSDT Price', color='green', alpha=0.4, linestyle=':')
        ax2.set_ylabel('BTCUSDT Price (S)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        lines, labels = main_ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        main_ax.legend(lines + lines2, labels + labels2, loc='upper right')
        main_ax.grid(True, linestyle='--', alpha=0.6)


        # --- 8. สร้างไดเรกทอรีและบันทึกไฟล์กราฟ (กราฟที่ 1) ---
        
        # กำหนดไดเรกทอรีหลัก
        base_save_dir = os.path.join('btcusdt_options_call_V2', 'load_model_and_predict_realdata')
        
        # ค้นหาหมายเลข result_X ถัดไปที่ว่าง
        run_number = 1
        while True:
            save_dir = os.path.join(base_save_dir, f"result_{run_number}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"Created directory: {save_dir}")
                break
            run_number += 1

        # กำหนด Path สำหรับบันทึกกราฟแรก
        save_path = os.path.join(save_dir, f"{file_basename}_comparison.png")
        
        plt.savefig(save_path)
        plt.close("Market Data Comparison")
        
        print(f"Graph saved successfully to: {save_path}")

        # --- 9. สร้างและบันทึกกราฟที่ 2 (ไม่มี Analytical Price) ---

        # ตั้งชื่อ Title สำหรับกราฟที่สอง
        plot_title_no_analytical = (f'Market vs. PINN Price Comparison'
                                     f'\n(RMSE: {rmse:.2f}, R: {correlation:.4f}) | {file_basename}')

        plt.figure("Market vs PINN Comparison", figsize=(12, 7))
        main_ax_2 = plt.gca()

        # พล็อตกราฟโดยไม่มีเส้น Analytical Price
        main_ax_2.plot(t_path, market_price_path, label='Market Price (Close)', color='purple', linestyle='-', markersize=4, alpha=0.7)
        main_ax_2.plot(t_path, y_pinn_path_np, label='PINN Price', color='darkorange', linestyle='--', linewidth=2)

        
        main_ax_2.set_title(plot_title_no_analytical)
        main_ax_2.set_xlabel('Time (t)')
        main_ax_2.set_ylabel('Option Price (V)')
        
        ax2_2 = main_ax_2.twinx()
        ax2_2.plot(t_path, S_path, label='BTCUSDT Price', color='green', alpha=0.4, linestyle=':')
        ax2_2.set_ylabel('BTCUSDT Price (S)', color='green')
        ax2_2.tick_params(axis='y', labelcolor='green')
        
        lines, labels = main_ax_2.get_legend_handles_labels()
        lines2, labels2 = ax2_2.get_legend_handles_labels()
        main_ax_2.legend(lines + lines2, labels + labels2, loc='upper right')
        main_ax_2.grid(True, linestyle='--', alpha=0.6)

        # บันทึกไฟล์กราฟที่ 2 ในไดเรกทอรีเดียวกัน
        save_path_2 = os.path.join(save_dir, f"{file_basename}_comparison_no_analytical.png")
        
        plt.savefig(save_path_2)
        plt.close("Market vs PINN Comparison")

        print(f"Second graph (no analytical) saved successfully to: {save_path_2}")
        