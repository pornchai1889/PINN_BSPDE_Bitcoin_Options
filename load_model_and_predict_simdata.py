import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd

# --- 1. กำหนดพารามิเตอร์และโครงสร้างโมเดล (ต้องตรงกับตอนที่ฝึก) ---

# พารามิเตอร์ของตลาด (Market Parameters)
K = 116000.0
r = 0.05
sigma = 0.321775
T = 7 / 365.0
# T = 0.01916496
S_range = [90000.0, 140000.0]
t_range = [0.0, T]

# พารามิเตอร์โมเดล (PINN Hyperparameters)
N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS = 2, 1, 100, 8

# ค่าคงที่สำหรับ Normalization
S_min, S_max = S_range
t_min, t_max = t_range

# ตั้งค่าอุปกรณ์
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. นิยามฟังก์ชันและคลาสที่จำเป็น ---

# ตั้งค่า Font ภาษาไทย (ถ้ามี)
try:
    # The following line for setting a Thai font is commented out for English plots.
    # mpl.rcParams['font.family'] = 'Leelawadee UI'
    mpl.rcParams['axes.unicode_minus'] = False
    print("Font settings configured for plotting.")
except Exception as e:
    print(f"Warning: Could not configure font settings. Error: {e}")

# ฟังก์ชัน Normalization
def normalize(t, S):
    t_norm = (t - t_min) / (t_max - t_min)
    S_norm = (S - S_min) / (S_max - S_min)
    return t_norm, S_norm

# คลาสของโมเดล
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

model_path = os.path.join('btcusdt_options_call_V2', 'btcusdt_call_pricing_mlp', 'result_14', 'pinn_model.pth')

model = EuropeanCallPINN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS).to(DEVICE)

if not os.path.exists(model_path):
    print(f"Error: Model file not found at '{model_path}'")
    print("Please make sure the model file exists at the specified location.")
else:
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded successfully from: {model_path}")

    # --- 4. โหลดข้อมูลตลาดจำลองจากไฟล์ CSV ---
    
    data_file_path = os.path.join('btcusdt_options_call_V2', 'bitcoin_option_simulated_data', 'S_90k-140k_T_7d', 'simulated_market_data_10.csv')
    
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at '{data_file_path}'")
        print("Please run 'generate_simulated_market_data.py' first to create the data.")
    else:
        print(f"Loading simulated market data from: {data_file_path}")
        market_data_df = pd.read_csv(data_file_path)
        
        # ดึงข้อมูลจาก DataFrame มาใช้
        t_path = market_data_df['current_time_t'].values
        S_path = market_data_df['spot_price'].values

        # --- 5. ทำนายและคำนวณราคา ---
        
        # แปลงข้อมูล Path ให้อยู่ในรูปแบบที่โมเดลต้องการ (Normalized)
        t_path_norm, S_path_norm = normalize(t_path, S_path)
        X_path_test_norm = torch.tensor(np.column_stack((t_path_norm, S_path_norm)), dtype=torch.float).to(DEVICE)

        # ทำนายราคาด้วยโมเดล PINN ที่โหลดมา
        with torch.no_grad():
            y_pinn_path_norm = model(X_path_test_norm)
        
        # แปลงผลลัพธ์กลับเป็นสเกลราคาจริง (Denormalize)
        y_pinn_path_np = y_pinn_path_norm.cpu().numpy() * K

        # คำนวณราคาตามทฤษฎีเพื่อเปรียบเทียบ
        y_analytical_path = eur_call_analytical_price(torch.tensor(S_path, dtype=torch.float).to(DEVICE), torch.tensor(t_path, dtype=torch.float).to(DEVICE), K, r, sigma, T)
        y_analytical_path_np = y_analytical_path.cpu().numpy()
        
        # <<< START: MODEL EVALUATION SECTION (REVISED) >>>
        # --- 6. ประเมินความแม่นยำของโมเดล ---

        # กำหนดค่าจริงและค่าที่ทำนาย
        actual_prices = y_analytical_path_np
        predicted_prices = y_pinn_path_np.flatten() # ใช้ .flatten() เพื่อให้มี dimension เท่ากัน

        # 1. คำนวณ Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((actual_prices - predicted_prices)**2))
        
        # 2. คำนวณ Correlation Coefficient
        # np.corrcoef จะคืนค่าเป็น matrix 2x2, เราต้องการค่าที่ตำแหน่ง [0, 1]
        correlation = np.corrcoef(actual_prices, predicted_prices)[0, 1]

        print("\n--- Model Performance Evaluation ---")
        print(f"Root Mean Squared Error (RMSE)    : {rmse:.4f}")
        print(f"Correlation Coefficient (R)       : {correlation:.4f}")
        print("------------------------------------\n")
        
        # เพิ่มค่า RMSE และ Correlation ลงใน Title ของกราฟ
        plot_title = (f'Option Price Comparison (RMSE: {rmse:.2f}, R: {correlation:.4f})'
                      f'\nSimulated Path | from file')
        # <<< END: MODEL EVALUATION SECTION (REVISED) >>>

        # --- 7. พล็อตกราฟเปรียบเทียบ ---
        
        plt.figure("Simulated Path Comparison", figsize=(12, 7))
        main_ax = plt.gca()

        main_ax.plot(t_path, y_analytical_path_np, label='Analytical Price', color='dodgerblue', linewidth=2)
        main_ax.plot(t_path, y_pinn_path_np, label='PINN Price (Loaded Model)', color='darkorange', linestyle='--', linewidth=2)
        
        main_ax.set_title(plot_title)
        main_ax.set_xlabel('Time (t)')
        main_ax.set_ylabel('Option Price (V)')
        
        ax2 = main_ax.twinx()
        ax2.plot(t_path, S_path, label='BTCUSDT Price Path', color='green', alpha=0.4, linestyle=':')
        ax2.set_ylabel('BTCUSDT Price (S)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        lines, labels = main_ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        main_ax.legend(lines + lines2, labels + labels2, loc='upper left')
        main_ax.grid(True, linestyle='--', alpha=0.6)

        # --- 8. บันทึกไฟล์กราฟ ---
        
        save_dir = os.path.join('btcusdt_options_call_V2', 'load_model_and_predict_simdata')
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        run_number = 1
        while True:
            save_path = os.path.join(save_dir, f"test_comparison_{run_number}.png")
            if not os.path.exists(save_path):
                break
            run_number += 1

        plt.savefig(save_path)
        plt.close("Simulated Path Comparison")
        
        print(f"Graph saved successfully to: {save_path}")