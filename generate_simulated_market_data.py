import numpy as np
import pandas as pd
import os
from datetime import datetime

def generate_and_save_market_data(K, T, S_range, sigma, n_steps, S0=None):
    """
    สร้างข้อมูลตลาดจำลอง (เส้นทางราคา BTCUSDT) และบันทึกลงไฟล์ CSV

    Args:
        K (float): ราคาใช้สิทธิ (Strike Price).
        T (float): เวลาที่เหลือจนหมดอายุ (เป็นเศษส่วนของปี).
        S_range (list): ขอบเขตของราคา [S_min, S_max].
        sigma (float): ความผันผวน (Volatility).
        n_steps (int, optional): จำนวนขั้นในการจำลอง.
        S0 (float, optional): ราคาเริ่มต้น ถ้าไม่กำหนดจะใช้ K เป็นค่าเริ่มต้น. Defaults to None.
    """
    if S0 is None:
        S0 = K  # ใช้ราคา Strike เป็นราคาเริ่มต้นหากไม่ระบุ

    # --- 1. สร้างเส้นทางราคาจำลอง (Simulated Path) ---
    current_time_t = np.linspace(0, T, n_steps)
    t2m = T - current_time_t
    # การจำลองแบบ Geometric Brownian Motion (GBM)
    returns = np.random.randn(n_steps) * sigma * np.sqrt(T / n_steps)
    S_path = S0 * np.exp(np.cumsum(returns))

    print(f"Generated {n_steps} simulated data points.")
    
    # --- 2. สร้าง DataFrame ---
    market_data_df = pd.DataFrame({
        'current_time_t': current_time_t,
        'time_to_maturity': t2m,
        'spot_price': S_path
    })

    # --- 3. จัดการ Directory และชื่อไฟล์สำหรับบันทึก ---
    # สร้างชื่อโฟลเดอร์ตามขอบเขตราคาและวันหมดอายุ
    s_min_str = f"{int(S_range[0] / 1000)}k"
    s_max_str = f"{int(S_range[1] / 1000)}k"
    T_days = int(T * 365)
    
    folder_name = f"S_{s_min_str}-{s_max_str}_T_{T_days}d"
    
    # สร้าง Path เต็มตามโครงสร้างที่ต้องการ
    base_dir = "btcusdt_options_call_V2"
    database_name = "bitcoin_option_simulated_data"
    data_dir = os.path.join(base_dir, database_name, folder_name)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created new directory: {data_dir}")
        
    # --- 4. ค้นหาเลขไฟล์ล่าสุดและกำหนดชื่อไฟล์ใหม่ ---
    file_prefix = "simulated_market_data"
    run_number = 1
    while True:
        output_filename = f"{file_prefix}_{run_number}.csv"
        full_path = os.path.join(data_dir, output_filename)
        if not os.path.exists(full_path):
            break
        run_number += 1

    # --- 5. บันทึกไฟล์ CSV ---
    market_data_df.to_csv(full_path, index=False)
    print(f"Simulated market data successfully saved to: {full_path}")
    
if __name__ == '__main__':
    # --- กำหนดพารามิเตอร์ของตลาด (Market Parameters) ---
    # สามารถปรับค่าเหล่านี้เพื่อสร้างข้อมูลจำลองชุดต่างๆ ได้
    K_param = 116000.0
    T_param = 7 / 365.0
    #T_param = 0.01916496
    S_range_param = [90000.0, 140000.0]
    sigma_param = 0.321775
    
    # เรียกใช้ฟังก์ชันเพื่อสร้างและบันทึกข้อมูล
    generate_and_save_market_data(
        K=K_param,
        T=T_param,
        S_range=S_range_param,
        sigma=sigma_param,
        n_steps=168, # จำนวนชั่วโมงใน 7 วัน
        S0=115000.0 # ราคาเริ่มต้น สามารถปรับได้
    )