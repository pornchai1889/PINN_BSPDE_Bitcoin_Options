import json
import urllib.request
import urllib.error
import time
import csv
import os
import schedule
from datetime import datetime, timezone, timedelta

# --- 1. ตั้งค่า ---
LIQUID_OPTION_SYMBOL = "BTC-250915-116000-C"
TICKER_API_URL = f"https://eapi.binance.com/eapi/v1/ticker?symbol={LIQUID_OPTION_SYMBOL}"
MARK_API_URL = f"https://eapi.binance.com/eapi/v1/mark?symbol={LIQUID_OPTION_SYMBOL}"
SPOT_API_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

DIRECTORY_PATH = os.path.join("btcusdt_options_call_V2", "realtime_data_logger")
FILENAME = f"{LIQUID_OPTION_SYMBOL}.csv"
OUTPUT_CSV_FILE = os.path.join(DIRECTORY_PATH, FILENAME)

# --- 2. ตั้งค่าการเก็บข้อมูล (เหมือนเดิม) ---
DATA_COLLECTION_LIMIT = 5000
run_count = 0

def fetch_and_combine_data():
    """
    ฟังก์ชันสำหรับดึงข้อมูลและเพิ่มฟิลด์เวลาที่อ่านง่าย
    """
    print(f"   - Retrieving data for: {LIQUID_OPTION_SYMBOL}...")
    try:
        def get_data_from_url(url):
            with urllib.request.urlopen(url) as response:
                if response.getcode() != 200:
                    raise urllib.error.URLError(f"Server returned status code {response.getcode()}")
                data = response.read().decode('utf-8')
                return json.loads(data)
        
        ticker_data_list = get_data_from_url(TICKER_API_URL)
        mark_data_list = get_data_from_url(MARK_API_URL)
        spot_data_dict = get_data_from_url(SPOT_API_URL)
        
        fetch_timestamp = int(time.time() * 1000)

        # --- ส่วนที่เพิ่มเข้ามา: แปลง Unix Timestamp เป็นเวลาที่อ่านง่าย ---
        timestamp_s = fetch_timestamp / 1000
        bkk_tz = timezone(timedelta(hours=7))
        utc_dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        bkk_dt = utc_dt.astimezone(bkk_tz)
        
        timestamp_bkk_str = bkk_dt.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_utc_str = utc_dt.strftime("%Y-%m-%d %H:%M:%S")
        # --- จบส่วนที่เพิ่มเข้ามา ---

        combined_data = {}
        combined_data.update(ticker_data_list[0])
        combined_data.update(mark_data_list[0])
        combined_data['spotPrice'] = spot_data_dict['price']
        combined_data['fetchTimestamp'] = fetch_timestamp
        # เพิ่มฟิลด์การแปลงเวลาแล้วเข้าไปในข้อมูล
        combined_data['timestamp_bkk'] = timestamp_bkk_str
        combined_data['timestamp_utc'] = timestamp_utc_str
        
        return combined_data

    except urllib.error.URLError as e:
        print(f"   - Error connecting or fetching data: {e}")
    except (json.JSONDecodeError, IndexError) as e:
        print(f"   - Error processing JSON data: {e}")
    return None

def write_to_csv(data_dict, filename):
    """
    ฟังก์ชันสำหรับเขียนข้อมูลลงไฟล์ CSV
    แก้ไขให้ timestamp_bkk อยู่คอลัมน์แรกเสมอ
    """
    file_exists = os.path.isfile(filename)
    
    # --- จัดลำดับคอลัมน์ใหม่ ---
    # ดึง key ทั้งหมดที่ไม่ใช่ 'timestamp_bkk' แล้วเรียงตามตัวอักษร
    other_keys = sorted([key for key in data_dict.keys() if key != 'timestamp_bkk'])
    # สร้าง list ของ fieldnames โดยเอา 'timestamp_bkk' ขึ้นก่อน
    fieldnames = ['timestamp_bkk'] + other_keys
    # --- จบส่วนที่แก้ไข ---
    
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(data_dict)
    print(f"   - Data saved successfully to {filename}")

def data_collection_job():
    global run_count
    run_count += 1
    print(f"\n---> [{time.strftime('%Y-%m-%d %H:%M:%S')}] Collecting data set {run_count} of {DATA_COLLECTION_LIMIT}...")
    data = fetch_and_combine_data()
    if data:
        write_to_csv(data, OUTPUT_CSV_FILE)
    else:
        print("   - Failed to fetch data for this set.")
    if run_count >= DATA_COLLECTION_LIMIT:
        print("\nCollection limit reached. Stopping schedule.")
        return schedule.CancelJob

def main():
    os.makedirs(DIRECTORY_PATH, exist_ok=True)
    print(f"Data will be saved to: {OUTPUT_CSV_FILE}")
    print("Scheduler started. Waiting for the next minute to begin...")
    schedule.every().minute.at(":00").do(data_collection_job)
    while True:
        schedule.run_pending()
        if not schedule.jobs:
            break
        time.sleep(1)
    print("\nData collection process finished.")

if __name__ == "__main__":
    main()