import json
import urllib.request
import urllib.error
import time # เปลี่ยนจาก datetime มาใช้ time

# --- 1. ตั้งค่า ---
LIQUID_OPTION_SYMBOL = "BTC-250912-116000-C"
TICKER_API_URL = f"https://eapi.binance.com/eapi/v1/ticker?symbol={LIQUID_OPTION_SYMBOL}"
MARK_API_URL = f"https://eapi.binance.com/eapi/v1/mark?symbol={LIQUID_OPTION_SYMBOL}"
SPOT_API_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

def fetch_and_combine_data():
    """
    ฟังก์ชันสำหรับดึงข้อมูลจาก Endpoint /ticker, /mark และราคา Spot
    จากนั้นนำมารวมกัน แล้วแสดงผล
    """
    print(f"Retrieving data for: {LIQUID_OPTION_SYMBOL} from 3 Endpoints...")
    
    try:
        # --- 2. ฟังก์ชันช่วยดึงและแปลงข้อมูล ---
        def get_data_from_url(url):
            with urllib.request.urlopen(url) as response:
                if response.getcode() != 200:
                    raise urllib.error.URLError(f"Server returned status code {response.getcode()}")
                data = response.read().decode('utf-8')
                return json.loads(data)

        # --- 3. ดึงข้อมูลจากทั้งสามแหล่ง ---
        print(" - Retrieving Ticker data...")
        ticker_data_list = get_data_from_url(TICKER_API_URL)
        
        print(" - Retrieving Mark Price and Greeks...")
        mark_data_list = get_data_from_url(MARK_API_URL)
        
        print(" - Retrieving BTC Spot Price...")
        spot_data_dict = get_data_from_url(SPOT_API_URL)

        # เปลี่ยนแปลง: สร้าง timestamp เป็น Unix milliseconds
        fetch_timestamp = int(time.time() * 1000)

        ticker_data = ticker_data_list[0]
        mark_data = mark_data_list[0]
        
        # --- 4. รวมข้อมูล ---
        combined_data = {}
        combined_data.update(ticker_data)
        combined_data.update(mark_data)
        combined_data['spotPrice'] = spot_data_dict['price']
        combined_data['fetchTimestamp'] = fetch_timestamp
        
        # --- 5. แสดงผลข้อมูลที่รวมแล้ว ---
        print("\n" + "="*50)
        print(f"Success! Displaying all combined fields:")
        print("="*50)
        
        for key in sorted(combined_data.keys()):
            value = combined_data[key]
            print(f"- {key}: {value}")
            
        print("="*50)

    except urllib.error.URLError as e:
        print(f"Error connecting or fetching data: {e}")
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error processing JSON data: {e}")

if __name__ == "__main__":
    fetch_and_combine_data()