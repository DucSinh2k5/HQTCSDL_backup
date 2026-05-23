import requests
import csv
import os
from datetime import datetime

#File này sẽ lấy dữ liệu data thực ở trên web có url dưới đây mà không lấy qua thư viện
URL = "https://iboard-query.ssi.com.vn/stock/group/VN100"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Referer": "https://iboard.ssi.com.vn/",
    "Origin": "https://iboard.ssi.com.vn",
}

DATA_DIR = "HQTCSDL_stocks/ingestion/data_crawl_112026"

def main():
    
    os.makedirs(DATA_DIR, exist_ok=True)

    r = requests.get(URL, headers=HEADERS)
    r.raise_for_status()

    json_data = r.json()
    if json_data.get("code") != "SUCCESS":
        print("API error:", json_data)
        return

    today = datetime.now().strftime("%Y-%m-%d")
    stocks = json_data["data"]

    for s in stocks:
        symbol = s.get("stockSymbol")
        if not symbol:
            continue

        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        file_exists = os.path.isfile(file_path) #Trả về True nếu file đã tồn tại, dùng để ghi dòng tiêu đề

        row = {
            "date": today,
            "symbol": symbol,
            "Open": s.get("openPrice"),
            "High": s.get("highest"),
            "Low": s.get("lowest"),
            "Volume": s.get("stockVol"),
            "Close": s.get("matchedPrice"),
        }

        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["date", "symbol", "Open", "High", "Low", "Volume", "Close"]
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    print(f"Updated {len(stocks)} stock files for {today}")

if __name__ == "__main__":
    main()
