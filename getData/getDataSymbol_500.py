# Crawl dữ liệu 500 mã cổ phiếu
# Batch size = 10
# Sleep = 90s
# Source = KBS/VCI

import os
import time
import ast
import random
import re
import pandas as pd

# API mới của vnstock
from vnstock.api.quote import Quote
from vnstock.core.exceptions import RateLimitError
from tenacity import RetryError

# =========================
# CONFIG
# =========================

DATA_DIR = r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\getData\data\data_clean"
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "Data_500_stocks_2026.csv"
)

SYMBOL_FILE = r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\getData\symbol500.txt"

SOURCE = "KBS"

START_DATE = "2026-05-04"
END_DATE = "2026-05-04"

FAST_MODE = True
REQUESTS_PER_MINUTE = 20
MIN_REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE

BATCH_SIZE = 10
SLEEP_TIME = 90
REQUEST_DELAY_RANGE = (2, 5)
RATE_LIMIT_SLEEP = 60
MAX_RATE_LIMIT_RETRIES = 3


def _parse_retry_after(message: str, default: int = RATE_LIMIT_SLEEP) -> int:
    match = re.search(r"(\d+)\s*seconds", message)
    if match:
        return max(1, int(match.group(1)))
    match = re.search(r"Chờ\s*(\d+)", message)
    if match:
        return max(1, int(match.group(1)))
    return max(1, default)


def _throttle(last_time: float, min_interval: float) -> float:
    now = time.monotonic()
    sleep_for = min_interval - (now - last_time)
    if sleep_for > 0:
        time.sleep(sleep_for)
    return time.monotonic()

# =========================
# Đọc danh sách mã
# =========================

all_symbols = []

with open(SYMBOL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        symbols = ast.literal_eval(line)
        all_symbols.extend(symbols)

# lấy tối đa 500 mã
all_symbols = all_symbols[:500]
print(f"Tổng số mã: {len(all_symbols)}")

if FAST_MODE:
    BATCH_SIZE = len(all_symbols)
    SLEEP_TIME = 0
    REQUEST_DELAY_RANGE = (0.0, 0.0)

# =========================
# Chia batch
# =========================
batches = [
    all_symbols[i:i + BATCH_SIZE]
    for i in range(0, len(all_symbols), BATCH_SIZE)
]

print(f"Tổng batch: {len(batches)}")

# =========================
# XÓA FILE CŨ NẾU TỒN TẠI
# =========================
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# =========================
# CRAWL
# =========================
last_request_time = 0.0
for batch_index, batch in enumerate(batches, start=1):
    print("\n" + "=" * 50)
    print(f"Batch {batch_index}/{len(batches)}")
    print("=" * 50)

    batch_df_list = []
    for symbol in batch:
        print(f"\nĐang lấy: {symbol}")

        rate_limit_attempts = 0
        df_history = None
        status_printed = False

        while True:
            try:
                # =========================
                # API mới
                # =========================
                q = Quote(
                    symbol=symbol,
                    source=SOURCE
                )

                last_request_time = _throttle(
                    last_request_time,
                    MIN_REQUEST_INTERVAL
                )
                df_history = q.history(
                    start=START_DATE,
                    end=END_DATE
                )
                break

            except SystemExit as e:
                msg = str(e)
                if "Rate limit" not in msg:
                    raise

                rate_limit_attempts += 1
                if rate_limit_attempts > MAX_RATE_LIMIT_RETRIES:
                    print(f"Vượt quá retry rate limit: {symbol}")
                    status_printed = True
                    break

                wait_seconds = _parse_retry_after(msg)
                wait_seconds = max(1, wait_seconds)
                print(f"Rate limit, sleep {wait_seconds}s...")
                time.sleep(wait_seconds)

            except RateLimitError as e:
                rate_limit_attempts += 1
                if rate_limit_attempts > MAX_RATE_LIMIT_RETRIES:
                    print(f"Vượt quá retry rate limit: {symbol}")
                    status_printed = True
                    break

                wait_seconds = RATE_LIMIT_SLEEP
                if getattr(e, "details", None):
                    wait_seconds = e.details.get("retry_after", RATE_LIMIT_SLEEP)
                wait_seconds = max(1, wait_seconds)
                print(f"Rate limit, sleep {wait_seconds}s...")
                time.sleep(wait_seconds)

            except RetryError as e:
                last_exc = e.last_attempt.exception()
                if last_exc and "Không tìm thấy dữ liệu" in str(last_exc):
                    print(f"Không có dữ liệu: {symbol}")
                    status_printed = True
                else:
                    print(f"Lỗi với {symbol}: {last_exc or e}")
                    status_printed = True
                break

            except ValueError as e:
                if "Không tìm thấy dữ liệu" in str(e):
                    print(f"Không có dữ liệu: {symbol}")
                    status_printed = True
                else:
                    print(f"Lỗi với {symbol}: {e}")
                    status_printed = True
                break

            except Exception as e:
                print(f"Lỗi với {symbol}: {e}")
                status_printed = True
                break

        # =========================
        # Check dữ liệu
        # =========================
        if df_history is None or df_history.empty:
            if not status_printed:
                print(f"Không có dữ liệu: {symbol}")
            time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
            continue

        # thêm mã cổ phiếu
        df_history["symbol"] = symbol
        batch_df_list.append(df_history)
        print(f"Hoàn tất: {symbol}")

        # sleep nhỏ giữa từng request
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))

    # =========================
    # GHI FILE THEO BATCH
    # =========================
    if batch_df_list:
        batch_df = pd.concat(
            batch_df_list,
            ignore_index=True
        )

        # batch đầu tiên -> ghi mới
        if batch_index == 1:
            batch_df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig",
                mode="w"
            )

        # batch sau -> append
        else:
            batch_df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig",
                mode="a",
                header=False
            )
        print(f"\nĐã ghi batch {batch_index} vào file")

    # =========================
    # Sleep giữa batch
    # =========================
    if batch_index < len(batches) and SLEEP_TIME > 0:
        random_sleep = random.randint(
            max(0, SLEEP_TIME - 10),
            SLEEP_TIME + 20
        )

        print(f"\nSleep {random_sleep}s...\n")
        time.sleep(random_sleep)

# =========================
# DONE
# =========================
print("\n" + "=" * 50)
print("HOÀN TẤT")
print(f"Lưu file tại: {OUTPUT_FILE}")
print("=" * 50)