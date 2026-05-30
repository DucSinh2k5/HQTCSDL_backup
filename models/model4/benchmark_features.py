# ==============================================
# MODULE 4 — BENCHMARK OUTPERFORMANCE MODEL
# File: benchmark_features.py
# Mục đích: Đọc dữ liệu từ ClickHouse,
#           tính features, gán label outperform
# ==============================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import clickhouse_connect

# ==================
# 1. KẾT NỐI DB
# ==================
load_dotenv()  # đọc file .env

def get_client():
    """Tạo kết nối ClickHouse từ file .env"""
    return clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8443")),
        username=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        database=os.getenv("CLICKHOUSE_DATABASE", "default"),
        secure=True,
    )

# ==================
# ĐỌC DỮ LIỆU
# ==================
# ==================
# 2. LẤY TOP 100 MÃ ỔN ĐỊNH
# ==================
#SỬA
def get_top100_symbols(client) -> list:
    """
    Lọc 100 mã cổ phiếu ổn định nhất từ stock.stock_prices
    Tiêu chí:
    - Có ít nhất 2000 ngày giao dịch (lịch sử dài)
    - Volume trung bình cao nhất (thanh khoản tốt)
    → Gần với VN100 thật nhất mà không cần hardcode
    """
    query = """
        SELECT
            symbol,
            COUNT(*) AS so_ngay_giao_dich,
            AVG(volume) AS volume_trung_binh
        FROM stock.stock_prices
        GROUP BY symbol
        HAVING so_ngay_giao_dich >= 2000
        ORDER BY volume_trung_binh DESC
        LIMIT 100
    """
    print("[model4] Đang lọc 100 mã ổn định nhất...")
    df = client.query_df(query)
    symbols = df["symbol"].tolist()
    print(f"[model4] Lọc xong: {len(symbols)} mã")
    print(f"[model4] Top 5 mã: {symbols[:5]}")
    return symbols


def load_stock_prices(client, symbols: list) -> pd.DataFrame:
    """
    Đọc dữ liệu giá từ ClickHouse
    Chỉ lấy 100 mã ổn định nhất
    """
    # Chuyển list symbols thành chuỗi SQL
    symbols_str = ", ".join([f"'{s}'" for s in symbols])

    query = f"""
        SELECT
            symbol,
            date AS trading_date,
            open,
            high,
            low,
            close,
            volume
        FROM stock.stock_prices
        WHERE symbol IN ({symbols_str})
        ORDER BY symbol, trading_date
    """
    print("[model4] Đang đọc dữ liệu từ ClickHouse...")
    df = client.query_df(query)
    print(f"[model4] Đọc xong: {len(df):,} dòng, "
          f"{df['symbol'].nunique()} mã cổ phiếu")
    return df
#SỬA
# ==================
# 3. TÍNH BENCHMARK
# ==================
def calc_benchmark_return(df: pd.DataFrame,
                          horizon: int = 5) -> pd.DataFrame:
    """
    Tính return của benchmark (trung bình toàn thị trường)
    mỗi ngày trong N ngày tới

    horizon: số ngày tới để tính return (mặc định 5 ngày)
    """
    print(f"[model4] Tính benchmark return ({horizon} ngày)...")

    # Tính return N ngày tới cho từng mã
    # pct_change(-N) = (giá N ngày sau - giá hôm nay) / giá hôm nay
    df = df.sort_values(["symbol", "trading_date"])
    df["future_return"] = (
        df.groupby("symbol")["close"]
        .pct_change(-horizon)  # âm = nhìn về phía trước
        * -1                   # đảo dấu để ra đúng chiều
    )

    # Benchmark = trung bình return của TẤT CẢ mã mỗi ngày
    benchmark = (
        df.groupby("trading_date")["future_return"]
        .mean()
        .rename("benchmark_return")
        .reset_index()
    )

    print(f"[model4] Tính xong benchmark: "
          f"{len(benchmark)} ngày giao dịch")
    return df, benchmark  # ← thêm df vào đây!

# ==================
# 4. GÁN LABEL
# ==================
def create_labels(df: pd.DataFrame,
                  benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Gán nhãn outperform:
    label = 1 nếu return cổ phiếu > return benchmark
    label = 0 nếu ngược lại
    """
    print("[model4] Gán nhãn outperform...")

    # Merge dữ liệu với benchmark theo ngày
    df = df.merge(benchmark, on="trading_date", how="left")

    # Gán label
    df["label"] = (
        df["future_return"] > df["benchmark_return"]
    ).astype(int)

    # Xóa dòng không có label (những ngày cuối không có
    # đủ N ngày tương lai để tính)
    df = df.dropna(subset=["future_return", "benchmark_return"])

    # Thống kê label
    label_counts = df["label"].value_counts()
    print(f"[model4] Label = 1 (outperform): "
          f"{label_counts.get(1, 0):,} dòng")
    print(f"[model4] Label = 0 (không outperform): "
          f"{label_counts.get(0, 0):,} dòng")

    return df

# ==================
# 5. TÍNH FEATURES
# ==================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính các features kỹ thuật từ lịch sử giá OHLCV
    Tái sử dụng features tương tự model5
    """
    print("[model4] Tính features kỹ thuật...")

    g = df.groupby("symbol")

    # --- Return features ---
    for window in [1, 3, 5, 10, 20]:
        df[f"return_{window}d"] = g["close"].pct_change(window)

    # --- Moving Average features ---
    for window in [5, 20, 50]:
        df[f"ma_{window}"] = g["close"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w).mean()
        )

    # --- MA ratios ---
    df["price_vs_ma20"] = (df["close"] / df["ma_20"]) - 1
    df["ma5_vs_ma20"]   = (df["ma_5"] / df["ma_20"]) - 1

    # --- Volatility features ---
    df["volatility_5d"] = g["return_1d"].transform(
        lambda s: s.rolling(5, min_periods=5).std()
    )
    df["volatility_20d"] = g["return_1d"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    df["volatility_change"] = (
        df["volatility_5d"] / df["volatility_20d"]
    ) - 1

    # --- Drawdown features ---
    df["rolling_max_20d"] = g["close"].transform(
        lambda s: s.rolling(20, min_periods=20).max()
    )
    df["drawdown_20d"] = (df["close"] / df["rolling_max_20d"]) - 1

    # --- Volume features ---
    df["volume_ma_5"]  = g["volume"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )
    df["volume_ma_20"] = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    df["volume_ratio_5_20"] = df["volume_ma_5"] / df["volume_ma_20"]
    df["volume_change_1d"]  = g["volume"].pct_change(1)

    # --- Candlestick features ---
    hl_range = df["high"] - df["low"]
    df["daily_range"]   = hl_range / df["close"]
    df["body_ratio"]    = (
        (df["close"] - df["open"]).abs() / hl_range
    ).where(hl_range > 0, 0)
    df["close_position"] = (
        (df["close"] - df["low"]) / hl_range
    ).where(hl_range > 0, 0.5)

    # Xử lý inf và NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    print(f"[model4] Tính xong features!")
    return df

# ==================
# MAIN
# ==================
if __name__ == "__main__":

    # Bước 1: Kết nối DB
    client = get_client()
    #SỬA
    # Bước 2: Lấy 100 mã ổn định nhất
    top100 = get_top100_symbols(client)

    # Bước 3: Đọc dữ liệu chỉ 100 mã đó
    df = load_stock_prices(client, top100)
    #SỬA
    # Bước 3: Tính benchmark
    df, benchmark = calc_benchmark_return(df, horizon=5)

    # Bước 4: Gán label
    df = create_labels(df, benchmark)

    # Bước 5: Tính features
    df = create_features(df)

    # Xem kết quả
    print("\n[model4] Mẫu dữ liệu sau khi xử lý:")
    print(df[["symbol", "trading_date", "close",
              "return_5d", "ma_20", "label"]].head(10))

    print(f"\n[model4] Tổng số dòng có đủ features: "
          f"{df.dropna().shape[0]:,}")
    
    # Bước 6: Lưu features ra CSV
    OUTPUT_PATH = Path("model4/output/benchmark_features.csv")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Danh sách tất cả features cần có
    FEATURE_COLUMNS = [
        "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
        "ma_5", "ma_20", "ma_50",
        "price_vs_ma20", "ma5_vs_ma20",
        "volatility_5d", "volatility_20d", "volatility_change",
        "rolling_max_20d", "drawdown_20d",
        "volume_ma_5", "volume_ma_20", "volume_ratio_5_20", "volume_change_1d",
        "daily_range", "body_ratio", "close_position",
    ]

    # Chỉ lưu dòng có đủ TẤT CẢ features + label
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[model4] Đã lưu {len(df_clean):,} dòng "
          f"vào {OUTPUT_PATH}")