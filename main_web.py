from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

MODEL1_DIR = PROJECT_ROOT / "models" / "model1"
MODEL2_DIR = PROJECT_ROOT / "models" / "model2"
MODEL3_DIR = PROJECT_ROOT / "models" / "model3"

MODEL1_PATH = MODEL1_DIR / "models" / "price_forecasting_xgb.pkl"
MODEL2_PATH = MODEL2_DIR / "models" / "future_return_lgbm.pkl"
MODEL3_PATH = MODEL3_DIR / "models" / "trading_signal_xgb_classifier.pkl"

MODEL1_REPORT_DIR = MODEL1_DIR / "reports"
MODEL1_PREDICTION_LOG_PATH = MODEL1_REPORT_DIR / "streamlit_predictions.csv"
MODEL1_LATEST_PREDICTION_PATH = MODEL1_REPORT_DIR / "latest_streamlit_prediction.json"

CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "stock")
FEATURES_DATABASE = os.getenv("CLICKHOUSE_SOURCE_DATABASE", "stock")
FEATURES_TABLE = os.getenv("CLICKHOUSE_TABLE", "features_all")
PRICE_TABLE = "stock_prices"
SYMBOL_TABLE = "stock_symbols"

MODEL2_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "encode_sector",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "ma_5",
    "ma_20",
    "ma_50",
    "price_vs_ma20",
    "ma5_vs_ma20",
    "volatility_5d",
    "volatility_20d",
    "volatility_change",
    "rolling_max_20d",
    "drawdown_20d",
    "volume_ma_5",
    "volume_ma_20",
    "volume_ratio_5_20",
    "volume_change_1d",
    "daily_range",
    "body_ratio",
    "close_position",
]

FEATURE_DEFINITIONS = [
    ("return_1d", "Lợi suất 1 phiên gần nhất."),
    ("return_3d", "Lợi suất 3 phiên gần nhất."),
    ("return_5d", "Lợi suất 5 phiên gần nhất."),
    ("return_10d", "Lợi suất 10 phiên gần nhất."),
    ("return_20d", "Lợi suất 20 phiên gần nhất."),
    ("ma_5, ma_20, ma_50", "Trung bình động giá đóng cửa theo 5/20/50 phiên."),
    ("price_vs_ma20", "Khoảng cách tương đối giữa giá đóng cửa và MA20."),
    ("ma5_vs_ma20", "Chênh lệch tương đối giữa MA5 và MA20."),
    ("volatility_5d", "Độ biến động lợi suất 1 ngày trong cửa sổ 5 phiên."),
    ("volatility_20d", "Độ biến động lợi suất 1 ngày trong cửa sổ 20 phiên."),
    ("volatility_change", "Mức thay đổi biến động ngắn hạn so với dài hạn."),
    ("rolling_max_20d", "Giá đóng cửa cao nhất trong 20 phiên gần nhất."),
    ("drawdown_20d", "Mức sụt giảm từ đỉnh 20 phiên."),
    ("volume_ma_5, volume_ma_20", "Trung bình khối lượng giao dịch 5/20 phiên."),
    ("volume_ratio_5_20", "Tỷ lệ volume MA5 so với volume MA20."),
    ("volume_change_1d", "Tốc độ thay đổi khối lượng so với phiên trước."),
    ("daily_range", "Biên độ trong ngày: high - low so với close."),
    ("body_ratio", "Tỷ lệ thân nến so với biên độ trong ngày."),
    ("close_position", "Vị trí giá đóng cửa trong khoảng low-high."),
    ("encode_sector", "Mã hóa ngành/lĩnh vực của cổ phiếu."),
]

MART_DESCRIPTIONS = {
    "stock.stock_prices": "Dữ liệu giao dịch theo ngày.",
    "stock.features_all": "Dữ liệu OHLCV đã được feature engineering.",
    "stock.stock_symbols": "Danh mục cổ phiếu, tên công ty và ngành.",
    "stock.symbol_sector_encoding": "Bảng mã hóa ngành theo symbol.",
    "stock.mart_future_return_prediction": "Mart dự đoán future return của Model 2.",
    "stock.model4_benchmark_predictions": "Kết quả dự đoán outperform benchmark của Model 4.",
    "stock_mart.mart_model1_price_forecast": "Mart dự báo return/giá của Model 1.",
    "stock_mart.mart_model1_top_expected_return": "Top cổ phiếu kỳ vọng tăng theo Model 1.",
    "stock_mart.mart_model1_backtest_daily": "Backtest hằng ngày của Model 1.",
    "stock_mart.mart_model1_metrics": "Metrics của Model 1.",
    "stock_mart.mart_model3_trade_signal": "Mart tín hiệu giao dịch của Model 3.",
    "stock_mart_model5_risk_prediction.risk_features": "Feature và label rủi ro của Model 5.",
    "stock_mart_model5_risk_prediction.risk_predictions": "Dự đoán xác suất rủi ro của Model 5.",
    "stock_mart_model5_risk_prediction.risk_test_evaluation": "Đánh giá test set của Model 5.",
    "stock_mart_model5_risk_prediction.mart_risk_alerts": "Mart cảnh báo rủi ro của Model 5.",
}


st.set_page_config(page_title="HQTCSDL Stocks", layout="wide")


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def full_table_name(database: str, table: str) -> str:
    return f"{quote_identifier(database)}.{quote_identifier(table)}"


def sql_string(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def display_error(message: str, exc: Exception) -> None:
    st.error(message)
    st.caption(str(exc))


@st.cache_resource
def get_clickhouse_client():
    try:
        import clickhouse_connect
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện clickhouse-connect. Cài bằng: pip install clickhouse-connect"
        ) from exc

    host = os.getenv("CLICKHOUSE_HOST")
    username = os.getenv("CLICKHOUSE_USER", os.getenv("CLICKHOUSE_USERNAME", "default"))
    password = os.getenv("CLICKHOUSE_PASSWORD")
    port = int(os.getenv("CLICKHOUSE_PORT") or "8443")
    secure = os.getenv("CLICKHOUSE_SECURE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    if not host or not username or not password:
        raise RuntimeError(
            "Thiếu cấu hình ClickHouse. Cần CLICKHOUSE_HOST, CLICKHOUSE_USER, "
            "CLICKHOUSE_PASSWORD trong .env hoặc biến môi trường."
        )

    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        database=os.getenv("CLICKHOUSE_DATABASE", "stock"),
        secure=secure,
    )


@st.cache_data(ttl=300, show_spinner=False)
def run_query(query: str) -> pd.DataFrame:
    return get_clickhouse_client().query_df(query)


@st.cache_data(ttl=300, show_spinner=False)
def table_exists(database: str, table: str) -> bool:
    result = run_query(
        f"""
        SELECT count() AS cnt
        FROM system.tables
        WHERE database = {sql_string(database)}
          AND name = {sql_string(table)}
        """
    )
    return not result.empty and int(result.iloc[0]["cnt"]) > 0


@st.cache_data(ttl=300, show_spinner=False)
def get_table_columns(database: str, table: str) -> list[str]:
    result = run_query(
        f"""
        SELECT name
        FROM system.columns
        WHERE database = {sql_string(database)}
          AND table = {sql_string(table)}
        ORDER BY position
        """
    )
    if result.empty:
        return []
    return result["name"].astype(str).tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_symbols() -> list[str]:
    query = f"""
        SELECT symbol
        FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
        GROUP BY symbol
        ORDER BY symbol
    """
    result = run_query(query)
    if result.empty or "symbol" not in result.columns:
        return []
    return result["symbol"].astype(str).str.upper().sort_values().tolist()


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def pct(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def number(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.0f}"


def add_model_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@st.cache_resource
def load_model1_artifact():
    if not MODEL1_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL1_PATH}")
    add_model_path(MODEL1_DIR)
    saved = joblib.load(MODEL1_PATH)
    return saved


@st.cache_resource
def load_model2_artifact():
    if not MODEL2_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL2_PATH}")
    return joblib.load(MODEL2_PATH)


@st.cache_resource
def load_model3_artifact():
    if not MODEL3_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL3_PATH}")
    saved = joblib.load(MODEL3_PATH)
    return (
        saved["model"],
        saved["features"],
        saved.get("signal_labels", {0: "SELL", 1: "HOLD", 2: "BUY"}),
    )


def fetch_feature_rows(symbol: str | None = None, trading_date=None, limit: int = 500) -> pd.DataFrame:
    where = []
    if symbol:
        where.append(f"upper(trim(symbol)) = upper(trim({sql_string(symbol)}))")
    if trading_date:
        where.append(f"toDate(trading_date) = toDate({sql_string(trading_date)})")

    where_sql = "WHERE " + " AND ".join(where) if where else ""
    query = f"""
        SELECT *
        FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
        {where_sql}
        ORDER BY trading_date DESC, symbol
        LIMIT {int(limit)}
    """
    result = run_query(query)
    result.columns = [str(col).strip() for col in result.columns]
    if "trading_date" in result.columns:
        result["trading_date"] = pd.to_datetime(result["trading_date"], errors="coerce")
    return result


def add_prediction_signal(predicted_return: float) -> str:
    if predicted_return >= 0.03:
        return "STRONG_BUY"
    if predicted_return >= 0.01:
        return "BUY"
    if predicted_return > -0.01:
        return "HOLD"
    if predicted_return > -0.03:
        return "SELL"
    return "STRONG_SELL"


def apply_confidence_adjusted_signals(
    df: pd.DataFrame,
    min_action_probability: float = 0.60,
    min_action_margin: float = 0.0,
) -> pd.DataFrame:
    result_df = df.copy()
    probabilities = result_df[
        ["sell_probability", "hold_probability", "buy_probability"]
    ].apply(pd.to_numeric, errors="coerce")
    buy_edge = probabilities["buy_probability"] - probabilities["sell_probability"]
    sell_edge = probabilities["sell_probability"] - probabilities["buy_probability"]

    adjusted_signal = np.full(len(result_df), "HOLD", dtype=object)
    adjusted_label = np.full(len(result_df), 1, dtype=int)

    buy_mask = (
        (probabilities["buy_probability"] >= min_action_probability)
        & (probabilities["buy_probability"] >= probabilities["hold_probability"])
        & (buy_edge >= min_action_margin)
    )
    sell_mask = (
        (probabilities["sell_probability"] >= min_action_probability)
        & (probabilities["sell_probability"] >= probabilities["hold_probability"])
        & (sell_edge >= min_action_margin)
    )

    adjusted_signal[buy_mask.to_numpy()] = "BUY"
    adjusted_label[buy_mask.to_numpy()] = 2
    adjusted_signal[sell_mask.to_numpy()] = "SELL"
    adjusted_label[sell_mask.to_numpy()] = 0

    result_df["adjusted_signal_label"] = adjusted_label
    result_df["adjusted_signal"] = adjusted_signal
    result_df["signal_confidence"] = probabilities.max(axis=1)
    result_df["buy_sell_margin"] = buy_edge
    return result_df


def save_model1_prediction(result: dict) -> None:
    MODEL1_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame([result])
    if MODEL1_PREDICTION_LOG_PATH.exists():
        result_df.to_csv(
            MODEL1_PREDICTION_LOG_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        result_df.to_csv(MODEL1_PREDICTION_LOG_PATH, index=False)

    MODEL1_LATEST_PREDICTION_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=4, default=str),
        encoding="utf-8",
    )


def render_header() -> None:
    st.title("HQTCSDL Stocks")
    st.caption(
        "Dashboard dữ liệu chứng khoán, feature engineering, data mart, mô hình dự đoán "
        "và insight từ ClickHouse."
    )


def render_overview_page() -> None:
    st.header("Tổng Quan Dashboard")
    try:
        summary = run_query(
            f"""
            SELECT
                count() AS total_rows,
                countDistinct(symbol) AS total_symbols,
                min(toDate(date)) AS min_date,
                max(toDate(date)) AS max_date
            FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            """
        )
        sector_summary = run_query(
            f"""
            SELECT countDistinct(sector) AS sector_count
            FROM {full_table_name(CLICKHOUSE_DATABASE, SYMBOL_TABLE)}
            WHERE sector IS NOT NULL AND sector != ''
            """
        )
        feature_rows = run_query(
            f"""
            SELECT count() AS feature_rows
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
            """
        )
    except Exception as exc:
        display_error("Không truy vấn được dữ liệu tổng quan từ ClickHouse.", exc)
        return

    row = summary.iloc[0] if not summary.empty else {}
    total_symbols = row.get("total_symbols", 0)
    total_rows = row.get("total_rows", 0)
    min_date = row.get("min_date", "N/A")
    max_date = row.get("max_date", "N/A")
    sector_count = (
        int(sector_summary.iloc[0]["sector_count"]) if not sector_summary.empty else 0
    )
    feature_count = int(feature_rows.iloc[0]["feature_rows"]) if not feature_rows.empty else 0

    cols = st.columns(5)
    cols[0].metric("Mã cổ phiếu", number(total_symbols))
    cols[1].metric("Bản ghi giá", number(total_rows))
    cols[2].metric("Bản ghi feature", number(feature_count))
    cols[3].metric("Số ngành", number(sector_count))
    cols[4].metric("Cập nhật mới nhất", str(max_date))

    st.subheader("Tóm tắt hệ thống")
    st.dataframe(
        pd.DataFrame(
            [
                ["Tổng số mã cổ phiếu", f"{number(total_symbols)} mã"],
                ["Khoảng thời gian dữ liệu", f"{min_date} đến {max_date}"],
                ["Tổng số bản ghi", number(total_rows)],
                ["Số ngành/lĩnh vực", number(sector_count)],
                ["Ngày dữ liệu mới nhất", str(max_date)],
            ],
            columns=["Nội dung", "Giá trị"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Thanh khoản và giá trung bình gần đây")
    try:
        market = run_query(
            f"""
            SELECT
                toDate(date) AS trading_date,
                sum(volume) AS total_volume,
                avg(close) AS avg_close
            FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            GROUP BY trading_date
            ORDER BY trading_date DESC
            LIMIT 180
            """
        ).sort_values("trading_date")
        if not market.empty:
            chart_cols = st.columns(2)
            chart_cols[0].line_chart(market, x="trading_date", y="avg_close")
            chart_cols[1].bar_chart(market, x="trading_date", y="total_volume")
    except Exception as exc:
        st.warning(f"Không vẽ được biểu đồ tổng quan: {exc}")

    st.subheader("Top cổ phiếu theo volume phiên mới nhất")
    try:
        top_volume = run_query(
            f"""
            SELECT
                symbol,
                sum(volume) AS total_volume,
                avg(close) AS avg_close
            FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            WHERE toDate(date) = (
                SELECT max(toDate(date))
                FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            )
            GROUP BY symbol
            ORDER BY total_volume DESC
            LIMIT 15
            """
        )
        st.bar_chart(top_volume, x="symbol", y="total_volume")
        st.dataframe(top_volume, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Không lấy được top volume: {exc}")


def render_stock_lookup_page() -> None:
    st.header("Tra Cứu Dữ Liệu Cổ Phiếu")
    try:
        symbols = get_symbols()
    except Exception as exc:
        display_error("Không tải được danh sách mã cổ phiếu.", exc)
        return

    if not symbols:
        st.warning("Chưa có danh sách mã cổ phiếu trong ClickHouse.")
        return

    summary = run_query(
        f"""
        SELECT min(toDate(date)) AS min_date, max(toDate(date)) AS max_date
        FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
        """
    )
    min_date = pd.to_datetime(summary.iloc[0]["min_date"]).date()
    max_date = pd.to_datetime(summary.iloc[0]["max_date"]).date()

    col1, col2, col3 = st.columns([1.3, 1.3, 0.8])
    selected_symbols = col1.multiselect(
        "Symbol",
        symbols,
        default=[s for s in ["FPT"] if s in symbols] or symbols[:1],
    )
    selected_range = col2.date_input(
        "Khoảng ngày",
        value=(max(min_date, date(max_date.year - 1, max_date.month, max_date.day)), max_date),
        min_value=min_date,
        max_value=max_date,
    )
    limit = col3.number_input("Số dòng tối đa", min_value=50, max_value=10000, value=1000, step=50)

    if not selected_symbols:
        st.info("Chọn ít nhất một mã cổ phiếu để tra cứu.")
        return

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date = end_date = selected_range

    symbol_sql = ", ".join(sql_string(symbol) for symbol in selected_symbols)
    query = f"""
        SELECT
            toDate(date) AS trading_date,
            symbol,
            open,
            high,
            low,
            close,
            volume
        FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
        WHERE symbol IN ({symbol_sql})
          AND toDate(date) BETWEEN toDate({sql_string(start_date)})
                              AND toDate({sql_string(end_date)})
        ORDER BY trading_date, symbol
        LIMIT {int(limit)}
    """

    try:
        price_df = run_query(query)
    except Exception as exc:
        display_error("Không truy vấn được dữ liệu cổ phiếu.", exc)
        return

    if price_df.empty:
        st.warning("Không có dữ liệu phù hợp với bộ lọc.")
        return

    st.dataframe(price_df, use_container_width=True, hide_index=True)
    price_df["trading_date"] = pd.to_datetime(price_df["trading_date"])

    st.subheader("Biểu đồ giá đóng cửa")
    close_chart = price_df.pivot_table(
        index="trading_date",
        columns="symbol",
        values="close",
        aggfunc="last",
    )
    st.line_chart(close_chart)

    st.subheader("Biểu đồ khối lượng giao dịch")
    volume_chart = price_df.pivot_table(
        index="trading_date",
        columns="symbol",
        values="volume",
        aggfunc="sum",
    )
    st.bar_chart(volume_chart)

    if len(selected_symbols) == 1:
        st.subheader("OHLC preview")
        st.dataframe(
            price_df[["trading_date", "open", "high", "low", "close", "volume"]]
            .tail(30)
            .sort_values("trading_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def render_feature_engineering_page() -> None:
    st.header("Feature Engineering")
    st.dataframe(
        pd.DataFrame(FEATURE_DEFINITIONS, columns=["Feature", "Ý nghĩa"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Truy vấn bảng features_all")
    try:
        symbols = get_symbols()
    except Exception as exc:
        display_error("Không tải được danh sách symbol.", exc)
        return

    col1, col2, col3 = st.columns([1, 1.4, 0.7])
    symbol = col1.selectbox(
        "Symbol",
        symbols,
        index=symbols.index("FPT") if "FPT" in symbols else 0,
    )
    date_range = col2.date_input("Khoảng ngày feature", value=())
    limit = col3.number_input("Limit", min_value=50, max_value=5000, value=500, step=50)

    where = [f"symbol = {sql_string(symbol)}"]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        where.append(
            f"toDate(trading_date) BETWEEN toDate({sql_string(date_range[0])}) "
            f"AND toDate({sql_string(date_range[1])})"
        )
    query = f"""
        SELECT *
        FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
        WHERE {" AND ".join(where)}
        ORDER BY trading_date DESC
        LIMIT {int(limit)}
    """

    try:
        feature_df = run_query(query)
    except Exception as exc:
        display_error("Không truy vấn được bảng features_all.", exc)
        return

    if feature_df.empty:
        st.info("Không có feature phù hợp.")
        return

    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    feature_df["trading_date"] = pd.to_datetime(feature_df["trading_date"])
    feature_df = feature_df.sort_values("trading_date")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Close và MA")
        ma_cols = [col for col in ["close", "ma_5", "ma_20", "ma_50"] if col in feature_df]
        st.line_chart(feature_df[["trading_date", *ma_cols]], x="trading_date", y=ma_cols)
    with chart_cols[1]:
        st.subheader("Return và volatility")
        metric_cols = [
            col
            for col in ["return_5d", "return_20d", "volatility_5d", "volatility_20d"]
            if col in feature_df
        ]
        if metric_cols:
            st.line_chart(
                feature_df[["trading_date", *metric_cols]],
                x="trading_date",
                y=metric_cols,
            )


def render_data_mart_page() -> None:
    st.header("Data Mart")
    st.caption("Danh sách bảng mart/warehouse được lấy trực tiếp từ system.tables của ClickHouse.")

    databases = [
        CLICKHOUSE_DATABASE,
        "stock_mart",
        "stock_mart_model5_risk_prediction",
    ]
    db_sql = ", ".join(sql_string(db) for db in sorted(set(databases)))

    try:
        tables = run_query(
            f"""
            SELECT
                database,
                name AS table_name,
                total_rows,
                total_bytes
            FROM system.tables
            WHERE database IN ({db_sql})
            ORDER BY database, name
            """
        )
    except Exception as exc:
        display_error("Không lấy được danh sách bảng ClickHouse.", exc)
        return

    if tables.empty:
        st.warning("Không tìm thấy bảng trong các database mart/warehouse đã cấu hình.")
        return

    tables["full_name"] = tables["database"].astype(str) + "." + tables["table_name"].astype(str)
    tables["nội dung"] = tables["full_name"].map(MART_DESCRIPTIONS).fillna("Bảng dữ liệu ClickHouse.")
    tables["total_rows"] = tables["total_rows"].fillna(0).astype(int)

    st.dataframe(
        tables[["full_name", "nội dung", "total_rows", "total_bytes"]],
        use_container_width=True,
        hide_index=True,
    )

    selected_table = st.selectbox("Chọn bảng để preview", tables["full_name"].tolist())
    selected_db, selected_name = selected_table.split(".", 1)
    columns = get_table_columns(selected_db, selected_name)

    if not columns:
        st.info("Không đọc được schema bảng đã chọn.")
        return

    st.write("Schema")
    st.dataframe(pd.DataFrame({"column": columns}), use_container_width=True, hide_index=True)

    symbol_filter = None
    date_filter = None
    date_col = first_existing(
        columns,
        ["trading_date", "date", "prediction_date", "target_date", "created_at", "updated_at"],
    )

    filter_cols = st.columns(3)
    if "symbol" in columns:
        symbol_filter = filter_cols[0].text_input("Lọc symbol", value="")
    if date_col:
        date_filter = filter_cols[1].date_input(f"Lọc ngày theo {date_col}", value=())
    preview_limit = filter_cols[2].number_input(
        "Preview rows",
        min_value=20,
        max_value=1000,
        value=100,
        step=20,
    )

    where = []
    if symbol_filter:
        where.append(f"upper(trim(symbol)) = upper(trim({sql_string(symbol_filter)}))")
    if isinstance(date_filter, tuple) and len(date_filter) == 2 and date_col:
        where.append(
            f"toDate({quote_identifier(date_col)}) BETWEEN "
            f"toDate({sql_string(date_filter[0])}) AND toDate({sql_string(date_filter[1])})"
        )
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    order_sql = f"ORDER BY {quote_identifier(date_col)} DESC" if date_col else ""

    try:
        preview = run_query(
            f"""
            SELECT *
            FROM {full_table_name(selected_db, selected_name)}
            {where_sql}
            {order_sql}
            LIMIT {int(preview_limit)}
            """
        )
        st.dataframe(preview, use_container_width=True, hide_index=True)
    except Exception as exc:
        display_error("Không preview được bảng đã chọn.", exc)


def render_model1_prediction() -> None:
    st.subheader("Model 1 - Dự đoán return 5 phiên và suy ra giá")
    st.caption("Logic dựa trên models/model1/test.py.")

    try:
        symbols = get_symbols()
    except Exception as exc:
        display_error("Không tải được symbol.", exc)
        return

    col1, col2 = st.columns(2)
    symbol = col1.selectbox(
        "Symbol",
        symbols,
        index=symbols.index("FPT") if "FPT" in symbols else 0,
        key="model1_symbol",
    )
    trading_date = col2.date_input("Ngày giao dịch", key="model1_date")

    if not st.button("Dự đoán Model 1", type="primary"):
        return

    try:
        artifact = load_model1_artifact()
        row_df = fetch_feature_rows(symbol=symbol, trading_date=trading_date, limit=1)
    except Exception as exc:
        display_error("Không tải được model hoặc dữ liệu Model 1.", exc)
        return

    if row_df.empty:
        st.warning("Không tìm thấy feature cho symbol/ngày đã chọn.")
        return

    model = artifact["model"]
    features = artifact["features"]
    horizon = artifact.get("horizon", 5)
    target_type = artifact.get("target_type", "future_return")
    return_calibrator = artifact.get("return_calibrator")

    missing_features = [col for col in features if col not in row_df.columns]
    if missing_features:
        st.error("Thiếu feature: " + ", ".join(missing_features))
        return

    for col in features + ["close"]:
        row_df[col] = pd.to_numeric(row_df[col], errors="coerce")
    if row_df[features + ["close"]].isna().any(axis=None):
        st.error("Dòng dữ liệu được chọn có feature rỗng hoặc không phải số.")
        return

    raw_prediction = np.asarray(model.predict(row_df[features]), dtype=float)
    close = float(row_df["close"].iloc[0])

    if target_type == "future_return":
        if return_calibrator is not None:
            predicted_return = float(return_calibrator.predict(raw_prediction)[0])
        else:
            predicted_return = float(raw_prediction[0])
        predicted_close = close * (1 + predicted_return)
    else:
        predicted_close = float(raw_prediction[0])
        predicted_return = predicted_close / close - 1

    direction = "UP" if predicted_return >= 0 else "DOWN"
    result = {
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "trading_date": str(trading_date),
        "close": close,
        "predicted_return": predicted_return,
        "predicted_return_pct": predicted_return * 100,
        "predicted_close": predicted_close,
        "direction": direction,
        "horizon": horizon,
        "target_type": target_type,
    }
    save_model1_prediction(result)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Close hiện tại", f"{close:,.2f}")
    metric_cols[1].metric("Return dự đoán", pct(predicted_return))
    metric_cols[2].metric("Close dự đoán", f"{predicted_close:,.2f}")
    metric_cols[3].metric("Xu hướng", direction)

    st.dataframe(pd.DataFrame([result]), use_container_width=True, hide_index=True)
    st.write("Feature đưa vào model")
    st.dataframe(row_df[["symbol", "trading_date", *features]], use_container_width=True, hide_index=True)


def render_model2_prediction() -> None:
    st.subheader("Model 2 - Dự đoán future_return_5d")
    st.caption("Logic dựa trên models/model2/streamlit_model2_demo.py, nhưng dùng .env thay vì hardcode credential.")

    try:
        symbols = get_symbols()
    except Exception as exc:
        display_error("Không tải được symbol.", exc)
        return

    col1, col2 = st.columns(2)
    symbol = col1.selectbox(
        "Symbol",
        symbols,
        index=symbols.index("FPT") if "FPT" in symbols else 0,
        key="model2_symbol",
    )
    trading_date = col2.date_input("Ngày giao dịch", key="model2_date")

    if not st.button("Dự đoán Model 2", type="primary"):
        return

    try:
        model = load_model2_artifact()
        row_df = fetch_feature_rows(symbol=symbol, trading_date=trading_date, limit=1)
    except Exception as exc:
        display_error("Không tải được model hoặc dữ liệu Model 2.", exc)
        return

    if row_df.empty:
        st.warning("Không tìm thấy feature cho symbol/ngày đã chọn.")
        return

    missing_features = [col for col in MODEL2_FEATURE_COLUMNS if col not in row_df.columns]
    if missing_features:
        st.error("Thiếu feature: " + ", ".join(missing_features))
        return

    X = row_df[MODEL2_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    prediction = float(model.predict(X)[0])
    signal = add_prediction_signal(prediction)
    close = float(pd.to_numeric(row_df["close"], errors="coerce").iloc[0])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Symbol", symbol)
    metric_cols[1].metric("Close hiện tại", f"{close:,.2f}")
    metric_cols[2].metric("Future return 5D", pct(prediction))
    metric_cols[3].metric("Nhận định", signal)

    result = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trading_date": str(trading_date),
                "close": close,
                "predicted_future_return_5d": prediction,
                "predicted_future_return_pct": prediction * 100,
                "signal": signal,
            }
        ]
    )
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.write("Feature đưa vào model")
    st.dataframe(X, use_container_width=True, hide_index=True)


def render_model3_prediction() -> None:
    st.subheader("Model 3 - Tín hiệu BUY / HOLD / SELL")
    st.caption("Logic dựa trên models/model3/test.py.")

    try:
        symbols = ["Tất cả"] + get_symbols()
    except Exception as exc:
        display_error("Không tải được symbol.", exc)
        return

    col1, col2, col3 = st.columns([1, 1, 1])
    selected_symbol = col1.selectbox("Symbol", symbols, key="model3_symbol")
    trading_date = col2.date_input("Ngày giao dịch", key="model3_date")
    min_prob = col3.slider("Ngưỡng confidence", 0.0, 1.0, 0.60, 0.05)

    if not st.button("Dự đoán Model 3", type="primary"):
        return

    try:
        model, features, signal_labels = load_model3_artifact()
        symbol_filter = None if selected_symbol == "Tất cả" else selected_symbol
        df = fetch_feature_rows(symbol=symbol_filter, trading_date=trading_date, limit=1000)
    except Exception as exc:
        display_error("Không tải được model hoặc dữ liệu Model 3.", exc)
        return

    if df.empty:
        st.warning("Không tìm thấy dữ liệu feature cho ngày đã chọn.")
        return

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        st.error("Thiếu feature: " + ", ".join(missing_features))
        return

    df = df.dropna(subset=features).copy()
    if df.empty:
        st.error("Sau khi loại dòng thiếu feature, không còn dữ liệu hợp lệ.")
        return

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[features]
    pred_label = model.predict(X).astype(int)
    pred_proba = model.predict_proba(X)

    result = df.copy()
    result["predicted_signal_label"] = pred_label
    result["predicted_signal"] = result["predicted_signal_label"].map(signal_labels)
    for label_id, label_name in signal_labels.items():
        result[f"{label_name.lower()}_probability"] = pred_proba[:, label_id]
    result["predicted_signal_score"] = result["buy_probability"] - result["sell_probability"]
    result = apply_confidence_adjusted_signals(
        result,
        min_action_probability=min_prob,
        min_action_margin=0.0,
    )

    display_cols = [
        "symbol",
        "trading_date",
        "close",
        "predicted_signal",
        "adjusted_signal",
        "buy_probability",
        "hold_probability",
        "sell_probability",
        "signal_confidence",
        "buy_sell_margin",
    ]
    available_cols = [col for col in display_cols if col in result.columns]

    counts = result["adjusted_signal"].value_counts().reindex(["BUY", "HOLD", "SELL"]).fillna(0)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Số mã hợp lệ", number(len(result)))
    metric_cols[1].metric("BUY", number(counts.get("BUY", 0)))
    metric_cols[2].metric("HOLD", number(counts.get("HOLD", 0)))
    metric_cols[3].metric("SELL", number(counts.get("SELL", 0)))

    st.dataframe(
        result[available_cols].sort_values("buy_probability", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def render_prediction_models_page() -> None:
    st.header("Mô Hình Dự Đoán")
    model_page = st.radio(
        "Chọn mô hình",
        [
            "Model 1 - Dự đoán return/giá",
            "Model 2 - Dự đoán lợi suất 5 ngày",
            "Model 3 - BUY/HOLD/SELL",
            "Model 4 - Tạm thời bỏ trống",
            "Model 5 - Tạm thời bỏ trống",
        ],
        horizontal=True,
    )

    if model_page.startswith("Model 1"):
        render_model1_prediction()
    elif model_page.startswith("Model 2"):
        render_model2_prediction()
    elif model_page.startswith("Model 3"):
        render_model3_prediction()
    elif model_page.startswith("Model 4"):
        st.info("Model 4 hiện chưa có file Streamlit dự đoán riêng, nên trang này tạm thời để trống.")
    else:
        st.info("Model 5 hiện chưa có file Streamlit dự đoán riêng, nên trang này tạm thời để trống.")


def render_insight_page() -> None:
    st.header("Insight")

    try:
        latest_date_df = run_query(
            f"""
            SELECT max(toDate(trading_date)) AS latest_date
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
            """
        )
        latest_date = latest_date_df.iloc[0]["latest_date"]
        latest_filter = f"toDate(f.trading_date) = toDate({sql_string(latest_date)})"
    except Exception as exc:
        display_error("Không lấy được ngày feature mới nhất.", exc)
        return

    st.caption(f"Dữ liệu feature mới nhất: {latest_date}")

    try:
        top_gain = run_query(
            f"""
            SELECT symbol, close, return_5d
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            WHERE {latest_filter} AND return_5d IS NOT NULL
            ORDER BY return_5d DESC
            LIMIT 10
            """
        )
        top_loss = run_query(
            f"""
            SELECT symbol, close, return_5d
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            WHERE {latest_filter} AND return_5d IS NOT NULL
            ORDER BY return_5d ASC
            LIMIT 10
            """
        )
        top_volume = run_query(
            f"""
            SELECT symbol, close, volume
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            WHERE {latest_filter}
            ORDER BY volume DESC
            LIMIT 10
            """
        )
        sector_perf = run_query(
            f"""
            SELECT
                coalesce(s.sector, 'UNKNOWN') AS sector,
                avg(f.return_5d) AS avg_return_5d,
                count() AS stock_count
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            LEFT JOIN {full_table_name(CLICKHOUSE_DATABASE, SYMBOL_TABLE)} AS s
                ON f.symbol = s.symbol
            WHERE {latest_filter}
              AND f.return_5d IS NOT NULL
            GROUP BY sector
            ORDER BY avg_return_5d DESC
            LIMIT 15
            """
        )
        volatile = run_query(
            f"""
            SELECT symbol, close, volatility_20d, abs(return_5d) AS abs_return_5d
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            WHERE {latest_filter}
            ORDER BY volatility_20d DESC, abs_return_5d DESC
            LIMIT 10
            """
        )
    except Exception as exc:
        display_error("Không truy vấn được insight từ ClickHouse.", exc)
        return

    insight_rows = [
        ["Top cổ phiếu tăng mạnh nhất", ", ".join(top_gain["symbol"].astype(str).head(5))],
        ["Top cổ phiếu giảm mạnh nhất", ", ".join(top_loss["symbol"].astype(str).head(5))],
        ["Ngành có hiệu suất tốt nhất", ", ".join(sector_perf["sector"].astype(str).head(3))],
        ["Cổ phiếu biến động mạnh nhất", ", ".join(volatile["symbol"].astype(str).head(5))],
        ["Cổ phiếu thanh khoản cao nhất", ", ".join(top_volume["symbol"].astype(str).head(5))],
    ]
    st.dataframe(pd.DataFrame(insight_rows, columns=["Insight", "Kết quả"]), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top tăng 5 phiên")
        st.bar_chart(top_gain, x="symbol", y="return_5d")
        st.dataframe(top_gain, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Top giảm 5 phiên")
        st.bar_chart(top_loss, x="symbol", y="return_5d")
        st.dataframe(top_loss, use_container_width=True, hide_index=True)

    st.subheader("Hiệu suất theo ngành")
    st.bar_chart(sector_perf, x="sector", y="avg_return_5d")
    st.dataframe(sector_perf, use_container_width=True, hide_index=True)

    st.subheader("Thanh khoản cao nhất")
    st.bar_chart(top_volume, x="symbol", y="volume")


def render_data_quality_page() -> None:
    st.header("Chất Lượng Dữ Liệu")

    try:
        quality = run_query(
            f"""
            SELECT
                count() AS total_rows,
                countDistinct(symbol) AS total_symbols,
                min(toDate(date)) AS min_date,
                max(toDate(date)) AS max_date,
                countIf(open <= 0 OR high <= 0 OR low <= 0 OR close <= 0) AS non_positive_prices,
                countIf(volume < 0) AS negative_volume,
                countIf(
                    high < low
                    OR high < open
                    OR high < close
                    OR low > open
                    OR low > close
                ) AS invalid_ohlc,
                count() - uniqExact(tuple(symbol, date)) AS duplicate_symbol_date
            FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            """
        )
    except Exception as exc:
        display_error("Không truy vấn được quality check từ ClickHouse.", exc)
        return

    row = quality.iloc[0] if not quality.empty else {}
    cols = st.columns(4)
    cols[0].metric("Tổng dòng", number(row.get("total_rows")))
    cols[1].metric("Số mã", number(row.get("total_symbols")))
    cols[2].metric("Ngày đầu", str(row.get("min_date")))
    cols[3].metric("Ngày cuối", str(row.get("max_date")))

    cols = st.columns(4)
    cols[0].metric("Giá <= 0", number(row.get("non_positive_prices")))
    cols[1].metric("Volume âm", number(row.get("negative_volume")))
    cols[2].metric("OHLC lỗi", number(row.get("invalid_ohlc")))
    cols[3].metric("Trùng symbol-date", number(row.get("duplicate_symbol_date")))

    st.dataframe(quality, use_container_width=True, hide_index=True)

    clean_summary = PROJECT_ROOT / "data" / "clean_log" / "clean_summary.txt"
    survey_summary = PROJECT_ROOT / "data" / "khaosatdata" / "khaosat_summary.txt"

    st.subheader("Log làm sạch dữ liệu")
    if clean_summary.exists():
        st.text(clean_summary.read_text(encoding="utf-8"))
    else:
        st.info("Chưa tìm thấy data/clean_log/clean_summary.txt")

    st.subheader("Log khảo sát dữ liệu dirty")
    if survey_summary.exists():
        st.text(survey_summary.read_text(encoding="utf-8"))
    else:
        st.info("Chưa tìm thấy data/khaosatdata/khaosat_summary.txt")


def main() -> None:
    render_header()

    page = st.sidebar.radio(
        "Chức năng",
        [
            "1. Tổng quan Dashboard",
            "2. Tra cứu dữ liệu cổ phiếu",
            "3. Feature Engineering",
            "4. Data Mart",
            "5. Mô hình dự đoán",
            "6. Insight",
            "7. Chất lượng dữ liệu",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption(f"ClickHouse database: {CLICKHOUSE_DATABASE}")
    st.sidebar.caption(f"Feature table: {FEATURES_DATABASE}.{FEATURES_TABLE}")

    if page.startswith("1."):
        render_overview_page()
    elif page.startswith("2."):
        render_stock_lookup_page()
    elif page.startswith("3."):
        render_feature_engineering_page()
    elif page.startswith("4."):
        render_data_mart_page()
    elif page.startswith("5."):
        render_prediction_models_page()
    elif page.startswith("6."):
        render_insight_page()
    else:
        render_data_quality_page()


if __name__ == "__main__":
    main()
