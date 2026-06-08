import streamlit as st
import pandas as pd
import joblib
import clickhouse_connect
from pathlib import Path

# =========================
# CONFIG
# =========================
CLICKHOUSE_HOST = "cvzq3t560s.ap-southeast-1.aws.clickhouse.cloud"
CLICKHOUSE_PORT = 8443
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "K5clN_57i9pu6"
CLICKHOUSE_DATABASE = "stock"
CLICKHOUSE_SECURE = True

TABLE_NAME = "features_all"

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "future_return_lgbm.pkl"

DATE_COL = "trading_date"
SYMBOL_COL = "symbol"
TARGET_COL = "future_return_5d"

DROP_COLS = [
    DATE_COL,
    SYMBOL_COL,
    TARGET_COL,
    "companyname",
    "sector"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=CLICKHOUSE_SECURE
    )


def get_feature_by_symbol_date(symbol, trading_date):
    client = get_client()

    query = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE upper(trim(symbol)) = upper(trim('{symbol}'))
          AND toDate(trading_date) = toDate('{trading_date}')
        LIMIT 1
    """

    return client.query_df(query)

def get_latest_dates(symbol):
    client = get_client()

    query = f"""
        SELECT DISTINCT trading_date
        FROM {TABLE_NAME}
        WHERE upper(trim(symbol)) = upper(trim('{symbol}'))
        ORDER BY trading_date DESC
        LIMIT 10
    """

    return client.query_df(query)

def prepare_input(df):
    X = df.drop(columns=[col for col in DROP_COLS if col in df.columns], errors="ignore")

    # Chỉ giữ cột số
    X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])

    X = X.fillna(0)
    return X


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Model2 - Future Return 5D",
    layout="wide"
)

st.title("Demo Model2 - Dự đoán lợi suất cổ phiếu 5 ngày tới")

st.markdown("""
Model2 nhận đầu vào là **mã cổ phiếu** và **ngày giao dịch**, 
sau đó lấy feature tương ứng từ ClickHouse và dự đoán `future_return_5d`.
""")

col1, col2 = st.columns(2)

with col1:
    symbol = st.text_input("Nhập mã cổ phiếu", value="ACB")

with col2:
    trading_date = st.date_input("Chọn ngày giao dịch")

if st.button("Dự đoán"):
    model = load_model()

    date_str = trading_date.strftime("%Y-%m-%d")

    df = get_feature_by_symbol_date(symbol.upper(), date_str)

    if df.empty:
      st.error("Không tìm thấy dữ liệu feature cho symbol và ngày đã chọn.")

      latest_dates = get_latest_dates(symbol.upper())

      if not latest_dates.empty:
          st.warning("Các ngày gần nhất có dữ liệu của mã này:")
          st.dataframe(latest_dates)
    else:
        st.subheader("Dữ liệu feature lấy từ ClickHouse")
        st.dataframe(df)

        X = prepare_input(df)

        prediction = model.predict(X)[0]

        st.subheader("Kết quả dự đoán")

        st.metric(
            label="Predicted Future Return 5D",
            value=f"{prediction * 100:.2f}%"
        )

        if prediction > 0.03:
            signal = "Kỳ vọng tăng mạnh"
        elif prediction > 0:
            signal = "Kỳ vọng tăng nhẹ"
        elif prediction > -0.03:
            signal = "Trung tính / giảm nhẹ"
        else:
            signal = "Rủi ro giảm mạnh"

        st.info(f"Nhận định: {signal}")

        st.subheader("Feature đưa vào model")
        st.dataframe(X)