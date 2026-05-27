import os
import clickhouse_connect

def get_clickhouse_client():
    # Lấy cấu hình từ biến môi trường (do GitHub Actions hoặc file .env cấp)
    CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
    CH_PORT = os.getenv("CLICKHOUSE_PORT")
    CH_USER = os.getenv("CLICKHOUSE_USER", "default")
    CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")

    # Xử lý trường hợp PORT bị trống hoặc chuỗi rỗng thành số nguyên chuẩn
    if not CH_PORT or CH_PORT.strip() == "":
        CH_PORT = 8443 if "clickhouse.cloud" in CH_HOST else 8123
    else:
        CH_PORT = int(CH_PORT)

    # Bắt buộc phải bật kết nối bảo mật (secure=True) khi chạy trên ClickHouse Cloud (port 8443)
    is_secure = True if CH_PORT == 8443 or "clickhouse.cloud" in CH_HOST else False

    # Khởi tạo và trả về client kết nối
    client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        secure=is_secure
    )
    print(f"Đã kết nối thành công tới ClickHouse tại: {CH_HOST}:{CH_PORT} (Secure: {is_secure})")
    return client
