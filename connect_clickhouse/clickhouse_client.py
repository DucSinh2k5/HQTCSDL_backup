import os
import clickhouse_connect

# Lấy cấu hình từ biến môi trường (do GitHub Actions hoặc file .env cấp)
# Nếu không có biến môi trường, nó mới dùng giá trị dự phòng (máy local)
CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = os.getenv("CLICKHOUSE_PORT")
CH_USER = os.getenv("CLICKHOUSE_USER", "default")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")

# Xử lý trường hợp PORT bị trống hoặc chuỗi rỗng thành số nguyên chuẩn
if not CH_PORT or CH_PORT.strip() == "":
    # Nếu kết nối tới Cloud (host chứa chữ clickhouse.cloud) thì mặc định port 8443, ngược lại local là 8123
    CH_PORT = 8443 if "clickhouse.cloud" in CH_HOST else 8123
else:
    CH_PORT = int(CH_PORT)

# Bắt buộc phải bật kết nối bảo mật (secure=True) khi chạy trên ClickHouse Cloud (port 8443)
is_secure = True if CH_PORT == 8443 or "clickhouse.cloud" in CH_HOST else False

# Khởi tạo client kết nối chuẩn chỉnh
client = clickhouse_connect.get_client(
    host=CH_HOST,
    port=CH_PORT,
    username=CH_USER,
    password=CH_PASSWORD,
    secure=is_secure
)

print(f"Đã kết nối thành công tới ClickHouse tại: {CH_HOST}:{CH_PORT} (Secure: {is_secure})")
