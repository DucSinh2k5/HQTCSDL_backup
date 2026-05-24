# Hệ quản trị cơ sở dữ liệu

## Pipeline Stock

### Cấu trúc chính

- `ingestion/`: crawl danh sách mã, giá cổ phiếu và thông tin công ty.
- `etl/xu_li_du_lieu/`: khảo sát chất lượng dữ liệu và làm sạch CSV.
- `connect_clickhouse/`: upload/export dữ liệu nền sang ClickHouse.
- `models/model2/`: mô hình hồi quy `future_return_5d`.
- `models/model5/`: pipeline cảnh báo rủi ro giảm giá.
- `models/model5/output_model5/`: output CSV/JSON của model 5.

### Entrypoint

- Full pipeline: `python main.py`
- Model 5 pipeline: `python models/model5/run_pipeline.py`
- Upload output model 5: `python models/model5/upload_outputs_to_clickhouse.py`

`models/model5/run_pipeline.py` lấy feature mặc định từ ClickHouse
`stock.features_all`, sau đó tự tạo `future_return_5d` và `risk_drop_label`
cho bài toán cảnh báo rủi ro.

`connect_clickhouse/features_all.py` tạo bảng feature chung `stock.features_all`
từ `stock.stock_prices`, nên trong `main.py` bước này chạy trước model 5.
