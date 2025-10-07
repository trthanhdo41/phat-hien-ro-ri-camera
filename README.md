# Phát hiện rủi ro rò rỉ dữ liệu camera

Dự án phát hiện rủi ro rò rỉ dữ liệu camera sử dụng Machine Learning và Deep Learning.

## Mô tả

Dự án này sử dụng các phương pháp khoa học dữ liệu để:
- Tiền xử lý dữ liệu
- Xây dựng mô hình học máy
- Đánh giá mô hình

## Cấu trúc dự án

- `run_pipeline.py`: Script chính chạy toàn bộ pipeline
- `HDSD.txt`: Hướng dẫn sử dụng
- `outputs/`: Thư mục chứa kết quả (models, metrics, logs)
- `venv/`: Virtual environment Python

## Yêu cầu hệ thống

- Python 3.7+
- Các thư viện: pandas, numpy, scikit-learn, joblib

## Cách sử dụng

1. Cài đặt virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

2. Cài đặt dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

3. Chạy pipeline:
```bash
python run_pipeline.py
```

## Lưu ý

- File dữ liệu `iot23_combined_new.csv` (862MB) cần được đặt trong thư mục gốc
- Pipeline sẽ tạo thư mục `outputs/` để lưu kết quả
- Xem `HDSD.txt` để biết chi tiết hướng dẫn sử dụng

## Mô hình được sử dụng

1. **RandomForest**: Mô hình ensemble learning
2. **SVM**: Support Vector Machine  
3. **SGDClassifier**: Incremental Learning cho dữ liệu lớn

## Kết quả

Pipeline sẽ tạo ra:
- Các file model (.pkl)
- Báo cáo đánh giá (.txt)
- Ma trận confusion (.csv)
- Log quá trình chạy (.txt)
