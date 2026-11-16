# Nhom17_DeTai16_RuiRoGiaoHangChamTre

## 1. Giới thiệu dự án

Dự án **Dự đoán đơn hàng có nguy cơ giao trễ trong chuỗi cung ứng.** được xây dựng nhằm nghiên cứu và ứng dụng các thuật toán học máy trên dữ liệu dạng bảng (Tabular Data).

Đề tài tập trung vào việc phân tích dữ liệu chuỗi cung ứng và xây dựng mô hình Machine Learning để dự đoán tình trạng giao hàng dựa trên các đặc trưng của đơn hàng.

Dự án sử dụng hai thuật toán học máy chính:

- **K-Nearest Neighbors (KNN)**
- **Decision Tree (Cây quyết định)**

Ngoài việc huấn luyện và đánh giá mô hình, dự án còn xây dựng giao diện trực quan bằng Streamlit, cho phép người dùng theo dõi kết quả và thử nghiệm dự đoán.

## 2. Mục tiêu dự án

- Tìm hiểu và phân tích dữ liệu dạng bảng.
- Thực hiện khám phá dữ liệu (Exploratory Data Analysis – EDA).
- Xử lý dữ liệu thiếu và chuẩn bị dữ liệu trước khi huấn luyện.
- Xây dựng và so sánh hai quy trình tiền xử lý dữ liệu.
- Huấn luyện các mô hình KNN và Decision Tree.
- Thử nghiệm các bộ tham số khác nhau để tìm mô hình phù hợp.
- Đánh giá mô hình bằng các chỉ số Accuracy, Precision, Recall và F1-score.
- Phân tích các trường hợp dự đoán sai.
- Xây dựng giao diện dự đoán bằng Streamlit.

## 3. Công nghệ sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| Pandas | Xử lý và phân tích dữ liệu |
| NumPy | Tính toán dữ liệu |
| Matplotlib | Trực quan hóa dữ liệu |
| Seaborn | Trực quan hóa dữ liệu thống kê |
| Scikit-learn | Tiền xử lý, huấn luyện và đánh giá mô hình |
| Joblib | Lưu và tải mô hình |
| Streamlit | Xây dựng giao diện ứng dụng |

## 4. Cấu trúc dự án

```text
Tabular_Machine_Learning/
│
├── Data/
│   └── DataCoSupplyChainDataset.csv
│
├── Models/
│   └── best_model.pkl
│
├── Notebooks/
│
├── Results/
│   ├── model_comparison.csv
│   ├── danh_sach_dac_trung.csv
│   ├── error_analysis_summary.txt
│   └── *.png
│
├── Src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── train_model.py
│
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

### Mô tả các thư mục và tệp

- **Data/**: Chứa dữ liệu đầu vào của dự án.
- **Models/**: Lưu mô hình Machine Learning đã huấn luyện.
- **Notebooks/**: Lưu các notebook phục vụ quá trình thử nghiệm và phân tích.
- **Results/**: Lưu kết quả đánh giá, danh sách đặc trưng và biểu đồ.
- **Src/**: Chứa các chương trình xử lý dữ liệu và huấn luyện mô hình.
- **main.py**: Tệp chạy các bước chính của dự án.
- **app.py**: Tệp xây dựng giao diện ứng dụng bằng Streamlit.
- **requirements.txt**: Danh sách các thư viện cần thiết.

## 5. Các phương pháp thực hiện

### 5.1. Khám phá dữ liệu (EDA)

Thực hiện phân tích dữ liệu ban đầu nhằm tìm hiểu:

- Kích thước và cấu trúc dữ liệu.
- Kiểu dữ liệu của các thuộc tính.
- Giá trị thiếu.
- Phân bố dữ liệu.
- Mối quan hệ giữa các đặc trưng.

Kết quả phân tích được trực quan hóa và lưu trong thư mục `Results/`.

### 5.2. Tiền xử lý dữ liệu

Dự án xây dựng hai Pipeline tiền xử lý:

**Pipeline A:**

- Dữ liệu số: Thay thế giá trị thiếu bằng Median và chuẩn hóa bằng StandardScaler.
- Dữ liệu phân loại: Thay thế giá trị thiếu bằng giá trị xuất hiện nhiều nhất và mã hóa bằng OneHotEncoder.

**Pipeline B:**

- Dữ liệu số: Thay thế giá trị thiếu bằng Median, bổ sung cờ nhận biết dữ liệu thiếu và chuẩn hóa bằng RobustScaler.
- Dữ liệu phân loại: Thay thế giá trị thiếu bằng giá trị mặc định và mã hóa bằng OneHotEncoder.

Dữ liệu được chia thành tập huấn luyện và tập kiểm tra theo tỷ lệ 80:20. Các bước học tham số tiền xử lý chỉ được thực hiện trên tập huấn luyện nhằm hạn chế rò rỉ dữ liệu (Data Leakage).

### 5.3. Huấn luyện mô hình

Hai thuật toán được sử dụng:

- **K-Nearest Neighbors (KNN):** Dự đoán dựa trên các điểm dữ liệu gần nhất.
- **Decision Tree:** Sử dụng cấu trúc cây quyết định để phân loại dữ liệu.

Các bộ tham số khác nhau được thử nghiệm để so sánh hiệu quả của từng mô hình và Pipeline.

### 5.4. Đánh giá mô hình

Các chỉ số đánh giá được sử dụng:

- **Accuracy:** Tỷ lệ dự đoán chính xác trên tổng số mẫu.
- **Precision:** Tỷ lệ dự đoán dương tính đúng trên tổng số dự đoán dương tính.
- **Recall:** Tỷ lệ mẫu dương tính thực tế được mô hình nhận diện đúng.
- **F1-score:** Trung bình điều hòa giữa Precision và Recall.
- **Confusion Matrix:** Ma trận thể hiện số lượng dự đoán đúng và sai.

Ngoài ra, dự án thực hiện phân tích lỗi nhằm tìm hiểu các trường hợp False Positive và False Negative.

## 6. Kết quả thực nghiệm

Kết quả thực nghiệm cho thấy hai thuật toán có hiệu quả khác nhau tùy theo Pipeline và bộ tham số.

Một số kết quả nổi bật:

| Mô hình | Pipeline | Tham số | Accuracy | F1-score |
|---|---|---|---:|---:|
| KNN | A | K = 9 | 65.90% | 68.20% |
| Decision Tree | A | max_depth = 10 | 68.24% | 66.54% |

Trong các cấu hình đã thử nghiệm, KNN với Pipeline A và K = 9 đạt F1-score cao nhất.

Decision Tree với độ sâu tối đa bằng 10 đạt Accuracy cao nhất trong các cấu hình được thử nghiệm.

**Lưu ý:** Kết quả trên được tổng hợp từ lần thực nghiệm hiện tại và có thể thay đổi khi dữ liệu hoặc cấu hình huấn luyện thay đổi.

## 7. Giao diện ứng dụng

Dự án sử dụng Streamlit để xây dựng giao diện trực quan với các chức năng chính:

- **Tổng quan:** Giới thiệu dự án và hiển thị các thông tin tổng quát.
- **Phân tích dữ liệu:** Hiển thị các biểu đồ phân tích dữ liệu.
- **Đánh giá mô hình:** Theo dõi và so sánh kết quả các mô hình.
- **Dự đoán giao hàng:** Nhập thông tin đơn hàng để thực hiện dự đoán.

## 8. Hướng dẫn cài đặt và sử dụng

### Bước 1: Tải dự án

Clone dự án từ GitHub:

```bash
git clone <GITHUB_REPOSITORY_URL>
```

Di chuyển vào thư mục dự án:

```bash
cd Tabular_Machine_Learning
```

### Bước 2: Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
venv\Scripts\activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 4: Chạy chương trình

Để thực hiện các bước xử lý và huấn luyện mô hình:

```bash
python main.py
```

Hoặc chạy riêng chương trình huấn luyện:

```bash
python -m Src.train_model
```

### Bước 5: Khởi chạy giao diện Streamlit

```bash
streamlit run app.py
```

Sau khi chạy thành công, Streamlit sẽ cung cấp địa chỉ truy cập ứng dụng trên trình duyệt.

## 9. Yêu cầu hệ thống

- Python 3.x
- pip
- Các thư viện được liệt kê trong `requirements.txt`
- Dữ liệu đầu vào được đặt đúng vị trí trong thư mục `Data/`

## 10. Tác giả

**Dự án học tập: Tabular Machine Learning**

Được thực hiện với mục đích nghiên cứu, thực hành và tìm hiểu quy trình xây dựng mô hình học máy trên dữ liệu dạng bảng.

---

*README.md – Tabular Machine Learning Project*
