
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder


# =====================================================
# 1. CAU HINH DUONG DAN
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Models" / "best_model.pkl"
RESULTS_DIR = BASE_DIR / "Results"

COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
FEATURE_PATH = RESULTS_DIR / "danh_sach_dac_trung.csv"
ERROR_PATH = RESULTS_DIR / "error_analysis_summary.txt"


# =====================================================
# 2. DANH SACH DAC TRUNG
# =====================================================

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Product Price",
    "Latitude",
    "Longitude"
]

CATEGORICAL_FEATURES = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Region",
    "Shipping Mode"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =====================================================
# 3. CAU HINH TRANG
# =====================================================

st.set_page_config(
    page_title="Delivery AI | Dự đoán giao hàng trễ",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 4. CSS GIAO DIEN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}

.stApp {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background-color: #10243a;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 9px 12px;
    border-radius: 8px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #1c3a58;
}

.main-title {
    color: #10243a;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    color: #718096;
    font-size: 14px;
    margin-bottom: 25px;
}

.section-title {
    color: #10243a;
    font-size: 21px;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 15px;
}

.info-card {
    background: #ffffff;
    padding: 20px;
    border-radius: 13px;
    border: 1px solid #e6ebf2;
    box-shadow: 0 3px 12px rgba(16, 36, 58, 0.04);
}

.info-label {
    color: #718096;
    font-size: 13px;
    font-weight: 600;
}

.info-value {
    color: #10243a;
    font-size: 27px;
    font-weight: 800;
    margin-top: 7px;
}

.info-description {
    color: #718096;
    font-size: 12px;
    margin-top: 5px;
}

.hero {
    background: linear-gradient(120deg, #10243a, #1764a0);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 30px;
    font-weight: 800;
}

.hero p {
    color: #d9e8f5;
    font-size: 14px;
}

div.stButton > button {
    background-color: #1769aa;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    padding: 10px 20px;
}

div.stButton > button:hover {
    background-color: #105386;
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e6ebf2;
    border-radius: 12px;
    padding: 17px;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #e6ebf2;
    border-radius: 13px;
    padding: 20px;
}

hr {
    border-color: #e6ebf2;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 5. DOC MO HINH
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


# =====================================================
# 6. DOC DU LIEU KET QUA
# =====================================================

@st.cache_data
def load_comparison():

    if not COMPARISON_PATH.exists():
        return None

    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def load_feature_list():

    if not FEATURE_PATH.exists():
        return None

    return pd.read_csv(FEATURE_PATH)


@st.cache_data
def load_error_summary():

    if not ERROR_PATH.exists():
        return None

    try:
        return pd.read_csv(ERROR_PATH)
    except Exception:
        return None


# =====================================================
# 7. TIM ONE HOT ENCODER
# =====================================================

def find_encoder(obj):

    if isinstance(obj, OneHotEncoder):
        return obj

    if hasattr(obj, "steps"):

        for _, step in obj.steps:

            result = find_encoder(step)

            if result is not None:
                return result

    if hasattr(obj, "transformers_"):

        for _, transformer, _ in obj.transformers_:

            if transformer == "drop" or transformer == "passthrough":
                continue

            result = find_encoder(transformer)

            if result is not None:
                return result

    return None


# =====================================================
# 8. DOC TOAN BO DU LIEU
# =====================================================

try:
    bundle = load_model()
except Exception as error:
    bundle = None
    model_error = str(error)
else:
    model_error = None

try:
    comparison = load_comparison()
except Exception:
    comparison = None

try:
    feature_list = load_feature_list()
except Exception:
    feature_list = None

try:
    error_summary = load_error_summary()
except Exception:
    error_summary = None


# =====================================================
# 9. HAM HO TRO
# =====================================================

def show_header(title, description):

    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">{description}</div>',
        unsafe_allow_html=True
    )


def show_card(label, value, description=""):

    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">{label}</div>
        <div class="info-value">{value}</div>
        <div class="info-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def get_error_value(name):

    if error_summary is None:
        return None

    row = error_summary[
        error_summary["Chi_tieu"] == name
    ]

    if row.empty:
        return None

    return int(row.iloc[0]["So_luong"])


def format_percent(value):

    return f"{value * 100:.2f}%"


# =====================================================
# 10. THANH DIEU HUONG
# =====================================================

with st.sidebar:

    st.markdown("# 📦 DELIVERY AI")

    st.caption("HỆ THỐNG PHÂN TÍCH CHUỖI CUNG ỨNG")

    st.divider()

    page = st.radio(
        "MENU CHÍNH",
        [
            "Tổng quan",
            "Phân tích dữ liệu",
            "Đánh giá mô hình",
            "Dự đoán giao hàng"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### 🤖 Trạng thái hệ thống")

    if bundle is not None:
        st.success("Mô hình đã sẵn sàng")
    else:
        st.error("Chưa tải được mô hình")

    st.markdown("---")

    st.caption(
        "Đề tài: Dự đoán nguy cơ giao hàng trễ "
        "bằng Machine Learning."
    )

    st.caption("Dataset: DataCo Supply Chain")


# =====================================================
# 11. TRANG TONG QUAN
# =====================================================

if page == "Tổng quan":

    show_header(
        "TỔNG QUAN HỆ THỐNG",
        "Theo dõi hoạt động và kết quả của hệ thống dự đoán giao hàng trễ."
    )

    st.markdown("""
    <div class="hero">
        <h1>📦 DELIVERY AI</h1>
        <p>
            Hệ thống ứng dụng Machine Learning để phân tích thông tin
            đơn hàng và nhận diện sớm nguy cơ giao hàng trễ.
        </p>
        <p>
            Hỗ trợ doanh nghiệp theo dõi rủi ro và đưa ra quyết định
            trong hoạt động quản lý chuỗi cung ứng.
        </p>
    </div>
    """, unsafe_allow_html=True)

    total_models = len(comparison) if comparison is not None else 0

    total_features = len(ALL_FEATURES)

    best_f1 = None

    if comparison is not None and "F1-score" in comparison.columns:
        best_f1 = comparison["F1-score"].max()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Thuật toán", "02", "K-NN và Decision Tree")

    with col2:
        st.metric("Cấu hình thử nghiệm", total_models)

    with col3:
        st.metric("Đặc trưng đầu vào", total_features)

    with col4:

        if best_f1 is not None:
            st.metric("F1-score cao nhất", format_percent(best_f1))
        else:
            st.metric("F1-score cao nhất", "N/A")

    st.markdown('<div class="section-title">Chức năng hệ thống</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
        **📊 Phân tích dữ liệu**

        Khám phá dữ liệu, quan sát phân bố và các biểu đồ trực quan.
        """)

        st.success("""
        **🤖 Đánh giá mô hình**

        So sánh các thuật toán và cấu hình huấn luyện.
        """)

    with col2:

        st.warning("""
        **🔍 Dự đoán giao hàng**

        Nhập thông tin đơn hàng và dự đoán nguy cơ giao trễ.
        """)

        st.info("""
        **📈 Phân tích lỗi**

        Theo dõi các trường hợp dự đoán đúng và dự đoán sai.
        """)

    st.markdown('<div class="section-title">Thông tin mô hình hiện tại</div>',
                unsafe_allow_html=True)

    if bundle is not None:

        st.success(
            "Mô hình đã được tải thành công."
        )

        st.write(
            "**Tên mô hình:**",
            bundle.get("model_name", "Không xác định")
        )

    else:

        st.error("Không thể tải mô hình best_model.pkl.")

        if model_error:
            st.code(model_error)


# =====================================================
# 12. TRANG PHAN TICH DU LIEU
# =====================================================

elif page == "Phân tích dữ liệu":

    show_header(
        "PHÂN TÍCH DỮ LIỆU",
        "Trực quan hóa dữ liệu và khám phá các đặc trưng của bộ DataCo Supply Chain."
    )

    st.markdown(
        '<div class="section-title">Tổng quan đặc trưng</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Đặc trưng số", len(NUMERIC_FEATURES))

    with col2:
        st.metric("Đặc trưng phân loại", len(CATEGORICAL_FEATURES))

    with col3:
        st.metric("Tổng đặc trưng", len(ALL_FEATURES))

    st.divider()

    if feature_list is not None:

        st.subheader("Danh sách đặc trưng sử dụng")

        st.dataframe(
            feature_list,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("Biểu đồ phân tích dữ liệu")

    image_files = sorted(
        RESULTS_DIR.glob("*.png")
    )

    if not image_files:

        st.warning(
            "Chưa tìm thấy biểu đồ trong thư mục Results."
        )

    else:

        for i in range(0, len(image_files), 2):

            cols = st.columns(2)

            for j in range(2):

                index = i + j

                if index >= len(image_files):
                    break

                img = image_files[index]

                with cols[j]:

                    st.markdown(
                        f"**{img.stem.replace('_', ' ').title()}**"
                    )

                    st.image(
                        str(img),
                        use_container_width=True
                    )


# =====================================================
# 13. TRANG DANH GIA MO HINH
# =====================================================

elif page == "Đánh giá mô hình":

    show_header(
        "ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH",
        "Phân tích hiệu suất các cấu hình Machine Learning."
    )

    if comparison is None:

        st.error(
            "Không tìm thấy hoặc không đọc được model_comparison.csv."
        )

    else:

        required_columns = [
            "Model",
            "Pipeline",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score"
        ]

        if not all(col in comparison.columns for col in required_columns):

            st.error("File kết quả thiếu các cột cần thiết.")

        else:

            # ---------------------------------------------
            # Bang ket qua
            # ---------------------------------------------

            st.subheader("1. Bảng kết quả thực nghiệm")

            display_data = comparison.copy()

            display_data["Accuracy"] = (
                display_data["Accuracy"] * 100
            ).round(2)

            display_data["Precision"] = (
                display_data["Precision"] * 100
            ).round(2)

            display_data["Recall"] = (
                display_data["Recall"] * 100
            ).round(2)

            display_data["F1-score"] = (
                display_data["F1-score"] * 100
            ).round(2)

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Tải bảng kết quả CSV",
                data=comparison.to_csv(index=False).encode("utf-8-sig"),
                file_name="model_comparison.csv",
                mime="text/csv"
            )

            st.divider()

            # ---------------------------------------------
            # Tim mo hinh tot nhat
            # ---------------------------------------------

            best_f1_index = comparison["F1-score"].idxmax()

            best_f1 = comparison.loc[best_f1_index]

            best_accuracy_index = comparison["Accuracy"].idxmax()

            best_accuracy = comparison.loc[best_accuracy_index]

            st.subheader("2. Cấu hình nổi bật")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**Mô hình có F1-score cao nhất**")

                st.metric(
                    "F1-score",
                    format_percent(best_f1["F1-score"])
                )

                st.write("Mô hình:", best_f1["Model"])
                st.write("Pipeline:", best_f1["Pipeline"])
                st.write("Tham số:", best_f1.get("Tham so", "N/A"))

            with col2:

                st.markdown("**Mô hình có Accuracy cao nhất**")

                st.metric(
                    "Accuracy",
                    format_percent(best_accuracy["Accuracy"])
                )

                st.write("Mô hình:", best_accuracy["Model"])
                st.write("Pipeline:", best_accuracy["Pipeline"])
                st.write("Tham số:", best_accuracy.get("Tham so", "N/A"))

            st.divider()

            # ---------------------------------------------
            # Bieu do so sanh
            # ---------------------------------------------

            st.subheader("3. Biểu đồ so sánh hiệu suất")

            chart_data = comparison.copy()

            if "Tham so" in chart_data.columns:

                chart_data["Cấu hình"] = (
                    chart_data["Model"].astype(str)
                    + " | Pipeline "
                    + chart_data["Pipeline"].astype(str)
                    + " | "
                    + chart_data["Tham so"].astype(str)
                )

            else:

                chart_data["Cấu hình"] = (
                    chart_data["Model"].astype(str)
                    + " | Pipeline "
                    + chart_data["Pipeline"].astype(str)
                )

            selected_metrics = st.multiselect(
                "Chọn chỉ số muốn so sánh",
                ["Accuracy", "Precision", "Recall", "F1-score"],
                default=["Accuracy", "F1-score"]
            )

            if selected_metrics:

                chart = chart_data.set_index("Cấu hình")[
                    selected_metrics
                ]

                st.bar_chart(chart)

            else:

                st.info("Hãy chọn ít nhất một chỉ số.")

            st.divider()

            # ---------------------------------------------
            # Phan tich loi
            # ---------------------------------------------

            st.subheader("4. Phân tích lỗi dự đoán")

            if error_summary is None:

                st.warning(
                    "Chưa tìm thấy error_analysis_summary.txt."
                )

            else:

                total = get_error_value("Tong so mau kiem tra")
                correct = get_error_value("Du doan dung")
                wrong = get_error_value("Du doan sai")
                fp = get_error_value("False Positive")
                fn = get_error_value("False Negative")

                if total is not None:

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Tổng mẫu kiểm tra", f"{total:,}")

                    col2.metric("Dự đoán đúng", f"{correct:,}")

                    col3.metric("Dự đoán sai", f"{wrong:,}")

                    if total > 0:

                        st.progress(
                            min(max(correct / total, 0.0), 1.0),
                            text=f"Tỷ lệ dự đoán đúng: {correct / total * 100:.2f}%"
                        )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "False Positive",
                            f"{fp:,}" if fp is not None else "N/A"
                        )

                        st.caption(
                            "Dự đoán giao trễ nhưng thực tế không trễ."
                        )

                    with col2:

                        st.metric(
                            "False Negative",
                            f"{fn:,}" if fn is not None else "N/A"
                        )

                        st.caption(
                            "Dự đoán không trễ nhưng thực tế lại trễ."
                        )

                    if fp is not None and fn is not None:

                        if fn > fp:

                            st.warning(
                                f"False Negative cao hơn False Positive "
                                f"{fn - fp} trường hợp. Mô hình còn bỏ sót "
                                "một số đơn hàng giao trễ."
                            )

                        elif fp > fn:

                            st.info(
                                "False Positive cao hơn False Negative. "
                                "Mô hình có xu hướng cảnh báo nhầm nhiều hơn."
                            )

                        else:

                            st.info(
                                "False Positive và False Negative bằng nhau."
                            )

                    st.caption(
                        "Lưu ý: Phần phân tích lỗi phản ánh mô hình được dùng "
                        "khi tạo file error_analysis_summary.txt. "
                        "Cần đảm bảo đó là mô hình đang được đánh giá."
                    )


# =====================================================
# 14. TRANG DU DOAN GIAO HANG
# =====================================================

elif page == "Dự đoán giao hàng":

    show_header(
        "DỰ ĐOÁN NGUY CƠ GIAO HÀNG TRỄ",
        "Nhập thông tin đơn hàng để hệ thống phân tích và đưa ra dự đoán."
    )

    if bundle is None:

        st.error(
            "Không tìm thấy mô hình. "
            "Hãy kiểm tra Models/best_model.pkl."
        )

    elif not isinstance(bundle, dict):

        st.error(
            "Cấu trúc file mô hình không đúng. "
            "Cần có model và preprocessor."
        )

    elif "model" not in bundle or "preprocessor" not in bundle:

        st.error(
            "File mô hình thiếu model hoặc preprocessor."
        )

    else:

        model = bundle["model"]

        preprocessor = bundle["preprocessor"]

        encoder = find_encoder(preprocessor)

        if encoder is None:

            st.error(
                "Không tìm thấy OneHotEncoder trong bộ tiền xử lý."
            )

        elif len(encoder.categories_) != len(CATEGORICAL_FEATURES):

            st.error(
                "Số lượng nhóm danh mục trong OneHotEncoder "
                "không khớp danh sách đặc trưng."
            )

        else:

            category_options = encoder.categories_

            st.info(
                "Điền thông tin đơn hàng. Các trường được chia thành "
                "hai nhóm để thuận tiện nhập liệu."
            )

            with st.form("prediction_form"):

                # -----------------------------------------
                # Thong tin van chuyen
                # -----------------------------------------

                st.subheader("1. Thông tin vận chuyển")

                col1, col2 = st.columns(2)

                with col1:

                    scheduled_days = st.number_input(
                        "Thời gian vận chuyển dự kiến (ngày)",
                        min_value=0.0,
                        value=4.0,
                        step=1.0
                    )

                    shipping_mode = st.selectbox(
                        "Phương thức vận chuyển",
                        category_options[6].tolist()
                    )

                    market = st.selectbox(
                        "Thị trường",
                        category_options[4].tolist()
                    )

                    order_region = st.selectbox(
                        "Khu vực đặt hàng",
                        category_options[5].tolist()
                    )

                with col2:

                    order_type = st.selectbox(
                        "Loại đơn hàng",
                        category_options[0].tolist()
                    )

                    customer_segment = st.selectbox(
                        "Phân khúc khách hàng",
                        category_options[2].tolist()
                    )

                    department = st.selectbox(
                        "Phòng ban",
                        category_options[3].tolist()
                    )

                    category = st.selectbox(
                        "Danh mục sản phẩm",
                        category_options[1].tolist()
                    )

                st.divider()

                # -----------------------------------------
                # Thong tin san pham
                # -----------------------------------------

                st.subheader("2. Thông tin sản phẩm và đơn hàng")

                col1, col2 = st.columns(2)

                with col1:

                    discount = st.number_input(
                        "Giảm giá",
                        min_value=0.0,
                        value=0.0
                    )

                    discount_rate = st.number_input(
                        "Tỷ lệ giảm giá",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05
                    )

                    item_price = st.number_input(
                        "Giá sản phẩm trong đơn hàng",
                        min_value=0.0,
                        value=100.0
                    )

                    quantity = st.number_input(
                        "Số lượng sản phẩm",
                        min_value=1,
                        value=1,
                        step=1
                    )

                    sales = st.number_input(
                        "Doanh thu",
                        min_value=0.0,
                        value=100.0
                    )

                with col2:

                    item_total = st.number_input(
                        "Tổng giá trị đơn hàng",
                        min_value=0.0,
                        value=100.0
                    )

                    product_price = st.number_input(
                        "Giá sản phẩm",
                        min_value=0.0,
                        value=100.0
                    )

                    latitude = st.number_input(
                        "Vĩ độ",
                        value=0.0,
                        format="%.6f"
                    )

                    longitude = st.number_input(
                        "Kinh độ",
                        value=0.0,
                        format="%.6f"
                    )

                st.divider()

                submitted = st.form_submit_button(
                    "🔍 PHÂN TÍCH VÀ DỰ ĐOÁN",
                    use_container_width=True
                )

            # ---------------------------------------------
            # Xu ly du doan
            # ---------------------------------------------

            if submitted:

                try:

                    input_values = {
                        "Days for shipment (scheduled)": scheduled_days,
                        "Order Item Discount": discount,
                        "Order Item Discount Rate": discount_rate,
                        "Order Item Product Price": item_price,
                        "Order Item Quantity": quantity,
                        "Sales": sales,
                        "Order Item Total": item_total,
                        "Product Price": product_price,
                        "Latitude": latitude,
                        "Longitude": longitude,
                        "Type": order_type,
                        "Category Name": category,
                        "Customer Segment": customer_segment,
                        "Department Name": department,
                        "Market": market,
                        "Order Region": order_region,
                        "Shipping Mode": shipping_mode
                    }

                    input_data = pd.DataFrame(
                        [input_values],
                        columns=ALL_FEATURES
                    )

                    transformed_data = preprocessor.transform(
                        input_data
                    )

                    prediction = model.predict(
                        transformed_data
                    )[0]

                    st.divider()

                    st.subheader("3. Kết quả dự đoán")

                    if prediction == 1:

                        st.error(
                            "⚠️ ĐƠN HÀNG CÓ NGUY CƠ GIAO TRỄ"
                        )

                        st.warning(
                            "Đơn hàng nên được theo dõi và kiểm tra "
                            "kế hoạch vận chuyển."
                        )

                    else:

                        st.success(
                            "✅ ĐƠN HÀNG KHÔNG CÓ NGUY CƠ GIAO TRỄ "
                            "THEO DỰ ĐOÁN CỦA MÔ HÌNH"
                        )

                        st.info(
                            "Theo kết quả dự đoán, đơn hàng chưa được "
                            "xác định là có nguy cơ giao trễ."
                        )

                    # -------------------------------------
                    # Xac suat du doan
                    # -------------------------------------

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(
                            transformed_data
                        )[0]

                        classes = list(model.classes_)

                        if 1 in classes:

                            late_probability = probabilities[
                                classes.index(1)
                            ]

                            normal_probability = 1 - late_probability

                            col1, col2 = st.columns(2)

                            col1.metric(
                                "Xác suất giao hàng trễ",
                                format_percent(late_probability)
                            )

                            col2.metric(
                                "Xác suất không giao trễ",
                                format_percent(normal_probability)
                            )

                            st.progress(
                                float(late_probability),
                                text="Mức xác suất giao hàng trễ"
                            )

                    st.caption(
                        "Kết quả do mô hình Machine Learning dự đoán, "
                        "không phải cam kết về thời gian giao hàng thực tế."
                    )

                except Exception as error:

                    st.error(
                        f"Đã xảy ra lỗi khi dự đoán: {error}"
                    )


# =====================================================
# 15. CHAN TRANG
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#718096; font-size:12px;">
        DELIVERY AI | Hệ thống dự đoán nguy cơ giao hàng trễ
        <br>
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
