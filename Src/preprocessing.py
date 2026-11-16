
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler

from Src.load_data import load_data


# ==========================================
# 1. CAU HINH
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "Results"

RESULT_DIR.mkdir(exist_ok=True)

TARGET = "Late_delivery_risk"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ==========================================
# 2. LUA CHON DAC TRUNG
# ==========================================

numeric_features = [
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

categorical_features = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Region",
    "Shipping Mode"
]


# ==========================================
# 3. XAY DUNG PIPELINE A
# ==========================================

def build_pipeline_A():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor


# ==========================================
# 4. XAY DUNG PIPELINE B
# ==========================================

def build_pipeline_B():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("scaler", RobustScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Khong ro"
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor


# ==========================================
# 5. TIEN XU LY DU LIEU
# ==========================================

def prepare_data():

    print("\n========== BAT DAU TIEN XU LY ==========")

    # Doc du lieu
    df = load_data()

    print("Kich thuoc du lieu ban dau:", df.shape)

    # Danh sach dac trung
    all_features = numeric_features + categorical_features

    # Kiem tra cot
    missing_columns = [
        col for col in all_features + [TARGET]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Khong tim thay cac cot: {missing_columns}"
        )

    # Loai bo cac dong khong co nhan
    df = df.dropna(subset=[TARGET]).copy()

    # Kiem tra gia tri nhan
    print("\nPhan bo nhan:")
    print(df[TARGET].value_counts())

    # ======================================
    # KIEM TRA RO RI DU LIEU
    # ======================================

    leakage_candidates = [
        "Days for shipping (real)",
        "Delivery Status",
        "Late_delivery",
        "Actual Shipping Date",
        "Delivery Date"
    ]

    found_leakage_columns = [
        col for col in leakage_candidates
        if col in df.columns
    ]

    print("\n========== KIEM TRA RO RI DU LIEU ==========")

    if found_leakage_columns:
        print("Phat hien cac cot nguy co ro ri:")
        for col in found_leakage_columns:
            print("-", col)
    else:
        print("Khong tim thay cac cot trong danh sach kiem tra.")

    # Cac cot tren khong duoc dua vao danh sach dac trung.
    # Chi su dung numeric_features va categorical_features
    # da khai bao o phia tren.

    # ======================================
    # TACH DAC TRUNG VA NHAN
    # ======================================

    X = df[all_features].copy()
    y = df[TARGET].copy()

    print("\nSo luong dac trung:", len(all_features))
    print("So luong nhan:", len(y))

    # ======================================
    # CHIA TRAIN VA TEST
    # ======================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n========== CHIA DU LIEU ==========")

    print("Tap Train:", X_train.shape)
    print("Tap Test:", X_test.shape)

    # ======================================
    # TAO PIPELINE
    # ======================================

    pipeline_A = build_pipeline_A()
    pipeline_B = build_pipeline_B()

    # Chi fit tren tap Train
    X_train_A = pipeline_A.fit_transform(X_train)
    X_test_A = pipeline_A.transform(X_test)

    X_train_B = pipeline_B.fit_transform(X_train)
    X_test_B = pipeline_B.transform(X_test)

    # ======================================
    # LUU DANH SACH DAC TRUNG
    # ======================================

    feature_info = pd.DataFrame({
        "Ten dac trung": all_features,
        "Loai dac trung": (
            ["So"] * len(numeric_features)
            + ["Phan loai"] * len(categorical_features)
        )
    })

    feature_info.to_csv(
        RESULT_DIR / "danh_sach_dac_trung.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # IN KET QUA
    # ======================================

    print("\n========== KET QUA PIPELINE A ==========")
    print("Train:", X_train_A.shape)
    print("Test:", X_test_A.shape)

    print("\n========== KET QUA PIPELINE B ==========")
    print("Train:", X_train_B.shape)
    print("Test:", X_test_B.shape)

    print("\n========== HOAN THANH TIEN XU LY ==========")

    return {
        "X_train_A": X_train_A,
        "X_test_A": X_test_A,
        "X_train_B": X_train_B,
        "X_test_B": X_test_B,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline_A": pipeline_A,
        "pipeline_B": pipeline_B
    }


# ==========================================
# 6. CHAY CHUONG TRINH
# ==========================================

if __name__ == "__main__":

    data = prepare_data()

    print("\nDA XU LY DU LIEU THANH CONG!")
