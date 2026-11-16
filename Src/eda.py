
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from Src.load_data import load_data


# ==========================================
# 1. CAU HINH DUONG DAN
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "Results"

RESULT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# 2. PHAN TICH TONG QUAN DU LIEU
# ==========================================

def analyze_overview(df):

    print("\n========== TONG QUAN DU LIEU ==========")

    print("So dong:", df.shape[0])
    print("So cot:", df.shape[1])

    print("\nThong tin cac cot:")
    print(df.info())

    print("\nThong ke mo ta:")
    print(df.describe().round(2))

    # Luu thong ke mo ta
    df.describe().to_csv(
        RESULT_DIR / "thong_ke_mo_ta.csv"
    )


# ==========================================
# 3. PHAN TICH BIEN MUC TIEU
# ==========================================

def analyze_target(df):

    print("\n========== PHAN TICH BIEN MUC TIEU ==========")

    target_count = df["Late_delivery_risk"].value_counts()

    print(target_count)

    print("\nTy le phan tram:")
    print(
        (df["Late_delivery_risk"].value_counts(normalize=True) * 100).round(2)
    )

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Late_delivery_risk"
    )

    plt.title("Phan phoi nguy co giao hang tre")
    plt.xlabel("Late Delivery Risk")
    plt.ylabel("So luong don hang")

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "01_phan_phoi_nhan.png",
        dpi=300
    )

    plt.show()
    plt.close()


# ==========================================
# 4. PHAN TICH DU LIEU THIEU
# ==========================================

def analyze_missing(df):

    print("\n========== PHAN TICH DU LIEU THIEU ==========")

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    print(missing)

    if len(missing) == 0:
        print("Khong co du lieu thieu.")
        return

    missing_df = pd.DataFrame({
        "So luong thieu": missing,
        "Ty le thieu (%)": (missing / len(df) * 100).round(2)
    })

    print("\nChi tiet du lieu thieu:")
    print(missing_df)

    missing_df.to_csv(
        RESULT_DIR / "du_lieu_thieu.csv"
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=missing.values,
        y=missing.index
    )

    plt.title("So luong du lieu thieu theo cot")
    plt.xlabel("So luong gia tri thieu")
    plt.ylabel("Ten cot")

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "02_du_lieu_thieu.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 5. PHAN TICH DAC TRUNG SO
# ==========================================

def analyze_numeric(df):

    print("\n========== PHAN TICH DAC TRUNG SO ==========")

    numeric_columns = [
        "Days for shipment (scheduled)",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Product Price"
    ]

    # Chi giu cac cot ton tai trong du lieu
    numeric_columns = [
        col for col in numeric_columns
        if col in df.columns
    ]

    print("\nThong ke cac dac trung so:")
    print(df[numeric_columns].describe().round(2))

    # Histogram
    for col in numeric_columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=col,
            kde=True
        )

        plt.title("Histogram - " + col)
        plt.xlabel(col)
        plt.ylabel("Tan suat")

        plt.tight_layout()

        filename = col.replace(" ", "_").replace("/", "_")

        plt.savefig(
            RESULT_DIR / f"hist_{filename}.png",
            dpi=300
        )

        plt.close()

    # Boxplot
    for col in numeric_columns:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            data=df,
            x=col
        )

        plt.title("Boxplot - " + col)
        plt.xlabel(col)

        plt.tight_layout()

        filename = col.replace(" ", "_").replace("/", "_")

        plt.savefig(
            RESULT_DIR / f"box_{filename}.png",
            dpi=300
        )

        plt.close()

    print("\nDa tao Histogram va Boxplot.")


# ==========================================
# 6. PHAN TICH DAC TRUNG PHAN LOAI
# ==========================================

def analyze_categorical(df):

    print("\n========== PHAN TICH DAC TRUNG PHAN LOAI ==========")

    categorical_columns = [
        "Type",
        "Category Name",
        "Customer Segment",
        "Department Name",
        "Market",
        "Order Region",
        "Shipping Mode"
    ]

    for col in categorical_columns:

        if col not in df.columns:
            continue

        print(f"\nTan suat cua cot: {col}")

        print(df[col].value_counts().head(10))

        plt.figure(figsize=(10, 5))

        order = df[col].value_counts().head(10).index

        sns.countplot(
            data=df,
            y=col,
            order=order
        )

        plt.title("Tan suat - " + col)
        plt.xlabel("So luong")
        plt.ylabel(col)

        plt.tight_layout()

        filename = col.replace(" ", "_")

        plt.savefig(
            RESULT_DIR / f"category_{filename}.png",
            dpi=300
        )

        plt.close()

    print("\nDa tao bieu do dac trung phan loai.")


# ==========================================
# 7. PHAN TICH TUONG QUAN
# ==========================================

def analyze_correlation(df):

    print("\n========== PHAN TICH TUONG QUAN ==========")

    # Khong dua cac cot ID va thong tin giao hang
    # thuc te vao phan tich tuong quan dac trung.
    correlation_columns = [
        "Days for shipment (scheduled)",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Product Price",
        "Late_delivery_risk"
    ]

    correlation_columns = [
        col for col in correlation_columns
        if col in df.columns
    ]

    corr = df[correlation_columns].corr()

    print(corr.round(2))

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Ma tran tuong quan giua cac dac trung")

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "03_ma_tran_tuong_quan.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 8. CHUONG TRINH CHINH
# ==========================================

def main():

    print("======================================")
    print(" KHAM PHA VA PHAN TICH DU LIEU - EDA")
    print("======================================")

    df = load_data()

    analyze_overview(df)

    analyze_target(df)

    analyze_missing(df)

    analyze_numeric(df)

    analyze_categorical(df)

    analyze_correlation(df)

    print("\n======================================")
    print(" HOAN THANH PHAN TICH EDA")
    print(" Cac ket qua da duoc luu trong Results")
    print("======================================")


if __name__ == "__main__":
    main()
