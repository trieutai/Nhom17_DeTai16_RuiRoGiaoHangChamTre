
import pandas as pd
from pathlib import Path


# Lay duong dan thu muc du an
ROOT_DIR = Path(__file__).resolve().parent.parent

# Duong dan file CSV
DATA_PATH = ROOT_DIR / "Data" / "DataCoSupplyChainDataset.csv"


def load_data():
    """
    Doc du lieu tu file CSV
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, encoding="latin1")

    return df


def explore_data(df):
    """
    Kiem tra thong tin co ban cua du lieu
    """

    print("\n========== KICH THUOC DU LIEU ==========")
    print("So dong:", df.shape[0])
    print("So cot:", df.shape[1])

    print("\n========== 5 DONG DAU TIEN ==========")
    print(df.head())

    print("\n========== KIEU DU LIEU ==========")
    print(df.dtypes)

    print("\n========== DU LIEU THIEU ==========")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\n========== DU LIEU TRUNG LAP ==========")
    print("So dong trung lap:", df.duplicated().sum())

    print("\n========== PHAN PHOI NHAN ==========")

    if "Late_delivery_risk" in df.columns:
        print(df["Late_delivery_risk"].value_counts())
    else:
        print("Khong tim thay cot Late_delivery_risk")


if __name__ == "__main__":
    df = load_data()
    explore_data(df)
