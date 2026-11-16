
from Src.load_data import load_data, explore_data


def main():

    print("======================================")
    print(" DU DOAN NGUY CO GIAO HANG TRE")
    print("======================================")

    # Doc du lieu
    df = load_data()

    # Kham pha du lieu
    explore_data(df)


if __name__ == "__main__":
    main()
