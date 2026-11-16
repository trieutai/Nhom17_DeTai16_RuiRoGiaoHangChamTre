
import time
import json
import platform
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from Src.preprocessing import prepare_data


# ==========================================
# 1. CAU HINH DU AN
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT_DIR / "Results"
MODEL_DIR = ROOT_DIR / "Models"

RESULT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

TRAIN_SAMPLES = 15000
TEST_SAMPLES = 5000


# ==========================================
# 2. LAY MAU DU LIEU
# ==========================================

def sample_data(data):

    print("\n========== LAY MAU DU LIEU ==========")

    X_train_A = data["X_train_A"]
    X_test_A = data["X_test_A"]

    X_train_B = data["X_train_B"]
    X_test_B = data["X_test_B"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    # Kiem tra so luong mau
    if len(y_train) < TRAIN_SAMPLES:
        raise ValueError(
            f"Du lieu Train chi co {len(y_train)} mau, "
            f"khong du {TRAIN_SAMPLES} mau yeu cau."
        )

    if len(y_test) < TEST_SAMPLES:
        raise ValueError(
            f"Du lieu Test chi co {len(y_test)} mau, "
            f"khong du {TEST_SAMPLES} mau yeu cau."
        )

    # Lay mau Train
    train_indices, _ = train_test_split(
        np.arange(len(y_train)),
        train_size=TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    # Lay mau Test
    test_indices, _ = train_test_split(
        np.arange(len(y_test)),
        train_size=TEST_SAMPLES,
        random_state=RANDOM_STATE,
        stratify=y_test
    )

    # Lay du lieu Pipeline A
    X_train_A = X_train_A[train_indices]
    X_test_A = X_test_A[test_indices]

    # Lay du lieu Pipeline B
    X_train_B = X_train_B[train_indices]
    X_test_B = X_test_B[test_indices]

    y_train = y_train.iloc[train_indices]
    y_test = y_test.iloc[test_indices]

    print("So mau Train:", len(y_train))
    print("So mau Test:", len(y_test))

    return {
        "X_train_A": X_train_A,
        "X_test_A": X_test_A,
        "X_train_B": X_train_B,
        "X_test_B": X_test_B,
        "y_train": y_train,
        "y_test": y_test,

        # Giu lai bo tien xu ly
        "pipeline_A": data["pipeline_A"],
        "pipeline_B": data["pipeline_B"]
    }


# ==========================================
# 3. DANH GIA MO HINH
# ==========================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "y_pred": y_pred
    }


# ==========================================
# 4. HUAN LUYEN K-NN
# ==========================================

def train_knn(X_train, X_test, y_train, y_test,
              pipeline_name, k):

    print(f"\nDang huan luyen K-NN {pipeline_name}, K={k}")

    model = KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform",
        metric="minkowski",
        p=2,
        n_jobs=-1
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    evaluation = evaluate_model(
        model, X_test, y_test
    )

    return model, evaluation, training_time


# ==========================================
# 5. HUAN LUYEN DECISION TREE
# ==========================================

def train_decision_tree(X_train, X_test, y_train, y_test,
                        pipeline_name, depth):

    print(
        f"\nDang huan luyen Decision Tree "
        f"{pipeline_name}, max_depth={depth}"
    )

    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_split=2,
        random_state=RANDOM_STATE
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    evaluation = evaluate_model(
        model, X_test, y_test
    )

    return model, evaluation, training_time


# ==========================================
# 6. LUU CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(y_test, y_pred, model_name):

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Khong tre", "Giao tre"],
        yticklabels=["Khong tre", "Giao tre"]
    )

    plt.title("Confusion Matrix - " + model_name)
    plt.xlabel("Nhan du doan")
    plt.ylabel("Nhan thuc te")

    plt.tight_layout()

    filename = model_name.replace(" ", "_")

    plt.savefig(
        RESULT_DIR / f"confusion_{filename}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 7. PHAN TICH LOI DU DOAN
# ==========================================

def analyze_errors(best_model, X_test, y_test, y_pred):

    print("\n========== PHAN TICH LOI DU DOAN ==========")

    # Tao bang ket qua du doan
    error_df = pd.DataFrame({
        "Du_lieu_goc_index": y_test.index,
        "Nhan_thuc_te": y_test.to_numpy(),
        "Nhan_du_doan": y_pred
    })

    # Phan loai cac truong hop du doan
    error_df["Loai_du_doan"] = np.where(
        error_df["Nhan_thuc_te"] == error_df["Nhan_du_doan"],
        "Du doan dung",
        "Du doan sai"
    )

    # False Positive: Du doan tre nhung thuc te khong tre
    error_df.loc[
        (error_df["Nhan_thuc_te"] == 0) &
        (error_df["Nhan_du_doan"] == 1),
        "Loai_du_doan"
    ] = "False Positive"

    # False Negative: Du doan khong tre nhung thuc te tre
    error_df.loc[
        (error_df["Nhan_thuc_te"] == 1) &
        (error_df["Nhan_du_doan"] == 0),
        "Loai_du_doan"
    ] = "False Negative"

    # Lay xac suat du doan neu mo hinh ho tro
    if hasattr(best_model, "predict_proba"):

        probabilities = best_model.predict_proba(X_test)

        class_labels = list(best_model.classes_)

        if 1 in class_labels:
            late_index = class_labels.index(1)

            error_df["Xac_suat_giao_tre"] = probabilities[
                :, late_index
            ]

    # Chi giu lai cac dong du doan sai
    errors = error_df[
        error_df["Loai_du_doan"] != "Du doan dung"
    ].copy()

    # Luu danh sach loi
    errors.to_csv(
        RESULT_DIR / "prediction_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Thong ke loi
    total = len(error_df)

    fp = len(
        error_df[error_df["Loai_du_doan"] == "False Positive"]
    )

    fn = len(
        error_df[error_df["Loai_du_doan"] == "False Negative"]
    )

    correct = len(
        error_df[error_df["Loai_du_doan"] == "Du doan dung"]
    )

    summary = pd.DataFrame({
        "Chi_tieu": [
            "Tong so mau kiem tra",
            "Du doan dung",
            "Du doan sai",
            "False Positive",
            "False Negative"
        ],
        "So_luong": [
            total,
            correct,
            len(errors),
            fp,
            fn
        ]
    })

    summary.to_csv(
        RESULT_DIR / "error_analysis_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Tong so mau kiem tra:", total)
    print("Du doan dung:", correct)
    print("Du doan sai:", len(errors))
    print("False Positive:", fp)
    print("False Negative:", fn)

    print("\nDa luu ket qua phan tich loi.")


# ==========================================
# 8. LUU THONG TIN TAI LAP THI NGHIEM
# ==========================================

def save_reproducibility_info():

    information = {
        "Thoi_gian_chay": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Random_state": RANDOM_STATE,

        "Train_samples": TRAIN_SAMPLES,

        "Test_samples": TEST_SAMPLES,

        "Python_version": platform.python_version(),

        "Pandas_version": pd.__version__,

        "NumPy_version": np.__version__,

        "Scikit_learn_version": sklearn.__version__,

        "Train_test_split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "stratify": True
        },

        "Models": {
            "KNN": {
                "n_neighbors": [5, 9],
                "weights": "uniform",
                "metric": "minkowski",
                "p": 2
            },

            "Decision_Tree": {
                "max_depth": [10, 20],
                "min_samples_split": 2,
                "random_state": RANDOM_STATE
            }
        },

        "Pipelines": [
            "Pipeline A",
            "Pipeline B"
        ],

        "Evaluation_metrics": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "Confusion Matrix"
        ]
    }

    with open(
        RESULT_DIR / "reproducibility_info.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            information,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nDa luu thong tin tai lap thi nghiem.")


# ==========================================
# 9. CHAY CAC THI NGHIEM
# ==========================================

def run_experiments():

    print("======================================")
    print(" BAT DAU HUAN LUYEN MO HINH")
    print("======================================")

    # Tien xu ly du lieu
    data = prepare_data()

    # Lay mau
    data = sample_data(data)

    y_train = data["y_train"]
    y_test = data["y_test"]

    results = []
    trained_models = []

    pipelines = [
        ("A", data["X_train_A"], data["X_test_A"]),
        ("B", data["X_train_B"], data["X_test_B"])
    ]

    for pipeline_name, X_train, X_test in pipelines:

        # K-NN
        for k in [5, 9]:

            model, evaluation, training_time = train_knn(
                X_train,
                X_test,
                y_train,
                y_test,
                pipeline_name,
                k
            )

            model_name = f"KNN_Pipeline_{pipeline_name}_K{k}"

            results.append({
                "Model": "K-NN",
                "Pipeline": pipeline_name,
                "Tham so": f"K={k}",
                "Accuracy": evaluation["accuracy"],
                "Precision": evaluation["precision"],
                "Recall": evaluation["recall"],
                "F1-score": evaluation["f1_score"],
                "Training Time (s)": training_time
            })

            trained_models.append({
                "name": model_name,
                "model": model,
                "evaluation": evaluation,
                "preprocessor": data[f"pipeline_{pipeline_name}"],
                "X_test": X_test
            })

            save_confusion_matrix(
                y_test,
                evaluation["y_pred"],
                model_name
            )

        # Decision Tree
        for depth in [10, 20]:

            model, evaluation, training_time = train_decision_tree(
                X_train,
                X_test,
                y_train,
                y_test,
                pipeline_name,
                depth
            )

            model_name = (
                f"Decision_Tree_Pipeline_{pipeline_name}_D{depth}"
            )

            results.append({
                "Model": "Decision Tree",
                "Pipeline": pipeline_name,
                "Tham so": f"max_depth={depth}",
                "Accuracy": evaluation["accuracy"],
                "Precision": evaluation["precision"],
                "Recall": evaluation["recall"],
                "F1-score": evaluation["f1_score"],
                "Training Time (s)": training_time
            })

            trained_models.append({
                "name": model_name,
                "model": model,
                "evaluation": evaluation,
                "preprocessor": data[f"pipeline_{pipeline_name}"],
                "X_test": X_test
            })

            save_confusion_matrix(
                y_test,
                evaluation["y_pred"],
                model_name
            )

    # ======================================
    # 10. TONG HOP KET QUA
    # ======================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1-score",
        ascending=False
    )

    print("\n========== BANG KET QUA ==========")

    print(results_df.to_string(index=False))

    results_df.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # 11. LUU MO HINH TOT NHAT
    # ======================================

    best_row = results_df.iloc[0]

    best_name = best_row["Model"]
    best_pipeline = best_row["Pipeline"]
    best_parameter = best_row["Tham so"]

    if best_name == "K-NN":

        best_model_name = (
            f"KNN_Pipeline_{best_pipeline}_"
            f"K{best_parameter.split('=')[1]}"
        )

    else:

        best_model_name = (
            f"Decision_Tree_Pipeline_{best_pipeline}_"
            f"D{best_parameter.split('=')[1]}"
        )

    best = next(
        item for item in trained_models
        if item["name"] == best_model_name
    )

    joblib.dump(
        {
            "model": best["model"],
            "preprocessor": best["preprocessor"],
            "model_name": best["name"]
        },
        MODEL_DIR / "best_model.pkl"
    )

    print("\nMo hinh tot nhat:", best["name"])

    print("\nBao cao phan loai:")

    report = classification_report(
        y_test,
        best["evaluation"]["y_pred"],
        target_names=["Khong tre", "Giao tre"],
        zero_division=0,
        output_dict=True
    )

    print(
        classification_report(
            y_test,
            best["evaluation"]["y_pred"],
            target_names=["Khong tre", "Giao tre"],
            zero_division=0
        )
    )

    # Luu classification report
    pd.DataFrame(report).transpose().to_csv(
        RESULT_DIR / "classification_report_best_model.csv",
        encoding="utf-8-sig"
    )

    # ======================================
    # 12. PHAN TICH LOI MO HINH TOT NHAT
    # ======================================

    analyze_errors(
        best["model"],
        best["X_test"],
        y_test,
        best["evaluation"]["y_pred"]
    )

    # ======================================
    # 13. LUU THONG TIN TAI LAP
    # ======================================

    save_reproducibility_info()

    print("\n======================================")
    print(" HOAN THANH HUAN LUYEN MO HINH")
    print("======================================")


# ==========================================
# 14. CHUONG TRINH CHINH
# ==========================================

if __name__ == "__main__":

    run_experiments()
