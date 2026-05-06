from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
EXCEL_PATH = "data/COFICAB_CSR_cleaned.xlsx"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_JOBLIB_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
MODEL_PKL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
RESULTS_JSON_PATH = os.path.join(MODEL_DIR, "evaluation_results.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# ---------------- PRETTY PRINT ----------------

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_subsection(title):
    print("\n" + "-" * 50)
    print(f" {title}")
    print("-" * 50)

# ---------------- DATA ----------------

def load_sheets(path):
    return (
        pd.read_excel(path, sheet_name="PLAN_CLEAN"),
        pd.read_excel(path, sheet_name="EXEC_CLEAN"),
    )

def normalize(s):
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

def engineer_features(plan):
    df = plan.copy()

    df["Status"] = normalize(df["Status"])
    df["Start_Date"] = pd.to_datetime(df["Start_Date"], errors="coerce")

    df["Start_Year"] = df["Start_Date"].dt.year
    df["Start_Month"] = df["Start_Date"].dt.month
    df["Start_Quarter"] = df["Start_Date"].dt.quarter

    return df

def build_targets(plan, exec_):
    df = engineer_features(plan)
    executed_ids = set(exec_["Plan_ID"].dropna().astype(int))
    df["y_executed"] = df["Plan_ID"].astype(int).isin(executed_ids).astype(int)
    return df

# ---------------- FEATURES ----------------

def select_features(df):
    drop_cols = ["Plan_ID", "Status", "Start_Date", "y_executed"]
    X = df.drop(columns=drop_cols, errors="ignore")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    return X, num_cols, cat_cols

# ---------------- PREPROCESS ----------------

def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

# ---------------- MODEL ----------------

def build_model():
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight={0: 3, 1: 1},
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# ---------------- THRESHOLDS ----------------

def compute_thresholds(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1)

    best_threshold = thresholds[max(best_idx - 1, 0)]

    t_low = np.percentile(proba[y_true == 0], 20)
    t_high = np.percentile(proba[y_true == 1], 80)

    return best_threshold, t_low, t_high

# ---------------- RISK SYSTEM ----------------

def risk_mode(proba, t_low, t_high):
    labels = []

    for p in proba:
        if p <= t_low:
            labels.append("HIGH RISK (NOT executed)")
        elif p >= t_high:
            labels.append("SAFE (executed)")
        else:
            labels.append("UNCERTAIN")

    return np.array(labels)

def execution_mode(proba, threshold):
    return (proba >= threshold).astype(int)

# ---------------- RISK METRICS ----------------

def evaluate_risk_mode(y_true, risk_labels):
    preds = np.where(risk_labels == "HIGH RISK (NOT executed)", 0, 1)

    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds)
    }

# ---------------- MAIN ----------------

def main():
    plan, exec_ = load_sheets(EXCEL_PATH)
    df = build_targets(plan, exec_)

    # ---------------- DATA ----------------
    print_section("ORIGINAL DATA")
    print(pd.crosstab(df["Status"], df["y_executed"]))

    df = df[df["Status"] == "Finished"]

    print_section("FILTERED DATA (Finished only)")
    print(df["y_executed"].value_counts())

    # ---------------- SPLIT ----------------
    X, num_cols, cat_cols = select_features(df)
    y = df["y_executed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    # ---------------- MODEL ----------------
    preprocessor = make_preprocessor(num_cols, cat_cols)
    model = build_model()

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    # ---------------- THRESHOLDS ----------------
    val_proba = pipe.predict_proba(X_val)[:, 1]
    best_threshold, t_low, t_high = compute_thresholds(y_val, val_proba)

    print_section("THRESHOLDS")
    print(f"Execution threshold : {best_threshold:.3f}")
    print(f"Risk LOW boundary   : {t_low:.3f}")
    print(f"Risk HIGH boundary  : {t_high:.3f}")

    # ---------------- TEST ----------------
    test_proba = pipe.predict_proba(X_test)[:, 1]

    # ================= MODE 1 =================
    print_section("MODE 1: EXECUTION PREDICTION")

    exec_preds = execution_mode(test_proba, best_threshold)

    print_subsection("Confusion Matrix")
    cm = confusion_matrix(y_test, exec_preds)
    print(cm)

    print_subsection("Classification Report")
    print(classification_report(y_test, exec_preds))

    execution_metrics = {
        "accuracy": accuracy_score(y_test, exec_preds),
        "precision": precision_score(y_test, exec_preds, zero_division=0),
        "recall": recall_score(y_test, exec_preds),
        "f1": f1_score(y_test, exec_preds)
    }

    # ================= MODE 2 =================
    print_section("MODE 2: RISK DETECTION SYSTEM")

    risk_labels = risk_mode(test_proba, t_low, t_high)

    unique, counts = np.unique(risk_labels, return_counts=True)

    print_subsection("Risk Distribution")
    for u, c in zip(unique, counts):
        print(f"{u}: {c}")

    print_subsection("Sample Predictions")
    print(risk_labels[:20])

    print_subsection("Risk Mode Metrics")

    risk_metrics = evaluate_risk_mode(y_test.values, risk_labels)

    print(f"Accuracy : {risk_metrics['accuracy']:.3f}")
    print(f"Precision: {risk_metrics['precision']:.3f}")
    print(f"Recall   : {risk_metrics['recall']:.3f}")
    print(f"F1-score : {risk_metrics['f1']:.3f}")

    # ---------------- SAVE MODEL (.joblib + .pkl) ----------------
    dump(pipe, MODEL_JOBLIB_PATH)

    import pickle
    with open(MODEL_PKL_PATH, "wb") as f:
        pickle.dump({
            "model": pipe,
            "threshold": best_threshold,
            "t_low": t_low,
            "t_high": t_high
        }, f)

    # ---------------- SAVE RESULTS JSON ----------------
    results = {
        "thresholds": {
            "execution_threshold": float(best_threshold),
            "risk_low": float(t_low),
            "risk_high": float(t_high)
        },
        "mode_1_execution": execution_metrics,
        "mode_2_risk": risk_metrics
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=4)

    # ---------------- FINAL OUTPUT ----------------
    print_section("MODEL + RESULTS SAVED")

    print(f"Joblib model : {MODEL_JOBLIB_PATH}")
    print(f"Pickle model : {MODEL_PKL_PATH}")
    print(f"JSON results : {RESULTS_JSON_PATH}")

# ---------------- RUN ----------------

if __name__ == "__main__":
    main()