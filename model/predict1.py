from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_PATH = "models/best_model.pkl"
EXCEL_PATH = "data/COFICAB_CSR_cleaned.xlsx"

# ---------------- PRINT STYLE ----------------

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_subsection(title):
    print("\n" + "-" * 50)
    print(f" {title}")
    print("-" * 50)

# ---------------- LOAD DATA ----------------

def load_sheets(path):
    return pd.read_excel(path, sheet_name="PLAN_CLEAN")

def normalize(s):
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

def engineer_features(df):
    df = df.copy()

    df["Status"] = normalize(df["Status"])
    df["Start_Date"] = pd.to_datetime(df["Start_Date"], errors="coerce")

    df["Start_Year"] = df["Start_Date"].dt.year
    df["Start_Month"] = df["Start_Date"].dt.month
    df["Start_Quarter"] = df["Start_Date"].dt.quarter

    return df

def select_features(df):
    drop_cols = ["Plan_ID", "Status", "Start_Date"]
    return df.drop(columns=drop_cols, errors="ignore")

# ---------------- RISK MODE ----------------

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

# ---------------- MAIN ----------------

def main():

    # ---------------- LOAD MODEL ----------------
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    threshold = saved["threshold"]
    t_low = saved["t_low"]
    t_high = saved["t_high"]

    print_section("MODEL LOADED")
    print(f"Execution threshold : {threshold:.3f}")
    print(f"Risk LOW boundary   : {t_low:.3f}")
    print(f"Risk HIGH boundary  : {t_high:.3f}")

    # ---------------- LOAD DATA ----------------
    df = load_sheets(EXCEL_PATH)
    df = engineer_features(df)

    # keep only future-like inference data (optional rule)
    df = df[df["Status"] == "Finished"]

    X = select_features(df)

    # ---------------- PREDICTIONS ----------------
    proba = model.predict_proba(X)[:, 1]

    # ================= MODE 1 =================
    print_section("MODE 1: EXECUTION PREDICTION")

    exec_preds = execution_mode(proba, threshold)

    print_subsection("Sample Predictions")
    print(exec_preds[:20])

    print_subsection("Execution Probabilities")
    print(np.round(proba[:20], 3))

    # ================= MODE 2 =================
    print_section("MODE 2: RISK DETECTION SYSTEM")

    risk_labels = risk_mode(proba, t_low, t_high)

    unique, counts = np.unique(risk_labels, return_counts=True)

    print_subsection("Risk Distribution")
    for u, c in zip(unique, counts):
        print(f"{u}: {c}")

    print_subsection("Sample Risk Output")
    print(risk_labels[:20])

    # ================= FULL OUTPUT DATAFRAME =================
    df_results = df.copy()
    df_results["execution_probability"] = proba
    df_results["execution_prediction"] = exec_preds
    df_results["risk_label"] = risk_labels

    print_section("FINAL OUTPUT SAMPLE")
    print(df_results[[
        "Plan_ID",
        "execution_probability",
        "execution_prediction",
        "risk_label"
    ]].head(10))

# ---------------- RUN ----------------

if __name__ == "__main__":
    main()