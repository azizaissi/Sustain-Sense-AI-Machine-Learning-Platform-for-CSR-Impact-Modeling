import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
CLASSIC_DIR = BASE_DIR / "csr_models_output"

PLAN_SHEET = "PLAN_CLEAN"
EXEC_SHEET = "EXEC_CLEAN"

TARGET_MODEL_PATHS = {
    "Actual_Impact_Value": MODEL_DIR / "Actual_Impact_Value_best_model.joblib",
    "Internal_Participants": MODEL_DIR / "Internal_Participants_best_model.joblib",
    "Participation_Rate": MODEL_DIR / "Participation_Rate_best_model.joblib",
    "execution_rate": MODEL_DIR / "execution_rate_best_model.joblib",
    "Actual_Cost_EUR": CLASSIC_DIR / "Actual_Cost_EUR_best_model.joblib",
}


# =========================
# LOAD MODELS
# =========================
models = {}
missing = []

for target, path in TARGET_MODEL_PATHS.items():
    if path.exists():
        models[target] = joblib.load(path)
    else:
        missing.append(target)

if len(models) == 0:
    raise ValueError("❌ No models found. Run training first.")

if missing:
    print("⚠ Missing models:", missing)


# =========================
# FEATURE ENGINEERING (MATCH TRAINING)
# =========================
def add_date_features(df, col, prefix):
    if col not in df.columns:
        return df

    dt = pd.to_datetime(df[col], errors="coerce")

    df[f"{prefix}_year"] = dt.dt.year
    df[f"{prefix}_month"] = dt.dt.month
    df[f"{prefix}_day"] = dt.dt.day
    df[f"{prefix}_dow"] = dt.dt.dayofweek
    df[f"{prefix}_quarter"] = dt.dt.quarter
    df[f"{prefix}_is_month_end"] = dt.dt.is_month_end.astype(float)
    df[f"{prefix}_is_month_start"] = dt.dt.is_month_start.astype(float)

    return df


def add_text_length_features(df, cols):
    for col in cols:
        if col in df.columns:
            s = df[col].fillna("").astype(str)
            df[f"{col}_len"] = s.str.len()
            df[f"{col}_words"] = s.str.split().str.len()
    return df


def safe_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =========================
# BUILD DATASET (IDENTICAL TO TRAINING)
# =========================
def build_dataset(file_path):

    plan = pd.read_excel(file_path, sheet_name=PLAN_SHEET)
    exe = pd.read_excel(file_path, sheet_name=EXEC_SHEET)

    df = exe.merge(
        plan.add_prefix("plan_"),
        left_on="Plan_ID",
        right_on="plan_Plan_ID",
        how="left",
    )

    # numeric cleaning
    numeric_cols = [
        "Expected_Impact_Value",
        "Planned_Cost_EUR",
        "Planned_Internal_Volunteers",
        "Duration_Months",
        "plan_Planned_Cost_EUR",
        "plan_Expected_Impact_Value",
    ]

    df = safe_numeric(df, numeric_cols)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).clip(lower=0)

    # log features (IMPORTANT)
    if "plan_Expected_Impact_Value" in df.columns:
        df["plan_Expected_Impact_Value"] = np.log1p(df["plan_Expected_Impact_Value"])

    if "plan_Planned_Cost_EUR" in df.columns:
        df["plan_Planned_Cost_EUR"] = np.log1p(df["plan_Planned_Cost_EUR"])

    # engineered features
    if "plan_Expected_Impact_Value" in df.columns:
        df["Cost_per_Impact"] = (
            df["plan_Planned_Cost_EUR"] /
            (df["plan_Expected_Impact_Value"] + 1)
        )

    if "Internal_Participants" in df.columns:
        df["execution_rate"] = (
            df["Internal_Participants"] /
            df["plan_Planned_Internal_Volunteers"] * 100
        )

    df = add_date_features(df, "Execution_Date", "exec_date")
    df = add_date_features(df, "plan_Start_Date", "plan_start")

    df = add_text_length_features(
        df,
        [
            "Organizer",
            "External_Partners",
            "plan_Activity_Description",
            "plan_Organization",
            "plan_External_Entity",
        ],
    )

    return df


# =========================
# ALIGN INPUT WITH MODEL
# =========================
def align_input(df, model):
    if isinstance(model, TransformedTargetRegressor):
        prep = model.regressor_.named_steps["prep"]
    else:
        prep = model.named_steps["prep"]

    cols = list(prep.feature_names_in_)

    X = df.copy()

    for col in cols:
        if col not in X.columns:
            X[col] = np.nan

    return X[cols]


# =========================
# PREDICT
# =========================
def predict_from_file(file_path):

    df = build_dataset(file_path)
    result = df.copy()

    for target, model in models.items():

        X = align_input(df, model)
        preds = model.predict(X)

        # inverse log for cost
        if target == "Actual_Cost_EUR":
            preds = np.expm1(preds)

        result[f"Pred_{target}"] = preds

    return result


# =========================
# EVALUATION
# =========================
def evaluate(file_path):

    df = build_dataset(file_path)

    rows = []

    for target, model in models.items():

        if target not in df.columns:
            continue

        X = align_input(df, model)

        y_true = df[target].values
        y_pred = model.predict(X)

        if target == "Actual_Cost_EUR":
            y_pred = np.expm1(y_pred)

        rows.append({
            "Target": target,
            "R2": round(r2_score(y_true, y_pred), 4),
            "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        })

    return pd.DataFrame(rows)


# =========================
# FEATURE IMPORTANCE
# =========================
def get_feature_importance(target="Actual_Impact_Value"):

    if target not in models:
        return pd.DataFrame()

    model = models[target]

    if isinstance(model, TransformedTargetRegressor):
        reg = model.regressor_
    else:
        reg = model

    prep = reg.named_steps["prep"]
    model_core = reg.named_steps["model"]

    if not hasattr(model_core, "feature_importances_"):
        return pd.DataFrame()

    names = prep.get_feature_names_out()
    imp = model_core.feature_importances_

    return (
        pd.DataFrame({"feature": names, "importance": imp})
        .sort_values("importance", ascending=False)
    )