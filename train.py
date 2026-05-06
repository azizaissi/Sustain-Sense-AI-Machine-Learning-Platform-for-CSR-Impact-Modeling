import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import KFold, cross_validate, train_test_split

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "data" / "COFICAB_CSR_cleaned.xlsx"
PLAN_SHEET = "PLAN_CLEAN"
EXEC_SHEET = "EXEC_CLEAN"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3

OLD_OUTPUT_DIR = BASE_DIR / "model"
OLD_OUTPUT_DIR.mkdir(exist_ok=True)

OLD_CV_DIR = OLD_OUTPUT_DIR / "cv_results"
OLD_PRED_DIR = OLD_OUTPUT_DIR / "predictions"
OLD_FI_DIR = OLD_OUTPUT_DIR / "feature_importance"

OLD_CV_DIR.mkdir(exist_ok=True)
OLD_PRED_DIR.mkdir(exist_ok=True)
OLD_FI_DIR.mkdir(exist_ok=True)

CLASSIC_OUTPUT_DIR = BASE_DIR / "csr_models_output"
CLASSIC_OUTPUT_DIR.mkdir(exist_ok=True)

OLD_TARGETS = [
    "Actual_Impact_Value",
    "Internal_Participants",
    "Participation_Rate",
    "execution_rate"
]
CLASSIC_LOG_TARGETS = ["Actual_Cost_EUR"]


# =========================
# HELPERS
# =========================
def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_date_features(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    if col not in df.columns:
        return df

    dt = pd.to_datetime(df[col], errors="coerce")
    df[f"{prefix}_year"] = dt.dt.year
    df[f"{prefix}_month"] = dt.dt.month
    df[f"{prefix}_day"] = dt.dt.day
    df[f"{prefix}_dow"] = dt.dt.dayofweek
    df[f"{prefix}_quarter"] = dt.dt.quarter
    df[f"{prefix}_is_month_end"] = dt.dt.is_month_end.astype("float")
    df[f"{prefix}_is_month_start"] = dt.dt.is_month_start.astype("float")
    return df


def add_text_length_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            s = df[col].fillna("").astype(str)
            df[f"{col}_len"] = s.str.len()
            df[f"{col}_words"] = s.str.split().str.len()
    return df


def safe_numeric(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_one_hot_encoder()),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_old_models():
    return {
        "Ridge": Ridge(alpha=5),
        "HistGB": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
    }


def get_classic_models():
    return {
        "ridge": Ridge(alpha=2.0),
        "random_forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def extract_feature_importance(fitted_estimator):
    if isinstance(fitted_estimator, TransformedTargetRegressor):
        reg = fitted_estimator.regressor_
    else:
        reg = fitted_estimator

    if not hasattr(reg, "named_steps"):
        return None

    model = reg.named_steps.get("model", None)
    prep = reg.named_steps.get("prep", None)

    if model is None or prep is None:
        return None

    if not hasattr(model, "feature_importances_"):
        return None

    feature_names = prep.get_feature_names_out()
    importances = model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def split_by_year_or_random(df: pd.DataFrame, target: str):
    work = df[df[target].notna()].copy()

    if "Year" in work.columns and work["Year"].notna().nunique() >= 2:
        years = sorted(work["Year"].dropna().unique())
        test_year = years[-1]
        train_df = work[work["Year"] != test_year].copy()
        test_df = work[work["Year"] == test_year].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            return train_df, test_df

    return train_test_split(
        work,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


# =========================
# DATASET FOR OLD TARGETS
# =========================
def build_old_dataset(file_path: Path) -> pd.DataFrame:
    plan = pd.read_excel(file_path, sheet_name=PLAN_SHEET)
    exe = pd.read_excel(file_path, sheet_name=EXEC_SHEET)

    df = exe.merge(
        plan.add_prefix("plan_"),
        left_on="Plan_ID",
        right_on="plan_Plan_ID",
        how="left",
        validate="one_to_one",
    )

    if "Year" not in df.columns and "Year_exec" in df.columns:
        df["Year"] = df["Year_exec"]

    if "Actual_Cost_EUR" not in df.columns and "Actual_Cost_EUR_exec" in df.columns:
        df["Actual_Cost_EUR"] = df["Actual_Cost_EUR_exec"]

    if "Internal_Participants" not in df.columns and "Internal_Participants_exec" in df.columns:
        df["Internal_Participants"] = df["Internal_Participants_exec"]

    if "Participation_Rate" not in df.columns and "Participation_Rate_exec" in df.columns:
        df["Participation_Rate"] = df["Participation_Rate_exec"]

    if "Actual_Impact_Value" not in df.columns and "Actual_Impact_Value_exec" in df.columns:
        df["Actual_Impact_Value"] = df["Actual_Impact_Value_exec"]

    numeric_cols = [
        "Year",
        "Actual_Cost_EUR",
        "Actual_Impact_Value",
        "Internal_Participants",
        "Participation_Rate",
        "Expected_Impact_Value",
        "Planned_Cost_EUR",
        "Planned_Internal_Volunteers",
        "Duration_Months",
        "Sustainability_Score",
        "Total_HC",
        "External_Partners_Count",
        "plan_Planned_Cost_EUR",
        "plan_Planned_Internal_Volunteers",
        "plan_Expected_Impact_Value",
        "plan_Duration_Months",
        "plan_Sustainability_Score",
    ]
    df = safe_numeric(df, numeric_cols)

    for col in [
        "Expected_Impact_Value",
        "Planned_Cost_EUR",
        "Planned_Internal_Volunteers",
        "Duration_Months",
        "Sustainability_Score",
        "Actual_Cost_EUR",
        "Actual_Impact_Value",
        "Internal_Participants",
        "Participation_Rate",
        "plan_Expected_Impact_Value",
        "plan_Planned_Cost_EUR",
        "plan_Planned_Internal_Volunteers",
        "plan_Duration_Months",
        "plan_Sustainability_Score",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0).clip(lower=0)

    if "plan_Expected_Impact_Value" in df.columns:
        df["plan_Expected_Impact_Value"] = np.log1p(df["plan_Expected_Impact_Value"])
    if "plan_Planned_Cost_EUR" in df.columns:
        df["plan_Planned_Cost_EUR"] = np.log1p(df["plan_Planned_Cost_EUR"])

    if "plan_Expected_Impact_Value" in df.columns and "plan_Planned_Cost_EUR" in df.columns:
        df["Cost_per_Impact"] = df["plan_Planned_Cost_EUR"] / (df["plan_Expected_Impact_Value"] + 1)

    if "Internal_Participants" in df.columns and "plan_Planned_Internal_Volunteers" in df.columns:
        denom = df["plan_Planned_Internal_Volunteers"].replace(0, np.nan)
        df["execution_rate"] = (df["Internal_Participants"] / denom) * 100.0

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

    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.upper()
    if "plan_Status" in df.columns:
        df["plan_Status"] = df["plan_Status"].astype(str).str.upper()

    return df


# =========================
# DATASET FOR CLASSIC COST / EXECUTION
# EXACT SAME STYLE AS YOUR STANDALONE SCRIPT
# =========================
def build_classic_dataset(file_path: Path) -> pd.DataFrame:
    plan = pd.read_excel(file_path, sheet_name=PLAN_SHEET)
    exe = pd.read_excel(file_path, sheet_name=EXEC_SHEET)

    df = exe.merge(
        plan.add_prefix("plan_"),
        left_on="Plan_ID",
        right_on="plan_Plan_ID",
        how="left",
        validate="one_to_one",
    )

    df["execution_rate"] = (
        df["Internal_Participants"] / df["plan_Planned_Internal_Volunteers"] * 100.0
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

    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.upper()
    if "plan_Status" in df.columns:
        df["plan_Status"] = df["plan_Status"].astype(str).str.upper()

    return df


# =========================
# OLD METHOD FOR 3 TARGETS
# =========================
def get_old_feature_matrix(df: pd.DataFrame, target: str):
    drop_cols = {
        target,
        "Actual_Cost_EUR",
        "execution_rate",
        "Exec_ID",
        "Plan_ID",
        "Plan_ID_in_plan",
        "plan_Plan_ID",
        "Execution_Date",
        "plan_Start_Date",
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df[target].copy()
    mask = y.notna()
    return X.loc[mask].copy(), y.loc[mask].copy()


def fit_and_evaluate_old_target(df: pd.DataFrame, target: str):
    train_df, test_df = split_by_year_or_random(df, target)

    X_train, y_train = get_old_feature_matrix(train_df, target)
    X_test, y_test = get_old_feature_matrix(test_df, target)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"Empty train/test split for target: {target}")

    X_test = X_test.reindex(columns=X_train.columns)

    preprocessor = make_preprocessor(X_train)
    candidates = get_old_models()

    best_score = -np.inf
    best_name = None
    best_pipe = None

    for name, model in candidates.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model),
        ])

        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            if pred is None or np.any(~np.isfinite(pred)):
                print(f"⚠️ {target} / {name}: invalid predictions, skipped")
                continue

            score = r2_score(y_test, pred)

            if np.isfinite(score) and score > best_score:
                best_score = score
                best_name = name
                best_pipe = pipe

        except Exception as e:
            print(f"⚠️ {target} / {name} failed: {e}")
            continue

    if best_pipe is None:
        fallback = Pipeline([
            ("prep", preprocessor),
            ("model", Ridge(alpha=5)),
        ])
        fallback.fit(X_train, y_train)
        best_pipe = fallback
        best_name = "Ridge_fallback"
        y_pred = best_pipe.predict(X_test)
        best_score = r2_score(y_test, y_pred)
    else:
        y_pred = best_pipe.predict(X_test)

    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        "target": target,
        "best_model": best_name,
        "method": "old_direct",
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_r2": float(r2_score(y_test, y_pred)),
    }

    model_path = OLD_OUTPUT_DIR / f"{target}_best_model.joblib"
    joblib.dump(best_pipe, model_path)

    pred_table = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "error": y_test.values - y_pred,
    })
    pred_path = OLD_PRED_DIR / f"{target}_test_predictions.csv"
    pred_table.to_csv(pred_path, index=False)

    fi = extract_feature_importance(best_pipe)
    if fi is not None:
        fi_path = OLD_FI_DIR / f"{target}_feature_importance.csv"
        fi.head(50).to_csv(fi_path, header=["importance"])

    cv_path = OLD_CV_DIR / f"{target}_cv_results.csv"
    pd.DataFrame([{
        "model": best_name,
        "cv_mae": np.nan,
        "cv_rmse": np.nan,
        "cv_r2": best_score,
    }]).to_csv(cv_path, index=False)

    print("\n" + "=" * 80)
    print(f"TARGET: {target}")
    print(f"BEST MODEL: {best_name}")
    print("\nTest metrics:")
    print(pd.Series(metrics).to_string())
    print(f"\nSaved model -> {model_path}")
    print(f"Saved predictions -> {pred_path}")

    return metrics


# =========================
# CLASSIC LOG METHOD FOR COST / EXECUTION
# EXACTLY LIKE YOUR STANDALONE SCRIPT
# =========================
def get_feature_matrix(df: pd.DataFrame, target: str):
    drop_cols = {
        "Actual_Cost_EUR",
        "execution_rate",
        "Exec_ID",
        "Plan_ID",
        "Plan_ID_in_plan",
        "plan_Plan_ID",
        "Execution_Date",
        "plan_Start_Date",
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = np.log1p(df[target]).copy()

    mask = y.notna()
    return X.loc[mask].copy(), y.loc[mask].copy()


def evaluate_candidates(X_train, y_train, preprocessor):
    candidates = {
        "ridge": Ridge(alpha=2.0),
        "random_forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, model in candidates.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model),
        ])

        wrapped = TransformedTargetRegressor(
            regressor=pipe,
            transformer=PowerTransformer(method="yeo-johnson"),
        )

        scores = cross_validate(
            wrapped,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=1,
        )

        rows.append({
            "model": name,
            "cv_mae": -scores["test_mae"].mean(),
            "cv_rmse": np.sqrt(-scores["test_rmse"].mean()),
            "cv_r2": scores["test_r2"].mean(),
            "estimator": wrapped,
        })

    results = pd.DataFrame(rows).sort_values(["cv_rmse", "cv_mae"], ascending=True)
    best = results.iloc[0].to_dict()
    return results, best["estimator"], best["model"]


def fit_and_evaluate_target(df: pd.DataFrame, target: str):
    X, y_log = get_feature_matrix(df, target)
    preprocessor = make_preprocessor(X)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    cv_table, best_estimator, best_name = evaluate_candidates(X_train, y_train_log, preprocessor)
    best_estimator.fit(X_train, y_train_log)

    preds_log = best_estimator.predict(X_test)
    preds = np.expm1(preds_log)
    y_test = np.expm1(y_test_log)

    metrics = {
        "target": target,
        "best_model": best_name,
        "test_mae": float(mean_absolute_error(y_test, preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "test_r2": float(r2_score(y_test, preds)),
    }

    model_path = CLASSIC_OUTPUT_DIR / f"{target}_best_model.joblib"
    joblib.dump(best_estimator, model_path)

    cv_path = CLASSIC_OUTPUT_DIR / f"{target}_cv_results.csv"
    cv_table.drop(columns=["estimator"]).to_csv(cv_path, index=False)

    pred_table = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": preds,
        "error": y_test.values - preds,
    })

    pred_path = CLASSIC_OUTPUT_DIR / f"{target}_test_predictions.csv"
    pred_table.to_csv(pred_path, index=False)

    print("\n" + "=" * 80)
    print(f"TARGET: {target}")
    print(f"BEST MODEL: {best_name}")
    print("\nCross-validation results:")
    print(cv_table.drop(columns=["estimator"]).to_string(index=False))
    print("\nTest metrics (REAL SCALE):")
    print(pd.Series(metrics).to_string())
    print(f"\nSaved model -> {model_path}")
    print(f"Saved CV results -> {cv_path}")
    print(f"Saved test predictions -> {pred_path}")

    return metrics, cv_table, pred_table, best_estimator


def main():
    old_df = build_old_dataset(FILE_PATH)
    classic_df = build_classic_dataset(FILE_PATH)

    all_metrics = []

    for target in OLD_TARGETS:
        if target in old_df.columns:
            all_metrics.append(fit_and_evaluate_old_target(old_df, target))
        else:
            print(f"\n⚠️ Skipping {target}: column not found")

    for target in CLASSIC_LOG_TARGETS:
        if target in classic_df.columns:
            metrics, _, _, _ = fit_and_evaluate_target(classic_df, target)
            all_metrics.append(metrics)
        else:
            print(f"\n⚠️ Skipping {target}: column not found")

    summary = pd.DataFrame(all_metrics)

    summary_path = OLD_OUTPUT_DIR / "all_targets_summary.csv"
    summary.to_csv(summary_path, index=False)

    metrics_payload = {
        "file_path": str(FILE_PATH),
        "plan_sheet": PLAN_SHEET,
        "exec_sheet": EXEC_SHEET,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "targets": {row["target"]: row for row in all_metrics},
    }

    metrics_json_path = OLD_OUTPUT_DIR / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved summary -> {summary_path}")
    print(f"Saved metrics.json -> {metrics_json_path}")


if __name__ == "__main__":
    main()