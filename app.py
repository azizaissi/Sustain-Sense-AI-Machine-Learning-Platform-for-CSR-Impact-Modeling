import io
import json
import pickle
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

try:
    import shap
except Exception:
    shap = None

try:
    from scipy import sparse
except Exception:
    sparse = None

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CSR Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME / STYLE
# =========================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #effdf4;
            --bg-2: #e7f7ff;
            --bg-3: #f4fbff;
            --card: rgba(255,255,255,0.78);
            --card-border: rgba(71, 123, 156, 0.16);
            --text: #0f172a;
            --muted: rgba(15, 23, 42, 0.68);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(46, 204, 113, 0.18), transparent 32%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 30%),
                radial-gradient(circle at bottom left, rgba(167, 243, 208, 0.16), transparent 34%),
                linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1600px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(236,247,255,0.92));
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #0f172a;
        }

        .brand-shell {
            padding: 1.35rem 1.35rem 1.15rem 1.35rem;
            border: 1px solid var(--card-border);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,255,255,0.60));
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.10);
            margin-bottom: 1rem;
        }

        .brand-title {
            font-size: 2.55rem;
            font-weight: 900;
            line-height: 1.0;
            letter-spacing: -0.05em;
            margin: 0;
            color: #0f172a;
        }

        .brand-subtitle {
            color: var(--muted);
            margin-top: 0.55rem;
            font-size: 0.98rem;
            line-height: 1.5;
        }

        .glass-card, .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.64));
            border: 1px solid var(--card-border);
            border-radius: 24px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            color: #0f172a;
        }

        .glass-card {
            padding: 1.1rem 1.15rem;
        }

        .metric-card {
            padding: 1rem 1.05rem;
            min-height: 105px;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 800;
            margin: 0 0 0.75rem 0;
            letter-spacing: -0.02em;
            color: #0f172a;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .chip-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.7rem;
        }

        .chip {
            display: inline-block;
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            background: rgba(47, 128, 237, 0.11);
            border: 1px solid rgba(47, 128, 237, 0.18);
            color: #0f172a;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .divider-line {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(15, 23, 42, 0.16), transparent);
            margin: 1rem 0 1.1rem;
        }

        .callout {
            border-radius: 18px;
            border: 1px solid rgba(47, 128, 237, 0.22);
            background: rgba(47, 128, 237, 0.08);
            padding: 0.9rem 1rem;
            color: #0f172a;
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 0.72rem 1rem;
            font-weight: 800;
            background: linear-gradient(135deg, rgba(47, 128, 237, 0.95), rgba(43, 182, 115, 0.90));
            color: white;
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.10);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.14);
        }

        .footer-note {
            color: rgba(15, 23, 42, 0.55);
            font-size: 0.84rem;
            text-align: center;
            padding-top: 0.5rem;
        }

        .stMarkdown, .stCaption, .stInfo, .stWarning, .stError {
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
IMPACT_MODEL_DIR = BASE_DIR / "model"
IMPACT_CLASSIC_DIR = BASE_DIR / "csr_models_output"
IMPACT_METRICS_PATH = IMPACT_MODEL_DIR / "metrics.json"

IMPACT_TARGET_MODEL_PATHS = {
    "Actual_Impact_Value": IMPACT_MODEL_DIR / "Actual_Impact_Value_best_model.joblib",
    "Internal_Participants": IMPACT_MODEL_DIR / "Internal_Participants_best_model.joblib",
    "Participation_Rate": IMPACT_MODEL_DIR / "Participation_Rate_best_model.joblib",
    "execution_rate": IMPACT_MODEL_DIR / "execution_rate_best_model.joblib",
    "Actual_Cost_EUR": IMPACT_CLASSIC_DIR / "Actual_Cost_EUR_best_model.joblib",
}

STATUS_MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
STATUS_RESULTS_PATH = BASE_DIR / "models" / "evaluation_results.json"
PLAN_SHEET = "PLAN_CLEAN"
EXEC_SHEET = "EXEC_CLEAN"

# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "home"


def go_home():
    st.session_state.page = "home"


def go_impact():
    st.session_state.page = "impact"


def go_status():
    st.session_state.page = "status"


def go_shap():
    st.session_state.page = "shap"


def go_about():
    st.session_state.page = "about"


# =========================================================
# HELPERS
# =========================================================
def make_excel_file(uploaded_file):
    return pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()))


def add_date_features(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
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


def add_text_length_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            s = df[col].fillna("").astype(str)
            df[f"{col}_len"] = s.str.len()
            df[f"{col}_words"] = s.str.split().str.len()
    return df


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def fmt_num(value, digits=2):
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "—"


def render_top_banner(title: str, subtitle: str, chips: Optional[List[str]] = None):
    st.markdown(
        f"""
        <div class="brand-shell">
            <div class="brand-title">{title}</div>
            <div class="brand-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if chips:
        st.markdown(
            '<div class="chip-row">' + ''.join(f'<span class="chip">{c}</span>' for c in chips) + '</div>',
            unsafe_allow_html=True,
        )


def render_section_title(title: str, subtitle: str = ""):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='small-muted'>{subtitle}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: Optional[str] = None):
    delta_html = f"<div class='small-muted' style='margin-top:0.35rem;'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-muted">{label}</div>
            <div style="font-size:1.65rem; font-weight:900; margin-top:0.22rem; color:#0f172a;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_metric_bars(df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None):
    fig = px.bar(df, x=x, y=y, title=title, text_auto=True, color=color)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, color="#0f172a"),
        title=dict(x=0.02),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.08)")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_impact_models():
    models = {}
    missing = []
    for target, path in IMPACT_TARGET_MODEL_PATHS.items():
        if path.exists():
            try:
                models[target] = joblib.load(path)
            except Exception as e:
                missing.append(f"{target}: failed to load ({e})")
        else:
            missing.append(f"{target}: file not found")
    return models, missing


@st.cache_data
def load_impact_metrics():
    if not IMPACT_METRICS_PATH.exists():
        return None
    try:
        with open(IMPACT_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_loaded_preprocessor(model):
    if isinstance(model, TransformedTargetRegressor):
        return model.regressor_.named_steps.get("prep")
    if isinstance(model, Pipeline):
        return model.named_steps.get("prep")
    return None


def get_fitted_estimator(model):
    if isinstance(model, TransformedTargetRegressor):
        reg = model.regressor_
        if hasattr(reg, "named_steps"):
            return reg.named_steps.get("model", reg)
        return reg
    if isinstance(model, Pipeline):
        if hasattr(model, "named_steps"):
            return model.named_steps.get("model", model)
        return model
    return model


def align_input_to_model(df: pd.DataFrame, model) -> pd.DataFrame:
    prep = get_loaded_preprocessor(model)
    if prep is None or not hasattr(prep, "feature_names_in_"):
        return df.copy()
    expected_cols = list(prep.feature_names_in_)
    X = df.copy()
    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan
    return X[expected_cols]


@st.cache_resource
def load_status_model():
    if not STATUS_MODEL_PATH.exists():
        return None
    with open(STATUS_MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_status_results():
    if not STATUS_RESULTS_PATH.exists():
        return None
    try:
        with open(STATUS_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =========================================================
# READ WORKBOOKS
# =========================================================
def read_impact_base(uploaded_file) -> pd.DataFrame:
    xls = make_excel_file(uploaded_file)
    sheets = xls.sheet_names
    if PLAN_SHEET in sheets and EXEC_SHEET in sheets:
        plan = pd.read_excel(xls, sheet_name=PLAN_SHEET)
        exe = pd.read_excel(xls, sheet_name=EXEC_SHEET)
        if "Plan_ID" in exe.columns and "Plan_ID" in plan.columns:
            try:
                return exe.merge(plan.add_prefix("plan_"), left_on="Plan_ID", right_on="plan_Plan_ID", how="left")
            except Exception:
                return exe.merge(plan.add_prefix("plan_"), left_on="Plan_ID", right_on="plan_Plan_ID", how="left", suffixes=("", "_dup"))
        return exe if len(exe.columns) >= len(plan.columns) else plan

    preferred = [PLAN_SHEET, EXEC_SHEET, "Plan", "plan", "Sheet1"]
    for sheet in preferred:
        if sheet in sheets:
            return pd.read_excel(xls, sheet_name=sheet)
    return pd.read_excel(xls, sheet_name=sheets[0])


def read_status_sheet(uploaded_file) -> pd.DataFrame:
    xls = make_excel_file(uploaded_file)
    sheets = xls.sheet_names
    preferred = [PLAN_SHEET, "Plan", "plan", "Sheet1", EXEC_SHEET]
    for sheet in preferred:
        if sheet in sheets:
            return pd.read_excel(xls, sheet_name=sheet)
    return pd.read_excel(xls, sheet_name=sheets[0])


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def build_old_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    aliases = {
        "Year": ["Year_exec"],
        "Actual_Cost_EUR": ["Actual_Cost_EUR_exec"],
        "Internal_Participants": ["Internal_Participants_exec"],
        "Participation_Rate": ["Participation_Rate_exec"],
        "Actual_Impact_Value": ["Actual_Impact_Value_exec"],
    }
    for target_col, candidates in aliases.items():
        if target_col not in df.columns:
            for c in candidates:
                if c in df.columns:
                    df[target_col] = df[c]
                    break

    numeric_cols = [
        "Year", "Actual_Cost_EUR", "Actual_Impact_Value", "Internal_Participants", "Participation_Rate",
        "Expected_Impact_Value", "Planned_Cost_EUR", "Planned_Internal_Volunteers", "Duration_Months",
        "Sustainability_Score", "Total_HC", "External_Partners_Count", "plan_Planned_Cost_EUR",
        "plan_Planned_Internal_Volunteers", "plan_Expected_Impact_Value", "plan_Duration_Months",
        "plan_Sustainability_Score",
    ]
    df = safe_numeric(df, numeric_cols)

    for col in [
        "Expected_Impact_Value", "Planned_Cost_EUR", "Planned_Internal_Volunteers", "Duration_Months",
        "Sustainability_Score", "Actual_Cost_EUR", "Actual_Impact_Value", "Internal_Participants",
        "Participation_Rate", "plan_Expected_Impact_Value", "plan_Planned_Cost_EUR",
        "plan_Planned_Internal_Volunteers", "plan_Duration_Months", "plan_Sustainability_Score",
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
    df = add_text_length_features(df, ["Organizer", "External_Partners", "plan_Activity_Description", "plan_Organization", "plan_External_Entity"])

    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.upper()
    if "plan_Status" in df.columns:
        df["plan_Status"] = df["plan_Status"].astype(str).str.upper()
    return df


def build_classic_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Internal_Participants" in df.columns and "plan_Planned_Internal_Volunteers" in df.columns:
        planned = pd.to_numeric(df["plan_Planned_Internal_Volunteers"], errors="coerce").replace(0, np.nan)
        actual = pd.to_numeric(df["Internal_Participants"], errors="coerce")
        df["execution_rate"] = (actual / planned) * 100.0

    df = add_date_features(df, "Execution_Date", "exec_date")
    df = add_date_features(df, "plan_Start_Date", "plan_start")
    df = add_text_length_features(df, ["Organizer", "External_Partners", "plan_Activity_Description", "plan_Organization", "plan_External_Entity"])

    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.upper()
    if "plan_Status" in df.columns:
        df["plan_Status"] = df["plan_Status"].astype(str).str.upper()
    return df


def status_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Status" in df.columns:
        df["Status"] = normalize_series(df["Status"])
    if "Start_Date" in df.columns:
        df["Start_Date"] = pd.to_datetime(df["Start_Date"], errors="coerce")
        df["Start_Year"] = df["Start_Date"].dt.year
        df["Start_Month"] = df["Start_Date"].dt.month
        df["Start_Quarter"] = df["Start_Date"].dt.quarter
    return df


# =========================================================
# PREDICTIONS
# =========================================================
def predict_impact_with_model(df: pd.DataFrame, model, target_name: str) -> np.ndarray:
    X = align_input_to_model(df, model)
    preds = model.predict(X)
    if target_name == "Actual_Cost_EUR":
        preds = np.expm1(preds)
    return np.asarray(preds)


def status_align_features(df: pd.DataFrame, model) -> pd.DataFrame:
    prep = get_loaded_preprocessor(model)
    if prep is None or not hasattr(prep, "feature_names_in_"):
        return df.copy()
    expected_cols = list(prep.feature_names_in_)
    X = df.copy()
    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan
    return X[expected_cols]


def status_risk_mode(proba, t_low, t_high):
    labels = []
    for p in proba:
        if p <= t_low:
            labels.append("HIGH RISK (NOT executed)")
        elif p >= t_high:
            labels.append("SAFE (executed)")
        else:
            labels.append("UNCERTAIN")
    return np.array(labels)


def status_execution_mode(proba, threshold):
    return (proba >= threshold).astype(int)


# =========================================================
# METRICS / SHAP
# =========================================================
def impact_metrics_to_df(metrics_data):
    if not metrics_data or "targets" not in metrics_data:
        return pd.DataFrame()
    rows = []
    for target, info in metrics_data["targets"].items():
        row = {"Target": target}
        if isinstance(info, dict):
            for key in ["best_model", "method", "test_mae", "test_rmse", "test_r2"]:
                if key in info:
                    row[key] = info[key]
        rows.append(row)
    return pd.DataFrame(rows)


def status_metrics_to_df(results):
    if not results:
        return pd.DataFrame()
    mode1 = results.get("mode_1_execution", {})
    mode2 = results.get("mode_2_risk", {})
    return pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Execution Mode": [mode1.get("accuracy"), mode1.get("precision"), mode1.get("recall"), mode1.get("f1")],
            "Risk Mode": [mode2.get("accuracy"), mode2.get("precision"), mode2.get("recall"), mode2.get("f1")],
        }
    )


def get_feature_importance(model):
    if isinstance(model, TransformedTargetRegressor):
        reg = model.regressor_
    else:
        reg = model
    if not hasattr(reg, "named_steps"):
        return None
    prep = reg.named_steps.get("prep")
    fitted_model = reg.named_steps.get("model")
    if prep is None or fitted_model is None or not hasattr(fitted_model, "feature_importances_"):
        return None
    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        return None
    return pd.Series(fitted_model.feature_importances_, index=feature_names).sort_values(ascending=False)


def _to_dense_array(X):
    if sparse is not None and sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def transform_for_shap(model, X_raw: pd.DataFrame) -> pd.DataFrame:
    prep = get_loaded_preprocessor(model)
    if prep is None:
        return X_raw.copy()
    Xt = prep.transform(X_raw)
    Xt = _to_dense_array(Xt)
    try:
        feature_names = list(prep.get_feature_names_out())
    except Exception:
        feature_names = list(X_raw.columns)
    if Xt.ndim == 1:
        Xt = Xt.reshape(-1, 1)
    return pd.DataFrame(Xt, columns=feature_names, index=X_raw.index)


def build_shap_explainer(model, X_background: pd.DataFrame):
    if shap is None:
        return None, None
    estimator = get_fitted_estimator(model)
    X_bg = transform_for_shap(model, X_background)

    try:
        if hasattr(estimator, "feature_importances_") or "Tree" in estimator.__class__.__name__:
            return shap.TreeExplainer(estimator), X_bg
    except Exception:
        pass
    try:
        return shap.Explainer(estimator.predict, X_bg), X_bg
    except Exception:
        pass
    try:
        masker = shap.maskers.Independent(X_bg)
        return shap.Explainer(estimator.predict, masker), X_bg
    except Exception:
        return None, None


# =========================================================
# HOME
# =========================================================
def render_home():
    render_top_banner(
        "CSR Intelligence Platform",
        "Executive-grade CSR analytics for impact prediction, execution risk, and explainable AI.",
        ["Impact Forecasting", "Execution Risk", "SHAP Explainability", "Excel Upload", "Export Ready"],
    )

    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:1rem;">
            <div style="font-size:0.92rem; font-weight:700; color:#2f80ed; letter-spacing:0.06em; text-transform:uppercase;">Executive Dashboard</div>
            <div style="font-size:2.2rem; font-weight:900; color:#0f172a; line-height:1.05; margin-top:0.25rem;">Turn CSR data into decisions that are easy to read, explain, and present.</div>
            <div class="small-muted" style="margin-top:0.65rem; max-width: 920px;">Predict outcomes, compare model quality, inspect risk, and explain each prediction with SHAP in a clean interface built for high-trust reporting.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Modules", "3", "Impact, Status, SHAP")
    with c2:
        metric_card("Input", "Excel", "Multi-sheet supported")
    with c3:
        metric_card("Output", "CSV", "Download predictions")

    render_section_title("Choose a workspace", "Use the buttons below or the sidebar to move between dashboards.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="glass-card"><h3 style="margin-top:0; color:#0f172a;">📊 CSR Impact Intelligence</h3><p class="small-muted">Predict impact outcomes, participants, participation rate, execution rate, and actual cost.</p></div>""", unsafe_allow_html=True)
        st.button("Open Impact Dashboard", use_container_width=True, on_click=go_impact)
    with c2:
        st.markdown("""<div class="glass-card"><h3 style="margin-top:0; color:#0f172a;">🟢 Plan Status Predictor</h3><p class="small-muted">Predict whether a CSR plan will be executed and show confidence plus risk labels.</p></div>""", unsafe_allow_html=True)
        st.button("Open Status Predictor", use_container_width=True, on_click=go_status)
    with c3:
        st.markdown("""<div class="glass-card"><h3 style="margin-top:0; color:#0f172a;">🧠 SHAP Explanation</h3><p class="small-muted">See what drives predictions. Built for explainability and executive review.</p></div>""", unsafe_allow_html=True)
        st.button("Open SHAP Section", use_container_width=True, on_click=go_shap)

    st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)
    st.markdown("""<div class="callout"><strong>Tip:</strong> For SHAP and predictions, keep the workbook structure aligned with training. This version works with the Excel structure you shared and explains the transformed features when possible.</div>""", unsafe_allow_html=True)


# =========================================================
# IMPACT DASHBOARD
# =========================================================
def render_impact_dashboard():
    render_top_banner(
        "📊 CSR Impact Intelligence Dashboard",
        "Predict impact outcomes, participants, participation rate, execution rate, and actual cost with executive-style analytics.",
        ["Regression Models", "Model Metrics", "Top Projects", "Feature Importance", "CSV Export"],
    )

    models, missing = load_impact_models()
    metrics_data = load_impact_metrics()

    if missing:
        st.warning("Some impact models are missing or failed to load:\n" + "\n".join(f"- {m}" for m in missing))
    if not models:
        st.error("No impact models available. Run the training script first.")
        return

    metrics_df = impact_metrics_to_df(metrics_data)
    render_section_title("Model performance", "Compare the best model and metrics for every target.")
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            if "test_r2" in metrics_df.columns:
                metric_card("Best R²", fmt_num(metrics_df["test_r2"].max(), 3), "Across loaded targets")
        with m2:
            if "test_mae" in metrics_df.columns:
                metric_card("Lowest MAE", fmt_num(metrics_df["test_mae"].min(), 3), "Test set error")
        with m3:
            if "test_rmse" in metrics_df.columns:
                metric_card("Lowest RMSE", fmt_num(metrics_df["test_rmse"].min(), 3), "Test set error")
        with m4:
            metric_card("Targets", str(len(metrics_df)), "Regression outputs loaded")
        c1, c2 = st.columns(2)
        with c1:
            if "test_r2" in metrics_df.columns:
                plot_metric_bars(metrics_df, x="Target", y="test_r2", title="Test R² by target")
        with c2:
            if "test_mae" in metrics_df.columns:
                plot_metric_bars(metrics_df, x="Target", y="test_mae", title="Test MAE by target")
    else:
        st.info("No impact metrics found in metrics.json.")

    uploaded_file = st.file_uploader("Upload CSR workbook for impact prediction", type=["xlsx"], key="impact_upload")
    if not uploaded_file:
        return

    try:
        raw_df = read_impact_base(uploaded_file)
    except Exception as e:
        st.error(f"Could not read workbook: {e}")
        return

    render_section_title("Workbook preview", "The app reads your Excel file and prepares the data automatically.")
    st.dataframe(raw_df.head(25), use_container_width=True)

    old_df = build_old_dataset(raw_df)
    classic_df = build_classic_dataset(raw_df)

    render_section_title("Predictions", "Each target is predicted with its trained model.")
    predictions = raw_df.copy()
    for target, model in models.items():
        try:
            source_df = classic_df if target == "Actual_Cost_EUR" else old_df
            predictions[f"Pred_{target}"] = predict_impact_with_model(source_df, model, target)
        except Exception as e:
            st.warning(f"Prediction error for {target}: {e}")

    if any(c.startswith("Pred_") for c in predictions.columns):
        st.dataframe(predictions, use_container_width=True, hide_index=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("Projects", str(len(predictions)), "Uploaded rows")
        with k2:
            if "Pred_Actual_Impact_Value" in predictions.columns:
                metric_card("Avg Impact", fmt_num(predictions["Pred_Actual_Impact_Value"].mean(), 2), "Predicted mean")
        with k3:
            if "Pred_Internal_Participants" in predictions.columns:
                metric_card("Avg Participants", fmt_num(predictions["Pred_Internal_Participants"].mean(), 2), "Predicted mean")
        with k4:
            if "Pred_Actual_Cost_EUR" in predictions.columns:
                metric_card("Avg Cost (EUR)", fmt_num(predictions["Pred_Actual_Cost_EUR"].mean(), 2), "Predicted mean")

        if "Pred_Actual_Impact_Value" in predictions.columns:
            predictions["Impact_Category"] = pd.cut(
                predictions["Pred_Actual_Impact_Value"],
                bins=[-np.inf, 10000, 20000, np.inf],
                labels=["Low", "Medium", "High"],
            )

        c1, c2 = st.columns(2)
        with c1:
            if "Pred_Actual_Impact_Value" in predictions.columns:
                fig = px.histogram(predictions, x="Pred_Actual_Impact_Value", nbins=20, title="Predicted impact distribution")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "Impact_Category" in predictions.columns:
                cat_df = predictions["Impact_Category"].value_counts().reset_index()
                cat_df.columns = ["Category", "Count"]
                fig = px.bar(cat_df, x="Category", y="Count", title="Impact categories")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        render_section_title("Top projects", "Ranked by predicted impact.")
        if "Pred_Actual_Impact_Value" in predictions.columns:
            st.dataframe(predictions.sort_values(by="Pred_Actual_Impact_Value", ascending=False).head(10), use_container_width=True, hide_index=True)

        render_section_title("Feature importance", "Shown when the model exposes tree-based importance.")
        fi = get_feature_importance(models.get("Actual_Impact_Value"))
        if fi is not None and len(fi) > 0:
            fi_df = fi.head(15).reset_index()
            fi_df.columns = ["Feature", "Importance"]
            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Top 15 feature importances")
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is not available for the impact model.")

        st.download_button(
            "⬇ Download Impact Predictions",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="csr_impact_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning("No predictions were produced. Check the input columns and the trained models.")


# =========================================================
# STATUS DASHBOARD
# =========================================================
def render_status_dashboard():
    render_top_banner(
        "🟢 Plan Status Predictor",
        "Predict whether a CSR plan is likely to be executed and classify the execution risk with a decision-ready view.",
        ["Classification", "Probability", "Risk Labels", "Thresholds", "CSV Export"],
    )

    model_data = load_status_model()
    results = load_status_results()
    if model_data is None:
        st.error("Status model not found. Run the status training script first.")
        return

    model = model_data["model"]
    threshold = model_data["threshold"]
    t_low = model_data["t_low"]
    t_high = model_data["t_high"]

    render_section_title("Decision thresholds", "These boundaries drive the execution and risk outputs.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Execution threshold", f"{threshold:.3f}", "Binary decision boundary")
    with c2:
        metric_card("Risk LOW boundary", f"{t_low:.3f}", "High-risk cutoff")
    with c3:
        metric_card("Risk HIGH boundary", f"{t_high:.3f}", "Safe cutoff")

    render_section_title("Model performance", "Compare the two evaluation modes.")
    metrics_df = status_metrics_to_df(results)
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            plot_df = metrics_df.melt(id_vars="Metric", var_name="Mode", value_name="Score")
            fig = px.bar(plot_df, x="Metric", y="Score", color="Mode", barmode="group", title="Execution vs risk mode")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No evaluation results found.")

    uploaded_file = st.file_uploader("Upload Excel file for status prediction", type=["xlsx"], key="status_upload")
    if not uploaded_file:
        return

    try:
        df_raw = read_status_sheet(uploaded_file)
    except Exception as e:
        st.error(f"Could not read workbook: {e}")
        return

    render_section_title("Workbook preview", "The sheet is loaded before feature engineering.")
    st.dataframe(df_raw.head(25), use_container_width=True)

    try:
        df = status_engineer_features(df_raw)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return

    filter_finished = False
    if "Status" in df.columns:
        filter_finished = st.checkbox("Filter to Finished plans only", value=True, key="status_finished_filter")
    if filter_finished and "Status" in df.columns:
        df = df[df["Status"].astype(str).str.upper() == "FINISHED"].copy()
    if df.empty:
        st.warning("No rows left after filtering.")
        return

    try:
        X = status_align_features(df, model)
        proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error("Model prediction failed — check feature mismatch.")
        st.write(str(e))
        return

    exec_preds = status_execution_mode(proba, threshold)
    risk_labels = status_risk_mode(proba, t_low, t_high)
    df["Execution_Probability"] = proba
    df["Execution_Prediction"] = exec_preds
    df["Risk_Label"] = risk_labels

    render_section_title("Prediction results", "A clean decision table and risk view.")
    k1, k2, k3 = st.columns(3)
    with k1:
        metric_card("Average probability", fmt_num(np.mean(proba), 3), "Overall confidence")
    with k2:
        metric_card("Executed", str(int(exec_preds.sum())), "Predicted positive rows")
    with k3:
        metric_card("At risk", str(int((risk_labels == "HIGH RISK (NOT executed)").sum())), "High-risk rows")

    c1, c2 = st.columns(2)
    with c1:
        execution_table = df[[c for c in ["Plan_ID", "Execution_Probability", "Execution_Prediction"] if c in df.columns]].copy()
        execution_table.rename(columns={"Execution_Prediction": "Execution_Label"}, inplace=True)
        st.dataframe(execution_table, use_container_width=True, hide_index=True)
    with c2:
        risk_table = df[[c for c in ["Plan_ID", "Execution_Probability", "Risk_Label"] if c in df.columns]].copy()
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Execution_Probability", nbins=20, title="Execution probability distribution")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        risk_counts = pd.Series(risk_labels).value_counts().reset_index()
        risk_counts.columns = ["Risk Label", "Count"]
        fig = px.bar(risk_counts, x="Risk Label", y="Count", title="Risk label distribution")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    render_section_title("Final output preview", "Prepared for download and sharing.")
    show_cols = [c for c in ["Plan_ID", "Execution_Probability", "Execution_Prediction", "Risk_Label"] if c in df.columns]
    st.dataframe(df[show_cols].head(20), use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download Status Predictions",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="csr_status_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# SHAP DASHBOARD
# =========================================================
def render_shap_dashboard():
    render_top_banner(
        "🧠 SHAP Explainability",
        "See which features push a prediction up or down and make the model easier to explain to stakeholders.",
        ["Local Explanation", "Global Summary", "Decision Transparency", "Model Debugging"],
    )

    if shap is None:
        st.error("The SHAP package is not available in this environment.")
        st.info("Install it with: pip install shap")
        return

    models, _ = load_impact_models()
    if not models:
        st.error("No impact models are available for SHAP explanations.")
        return

    render_section_title("Choose a model", "SHAP works best on one model at a time.")
    target = st.selectbox("Explain this target", options=list(models.keys()), index=0)
    uploaded_file = st.file_uploader("Upload a CSR workbook for SHAP explanation", type=["xlsx"], key="shap_upload")
    if not uploaded_file:
        st.info("Upload a workbook to generate SHAP explanations.")
        return

    try:
        raw_df = read_impact_base(uploaded_file)
    except Exception as e:
        st.error(f"Could not read workbook: {e}")
        return

    source_df = build_classic_dataset(raw_df) if target == "Actual_Cost_EUR" else build_old_dataset(raw_df)
    model = models[target]
    X_raw = align_input_to_model(source_df, model)
    if X_raw.empty:
        st.warning("No aligned features available for SHAP.")
        return

    render_section_title("Input preview", "These are the features that will be explained.")
    st.dataframe(X_raw.head(20), use_container_width=True, hide_index=True)

    X_bg_raw = X_raw.sample(min(len(X_raw), 200), random_state=42) if len(X_raw) > 200 else X_raw.copy()
    explainer, X_bg = build_shap_explainer(model, X_bg_raw)
    if explainer is None or X_bg is None:
        st.warning("Could not build a SHAP explainer for this model.")
        st.info("The app now tries the transformed feature space first. Some models still need a custom SHAP setup.")
        return

    render_section_title("Global importance", "Average absolute SHAP values show the main drivers.")
    try:
        shap_values = explainer(X_bg)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        shap_importance = pd.DataFrame({"Feature": X_bg.columns, "Mean |SHAP|": mean_abs}).sort_values("Mean |SHAP|", ascending=False).head(15)
        fig = px.bar(shap_importance, x="Mean |SHAP|", y="Feature", orientation="h", title=f"Top SHAP drivers for {target}")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        return

    render_section_title("Local explanation", "Inspect a single row and understand its prediction.")
    row_idx = st.selectbox("Select a row to explain", options=list(range(len(X_bg))), index=0)
    row = X_bg.iloc[[row_idx]]
    try:
        row_exp = explainer(row)
        contrib = pd.DataFrame({"Feature": row.columns, "SHAP": row_exp.values[0], "Value": row.iloc[0].values}).sort_values(by="SHAP", key=lambda s: np.abs(s), ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(contrib.head(15), use_container_width=True, hide_index=True)
        with c2:
            top_contrib = contrib.head(15).iloc[::-1]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_contrib["SHAP"], y=top_contrib["Feature"], orientation="h"))
            fig.update_layout(title="Top local SHAP contributions", height=520, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#0f172a"))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Positive SHAP values push the prediction higher; negative values push it lower.")
    except Exception as e:
        st.warning(f"Could not build the local explanation: {e}")


# =========================================================
# ABOUT
# =========================================================
def render_about_page():
    render_top_banner(
        "About the platform",
        "Built for CSR analytics with a focus on clarity, explainability, and executive presentation quality.",
        ["Modern UI", "Reusable Pipelines", "Model Loading", "Explainable AI"],
    )
    st.markdown(
        """
        <div class="glass-card">
        <h3 style="margin-top:0; color:#0f172a;">What this app does</h3>
        <p class="small-muted">The platform combines CSR impact forecasting, plan execution prediction, and SHAP explainability in one dashboard. It is designed to support business review meetings, monitoring, and data-driven decision making.</p>
        <div class="divider-line"></div>
        <p class="small-muted">Keep the model artifacts in the expected folders, upload the Excel workbook, and the app will handle feature engineering, alignment, predictions, and exports.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.info("You can extend this app later with trend analysis, scenario simulation, and PDF report export.")


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="padding:0.25rem 0 0.75rem 0;">
            <div style="font-size:1.25rem; font-weight:900; line-height:1.1; color:#0f172a;">CSR Intelligence</div>
            <div style="color:rgba(15,23,42,0.70); font-size:0.88rem; margin-top:0.25rem;">Executive analytics suite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Impact Dashboard", "Status Predictor", "SHAP Explanation", "About"],
        index=["home", "impact", "status", "shap", "about"].index(st.session_state.page) if st.session_state.page in ["home", "impact", "status", "shap", "about"] else 0,
    )
    {
        "Home": go_home,
        "Impact Dashboard": go_impact,
        "Status Predictor": go_status,
        "SHAP Explanation": go_shap,
        "About": go_about,
    }[page]()
    st.sidebar.markdown("---")
    st.sidebar.caption("Suggested workflow")
    st.sidebar.caption("1. Load models")
    st.sidebar.caption("2. Upload workbook")
    st.sidebar.caption("3. Review analytics")
    st.sidebar.caption("4. Download results")


# =========================================================
# APP RUNTIME
# =========================================================
render_sidebar()
st.markdown("---")

if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "impact":
    render_impact_dashboard()
elif st.session_state.page == "status":
    render_status_dashboard()
elif st.session_state.page == "shap":
    render_shap_dashboard()
elif st.session_state.page == "about":
    render_about_page()
else:
    render_home()

st.markdown("<div class='footer-note'>CSR Intelligence Platform • Built for polished executive analytics</div>", unsafe_allow_html=True)
