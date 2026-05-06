import pandas as pd
import numpy as np

# ------------------------------------------
# EXPECTED FEATURES (must match training)
# ------------------------------------------
NUMERIC_FEATURES = [
    "Year",
    "Planned_Cost_EUR",
    "Planned_Internal_Volunteers",
    "Expected_Impact_Value",
    "Duration_Months",
    "Sustainability_Score"
]

CATEGORICAL_FEATURES = [
    "Entity",
    "Country",
    "Plant",
    "Category",
    "Activity_Type",
    "Organization",
    "External_Entity",
    "Contract_Type",
    "Periodicity",
    "Impact_Unit"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ------------------------------------------
# MAIN PREPROCESS FUNCTION (FOR STREAMLIT)
# ------------------------------------------
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --------------------------------------
    # 1. Keep only required columns
    # --------------------------------------
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan  # create missing columns

    df = df[ALL_FEATURES]

    # --------------------------------------
    # 2. Basic numeric cleaning
    # --------------------------------------
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------
    # 3. Basic categorical cleaning
    # --------------------------------------
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("Unknown").str.strip()

    # --------------------------------------
    # 4. Handle missing values (light)
    # --------------------------------------
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("Unknown")

    return df