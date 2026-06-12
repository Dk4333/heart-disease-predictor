"""
features.py — Feature engineering for heart disease prediction.
All transformations are deterministic and applied consistently
across train, validation, and inference.
"""

import pandas as pd
import numpy as np


CONTINUOUS_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "age_thalach", "chol_per_age",
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang",
    "slope", "ca", "thal", "bp_category", "high_risk_cp",
]

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms and derived features on top of raw UCI columns.
    Input df must contain the 13 raw UCI features.
    Returns a new DataFrame with all original + engineered columns.
    """
    df = df.copy()

    # Interaction: age × max heart rate — captures age-adjusted cardiovascular capacity
    df["age_thalach"] = df["age"] * df["thalach"]

    # Ratio: cholesterol normalized by age — older patients tolerate higher cholesterol differently
    df["chol_per_age"] = df["chol"] / df["age"]

    # Ordinal bin: resting blood pressure → hypertension stage (JNC-8 thresholds)
    df["bp_category"] = pd.cut(
        df["trestbps"],
        bins=[0, 120, 130, 140, 300],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(int)

    # Binary flag: asymptomatic chest pain (cp == 0) is the highest-risk subtype
    df["high_risk_cp"] = (df["cp"] == 0).astype(int)

    return df


def load_raw_data() -> pd.DataFrame:
    """
    Fetch UCI Heart Disease dataset and return a clean DataFrame
    with binary target (1 = disease, 0 = no disease).
    Falls back to reading data/heart.csv if ucimlrepo is unavailable.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        df = pd.DataFrame(heart.data.features, columns=heart.data.feature_names)
        df["target"] = (heart.data.targets.values.ravel() > 0).astype(int)
    except Exception:
        df = pd.read_csv("data/heart.csv")
        if "target" not in df.columns:
            raise ValueError("CSV must contain a 'target' column.")
        df["target"] = (df["target"] > 0).astype(int)

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply feature engineering and split into X, y.
    """
    df = engineer_features(df)
    X = df[ALL_FEATURES]
    y = df["target"]
    return X, y
