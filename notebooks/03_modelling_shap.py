# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Modelling & SHAP
# XGBoost training, Optuna tuning, threshold optimization, SHAP explainability.

# %%
import sys
sys.path.insert(0, '..')

import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna
import xgboost as xgb
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.features import load_raw_data, prepare_features, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
from src.evaluate import tune_threshold, print_report

RANDOM_STATE = 42

# %%
# Load + split
df = load_raw_data()
X, y = prepare_features(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# %%
# Preprocessor
preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), CONTINUOUS_FEATURES),
    ('passthrough', 'passthrough', CATEGORICAL_FEATURES),
])

# %%
# Baseline XGBoost (no tuning)
baseline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')),
])
baseline.fit(X_train, y_train)
baseline_proba = baseline.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_auc_score
print(f"Baseline AUC: {roc_auc_score(y_test, baseline_proba):.4f}")

# %%
# Optuna tuning
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
    }
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBClassifier(**params)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return np.mean(cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc'))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# %%
# Final model
best_params = {**study.best_params, 'eval_metric': 'logloss', 'random_state': RANDOM_STATE}
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb.XGBClassifier(**best_params)),
])
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)[:, 1]
print(f"Tuned AUC: {roc_auc_score(y_test, y_proba):.4f}")

# %%
# Threshold tuning
threshold = tune_threshold(y_test, y_proba, min_recall=0.90)
y_pred = (y_proba >= threshold).astype(int)
print_report(y_test, y_pred, y_proba, threshold)

# %%
# SHAP global
xgb_model = pipeline.named_steps['model']
X_test_proc = pipeline.named_steps['preprocessor'].transform(X_test)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_proc)

shap.summary_plot(shap_values, X_test_proc, feature_names=ALL_FEATURES)

# %%
# SHAP waterfall — single patient
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test_proc[0],
    feature_names=ALL_FEATURES,
))

# %%
# Save
import os
os.makedirs('../models', exist_ok=True)
joblib.dump(pipeline, '../models/heart_disease_pipeline.pkl')
joblib.dump(explainer, '../models/shap_explainer.pkl')
with open('../models/threshold.txt', 'w') as f:
    f.write(str(threshold))
print("Models saved.")
