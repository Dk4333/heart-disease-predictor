"""
train.py — Full training pipeline for heart disease prediction.

Run:
    python src/train.py

Outputs:
    models/heart_disease_pipeline.pkl
    models/shap_explainer.pkl
    models/threshold.txt
    outputs/shap_summary.png
    outputs/shap_bar.png
    outputs/confusion_matrix.png
    outputs/roc_curve.png
    outputs/precision_recall.png
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

# Add project root to path so src imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import load_raw_data, prepare_features, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from src.evaluate import tune_threshold, print_report, plot_confusion_matrix, plot_roc_curve, plot_precision_recall

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_TRIALS = 50
MIN_RECALL = 0.90
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── 1. Load & prepare data ────────────────────────────────────────────────────

print("Loading data...")
df = load_raw_data()
print(f"  Shape: {df.shape} | Target distribution:\n{df['target'].value_counts()}\n")

X, y = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")


# ── 2. Preprocessor ───────────────────────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), CONTINUOUS_FEATURES),
        ("passthrough", "passthrough", CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)


# ── 3. Optuna hyperparameter tuning ──────────────────────────────────────────

print(f"Running Optuna search ({N_TRIALS} trials)...")

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBClassifier(**params)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f"  Best CV AUC : {study.best_value:.4f}")
print(f"  Best params : {study.best_params}\n")


# ── 4. Train final model ──────────────────────────────────────────────────────

best_params = study.best_params
best_params.update({"eval_metric": "logloss", "random_state": RANDOM_STATE, "n_jobs": -1})

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb.XGBClassifier(**best_params)),
])
pipeline.fit(X_train, y_train)

y_proba_test = pipeline.predict_proba(X_test)[:, 1]


# ── 5. Threshold tuning ───────────────────────────────────────────────────────

print(f"Tuning decision threshold (target recall ≥ {MIN_RECALL:.0%})...")
threshold = tune_threshold(y_test, y_proba_test, min_recall=MIN_RECALL)
y_pred_tuned = (y_proba_test >= threshold).astype(int)

print_report(y_test, y_pred_tuned, y_proba_test, threshold)


# ── 6. Evaluation plots ───────────────────────────────────────────────────────

print("\nGenerating evaluation plots...")
plot_confusion_matrix(y_test, y_pred_tuned)
plot_roc_curve(y_test, y_proba_test)
plot_precision_recall(y_test, y_proba_test, threshold)


# ── 7. SHAP explainability ────────────────────────────────────────────────────

print("\nComputing SHAP values...")
xgb_model = pipeline.named_steps["model"]
X_test_proc = pipeline.named_steps["preprocessor"].transform(X_test)
feature_names = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

import json
my_booster = xgb_model.get_booster()
config = json.loads(my_booster.save_config())
base_score = config["learner"]["learner_model_param"]["base_score"]
if isinstance(base_score, str) and base_score.startswith('[') and base_score.endswith(']'):
    config["learner"]["learner_model_param"]["base_score"] = base_score[1:-1]
    my_booster.load_config(json.dumps(config))

explainer = shap.TreeExplainer(my_booster)
shap_values = explainer.shap_values(X_test_proc)

# Global beeswarm summary
plt.figure()
shap.summary_plot(shap_values, X_test_proc, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/shap_summary.png")

# Bar chart — mean |SHAP|
plt.figure()
shap.summary_plot(shap_values, X_test_proc, feature_names=feature_names,
                  plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/shap_bar.png")

# Single patient waterfall (first test sample)
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test_proc[0],
    feature_names=feature_names,
), show=False)
plt.tight_layout()
plt.savefig("outputs/shap_patient_example.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/shap_patient_example.png")


# ── 8. Save artefacts ─────────────────────────────────────────────────────────

joblib.dump(pipeline, "models/heart_disease_pipeline.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")

with open("models/threshold.txt", "w") as f:
    f.write(str(threshold))

print("\nSaved:")
print("  models/heart_disease_pipeline.pkl")
print("  models/shap_explainer.pkl")
print("  models/threshold.txt")
print("\nTraining complete.")
