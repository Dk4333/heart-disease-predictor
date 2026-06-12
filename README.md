# Heart Disease Prediction — XGBoost + SHAP

> End-to-end ML project: clinical feature engineering, XGBoost with Optuna tuning, SHAP explainability, and a live Streamlit demo.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange) ![SHAP](https://img.shields.io/badge/SHAP-explainability-green) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

---

## Project overview

Predicts the presence of heart disease from 13 clinical features using the UCI Heart Disease dataset. Key differentiators over standard notebooks:

- **Feature engineering** — interaction terms, BP category bins, high-risk chest pain flag
- **Optuna hyperparameter tuning** — 50-trial Bayesian search over XGBoost params
- **Threshold tuning** — optimized for 90%+ recall (minimizes missed diagnoses)
- **SHAP explainability** — global feature importance + per-patient waterfall plots
- **Streamlit app** — live risk predictor with real-time SHAP explanation

**Results:** ROC-AUC ~0.91 | Recall (disease) ~0.92 | Precision ~0.88

---

## Project structure

```
heart-disease-predictor/
├── data/
│   └── heart.csv                  # UCI Heart Disease dataset
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling_shap.ipynb    # Training, tuning, SHAP
├── src/
│   ├── features.py                # Feature engineering logic
│   ├── train.py                   # Training pipeline
│   └── evaluate.py                # Evaluation utilities
├── models/                        # Saved pipeline + explainer (after training)
├── outputs/                       # SHAP plots, confusion matrix
├── app.py                         # Streamlit application
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt

# 2. Train the model
python src/train.py

# 3. Launch the app
streamlit run app.py
```

---

## Dataset

**UCI Heart Disease Dataset** — 303 patients, 13 clinical features, binary target (disease / no disease).

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0–2) |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–3) |
| thal | Thalassemia (0–3) |

Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## Model approach

```
Raw data
  └── Feature engineering (interaction terms, bins, flags)
        └── ColumnTransformer (StandardScaler on continuous)
              └── XGBoostClassifier
                    └── Optuna tuning (50 trials, StratifiedKFold CV)
                          └── Threshold optimization (recall ≥ 0.90)
                                └── SHAP TreeExplainer
```

**Key hyperparameters tuned:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | ~0.91 |
| Accuracy | ~0.87 |
| Recall (disease) | ~0.92 |
| Precision (disease) | ~0.88 |
| F1 Score | ~0.90 |

Threshold tuned to prioritize recall — in clinical settings, false negatives (missed disease) carry higher cost than false positives.

---

## SHAP explainability

Top features by mean |SHAP value|:
1. `ca` — number of blocked major vessels (strongest signal)
2. `cp` — chest pain type (asymptomatic = highest risk)
3. `thal` — thalassemia type
4. `oldpeak` — ST depression on exercise
5. `thalach` — maximum heart rate

The Streamlit app shows a per-patient waterfall plot explaining exactly why the model gave that prediction — essential for clinical trust.

---

## Tech stack

| Tool | Purpose |
|---|---|
| `xgboost` | Gradient boosted trees |
| `shap` | Model explainability |
| `optuna` | Hyperparameter tuning |
| `scikit-learn` | Preprocessing, pipelines, evaluation |
| `streamlit` | Interactive web app |
| `pandas / numpy` | Data manipulation |
| `matplotlib / seaborn` | Visualization |
| `joblib` | Model serialization |
| `ucimlrepo` | Dataset loading |

---

## Resume bullet

> Built an end-to-end heart disease prediction system using XGBoost with Optuna hyperparameter tuning (ROC-AUC: 0.91), SHAP-based clinical explainability, and a live Streamlit interface — threshold tuned to achieve 90%+ recall for disease detection.
