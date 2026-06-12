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
# # 02 — Feature Engineering
# Deriving interaction terms, bins, and risk flags from raw UCI features.

# %%
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import load_raw_data, engineer_features, ALL_FEATURES

sns.set_theme(style="whitegrid")
df_raw = load_raw_data()
df = engineer_features(df_raw)
df.head()

# %%
# Show new engineered columns
new_cols = ['age_thalach', 'chol_per_age', 'bp_category', 'high_risk_cp']
print("New features added:")
df[new_cols + ['target']].describe()

# %%
# age_thalach vs target
fig, ax = plt.subplots(figsize=(6, 4))
for label, color in [(0, '#5DCAA5'), (1, '#D85A30')]:
    subset = df[df.target == label]['age_thalach']
    subset.hist(bins=25, alpha=0.6, label='Disease' if label else 'No disease',
                color=color, ax=ax)
ax.set_title('age × thalach interaction by target')
ax.set_xlabel('age_thalach')
ax.legend()
plt.tight_layout()
plt.show()

# %%
# bp_category distribution
bp_dist = df.groupby(['bp_category', 'target']).size().unstack(fill_value=0)
bp_dist.plot(kind='bar', color=['#5DCAA5', '#D85A30'], figsize=(6, 4))
plt.title('BP category (JNC-8 stages) vs disease')
plt.xlabel('BP category (0=normal, 3=stage2 hypertension)')
plt.xticks(rotation=0)
plt.legend(['No disease', 'Disease'])
plt.tight_layout()
plt.show()

# %%
# high_risk_cp (asymptomatic chest pain) — disease rate
rate = df.groupby('high_risk_cp')['target'].mean()
print("Disease rate by high_risk_cp flag:")
print(rate.rename({0: 'Symptomatic (cp≠0)', 1: 'Asymptomatic (cp=0)'}))

# %%
# Correlation of new features with target
corr_with_target = df[ALL_FEATURES + ['target']].corr()['target'].drop('target').sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
colors = ['#D85A30' if c > 0 else '#5DCAA5' for c in corr_with_target]
corr_with_target.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Feature correlation with target')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()
