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
# # 01 — Exploratory Data Analysis
# Heart Disease Prediction · UCI Dataset

# %%
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import load_raw_data

sns.set_theme(style="whitegrid", palette="muted")
df = load_raw_data()
print(df.shape)
df.head()

# %%
# Target distribution
fig, ax = plt.subplots(figsize=(5, 3))
df['target'].value_counts().plot(kind='bar', ax=ax, color=['#5DCAA5', '#D85A30'])
ax.set_xticklabels(['No disease', 'Disease'], rotation=0)
ax.set_title('Target distribution')
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()

# %%
# Missing values
print("Missing values:")
print(df.isnull().sum())
print(f"\nDtypes:\n{df.dtypes}")

# %%
# Age distribution by target
fig, ax = plt.subplots(figsize=(7, 4))
for label, color in [(0, '#5DCAA5'), (1, '#D85A30')]:
    df[df.target == label]['age'].hist(
        bins=20, alpha=0.6, label='Disease' if label else 'No disease',
        color=color, ax=ax
    )
ax.set_xlabel('Age')
ax.set_title('Age distribution by target')
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature correlation matrix')
plt.tight_layout()
plt.show()

# %%
# Key clinical features vs target
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
features = ['age', 'thalach', 'chol', 'oldpeak', 'trestbps', 'ca']
for ax, feat in zip(axes.flatten(), features):
    df.boxplot(column=feat, by='target', ax=ax)
    ax.set_title(feat)
    ax.set_xlabel('0 = No disease, 1 = Disease')
plt.suptitle('Clinical features by target', y=1.01)
plt.tight_layout()
plt.show()

# %%
# Chest pain type breakdown
cp_dist = df.groupby(['cp', 'target']).size().unstack(fill_value=0)
cp_dist.plot(kind='bar', figsize=(6, 4), color=['#5DCAA5', '#D85A30'])
plt.xlabel('Chest pain type')
plt.ylabel('Count')
plt.title('Chest pain type vs disease presence')
plt.xticks(rotation=0)
plt.legend(['No disease', 'Disease'])
plt.tight_layout()
plt.show()
