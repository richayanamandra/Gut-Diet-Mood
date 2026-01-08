"""
Step 3: Exploratory Data Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Load Processed Data ===
X = pd.read_csv('processed_features.csv', index_col=0, low_memory=False)
y = pd.read_csv('processed_target.csv', index_col=0, low_memory=False).squeeze()

# Normalize sample IDs to strings to avoid mixed-type index issues
def _norm_ids(ids):
    return ids.astype(str).str.replace('-', '').str.replace('_', '').str.upper().str.strip()

X.index = _norm_ids(X.index)
try:
    y.index = _norm_ids(y.index)
except Exception:
    # if y is a scalar or otherwise not indexable, skip
    pass

# === Split Features ===
diet_cols = ['VIOSCREEN_FIBER', 'FERMENTED_PLANT_FREQUENCY', 
             'VIOSCREEN_TOTSUGAR', 'VIOSCREEN_CAFFEINE', 'SUGARY_SWEETS_FREQUENCY']
otu_cols = [c for c in X.columns if c not in diet_cols]

# Convert diet columns to numeric (coerce errors, handle "Unknown")
for col in diet_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X_diet = X[diet_cols]
X_microbe = X[otu_cols]

# === 1. Target Distribution ===
plt.figure(figsize=(6, 4))
y.value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('Target Distribution (Mood/Stress)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda_target_dist.png', dpi=150)
plt.close()

# === 2. Diet Feature Distributions ===
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, col in enumerate(diet_cols):
    ax = axes.flat[i]
    X_diet[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(col)
    ax.set_xlabel('Value')
axes.flat[-1].axis('off')
plt.tight_layout()
plt.savefig('eda_diet_distributions.png', dpi=150)
plt.close()

# === 3. Top 10 Microbes (Abundance) ===
top_microbes = X_microbe.mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
top_microbes.plot(kind='barh', color='coral')
plt.title('Top 10 Most Abundant Microbes (Log-transformed)')
plt.xlabel('Mean Abundance')
plt.tight_layout()
plt.savefig('eda_top_microbes.png', dpi=150)
plt.close()

# === 4. Correlation Heatmap (Diet Features) ===
plt.figure(figsize=(7, 5))
sns.heatmap(X_diet.corr(), annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1)
plt.title('Diet Feature Correlations')
plt.tight_layout()
plt.savefig('eda_diet_correlation.png', dpi=150)
plt.close()

# === 5. Diet vs Target Boxplots ===
fig, axes = plt.subplots(1, len(diet_cols), figsize=(15, 4))
for i, col in enumerate(diet_cols):
    # Skip plotting if column missing or has no data
    if col not in X.columns or X[col].dropna().empty:
        axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[i].set_title(col)
        axes[i].set_xlabel('Target')
        continue

    df_plot = pd.DataFrame({'diet': X[col], 'target': y}).dropna()
    if df_plot.empty:
        axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[i].set_title(col)
        axes[i].set_xlabel('Target')
        continue

    df_plot.boxplot(column='diet', by='target', ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('Target')
plt.suptitle('Diet Features by Target Class', y=1.02)
plt.tight_layout()
plt.savefig('eda_diet_vs_target.png', dpi=150)
plt.close()

print("✅ EDA complete. Plots saved.")
