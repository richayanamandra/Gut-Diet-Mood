"""
Step 1-2: Load AGP data, align samples, engineer features
"""
import pandas as pd
import numpy as np
from biom import load_table
import os

# === Configuration ===
OTU_BIOM = r'data/03-otus/100nt/gg-13_8-97-percent/otu_table.biom'
META_FILE = r'data/04-meta/ag-cleaned.txt'
META_ALT = r'data/04-meta/ag-gg-cleaned.txt'

# === 1. Load OTU Table ===
print("Loading OTU table...")
table = load_table(OTU_BIOM)
otu_df = table.to_dataframe(dense=True).T  # Samples as rows

# === 2. Load Metadata ===
print("Loading metadata...")
if os.path.exists(META_FILE):
    meta_df = pd.read_csv(META_FILE, sep='\t', index_col=0, low_memory=False)
else:
    meta_df = pd.read_csv(META_ALT, sep='\t', index_col=0, low_memory=False)

# === 3. Clean Sample IDs ===
def norm_ids(ids):
    return [str(i).replace('-', '').replace('_', '').upper().strip() for i in ids]

otu_df.index = norm_ids(otu_df.index)
meta_df.index = norm_ids(meta_df.index)

# === 4. Align Samples ===
overlap = list(set(otu_df.index) & set(meta_df.index))
print(f"Overlapping samples: {len(overlap)}")

otu_df = otu_df.loc[overlap]
meta_df = meta_df.loc[overlap]

# === 5. Filter Rare OTUs (>1% prevalence) ===
print("Filtering rare OTUs...")
otu_prev = (otu_df > 0).sum(axis=0) / len(otu_df)
otu_keep = otu_prev[otu_prev > 0.01].index
otu_filtered = otu_df[otu_keep]

# === 6. Log-transform (compositional data) ===
otu_log = np.log1p(otu_filtered)

# === 7. Extract & Clean Diet Features ===
diet_cols = [
    'VIOSCREEN_FIBER', 'FERMENTED_PLANT_FREQUENCY',
    'VIOSCREEN_TOTSUGAR', 'VIOSCREEN_CAFFEINE', 
    'SUGARY_SWEETS_FREQUENCY'
]
diet_df = meta_df[diet_cols].replace("Unknown", np.nan)
diet_df = diet_df.apply(pd.to_numeric, errors='coerce')
diet_df = diet_df.fillna(diet_df.median())  # Median imputation

# === 8. Create Target Variable ===
# AGP doesn't have direct "mood" scores; simulate or use proxy
# Option 1: Use a health/wellness proxy if available
# Option 2: Create synthetic target for demo (remove for real analysis)
target_candidates = ['DEPRESSION_DIAGNOSED', 'ANXIETY_DIAGNOSED', 'SLEEP_DURATION']
target_col = None

for col in target_candidates:
    if col in meta_df.columns:
        target_col = col
        break

if target_col is None:
    # Simulate binary mood target based on fiber + fermented foods (for demo only)
    print("⚠️ No mood/stress variable found. Creating synthetic target for demo.")
    meta_df['MOOD_SYNTHETIC'] = (
        (diet_df['VIOSCREEN_FIBER'] > diet_df['VIOSCREEN_FIBER'].median()).astype(int) +
        (diet_df['FERMENTED_PLANT_FREQUENCY'].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0))
    )
    meta_df['MOOD_SYNTHETIC'] = (meta_df['MOOD_SYNTHETIC'] > 1).astype(int)
    target_col = 'MOOD_SYNTHETIC'

y = meta_df[target_col].replace("Unknown", np.nan).dropna()

# === 9. Combine Features ===
X = pd.concat([otu_log, diet_df], axis=1)
X = X.loc[y.index]  # Align with target

print(f"\nFinal dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target distribution:\n{y.value_counts()}")

# === 10. Save Processed Data ===
X.to_csv('processed_features.csv')
y.to_csv('processed_target.csv')
print("\n✅ Data saved: processed_features.csv, processed_target.csv")
