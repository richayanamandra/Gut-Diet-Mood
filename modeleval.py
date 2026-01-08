"""
Step 4-5: Train models, evaluate, explain with SHAP - FIXED VERSION
"""
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import joblib
import json

# Filter warnings
warnings.filterwarnings('ignore')

# === Load Data ===
X = pd.read_csv('processed_features.csv', index_col=0, low_memory=False)
X = X.fillna(0)
y = pd.read_csv('processed_target.csv', index_col=0, low_memory=False).squeeze()

# Normalize sample IDs to strings to avoid mixed-type index issues
def _norm_ids(ids):
    return ids.astype(str).str.replace('-', '').str.replace('_', '').str.upper().str.strip()

X.index = _norm_ids(X.index)
try:
    y.index = _norm_ids(y.index)
except Exception:
    pass

# === Clean target ===
ambiguous = ['Unspecified']
y = y[~y.isin(ambiguous)]

# Align X accordingly - ensure indices match after filtering
common_idx = y.index.intersection(X.index)
X = X.loc[common_idx]
y = y.loc[common_idx]

print(f"Data shape after cleaning: X={X.shape}, y={y.shape}")

# Encode target to numeric labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training features: {X_train.shape}, Test features: {X_test.shape}")

# FIX: Align columns between train and test to ensure same features
print("Aligning columns between train and test sets...")
common_cols = X_train.columns.intersection(X_test.columns)
print(f"Common columns: {len(common_cols)}")

if len(common_cols) < len(X_train.columns):
    missing_in_test = set(X_train.columns) - set(X_test.columns)
    missing_in_train = set(X_test.columns) - set(X_train.columns)
    print(f"Columns in train but not test: {len(missing_in_test)}")
    print(f"Columns in test but not train: {len(missing_in_train)}")
    
    # Use only common columns
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

print(f"After alignment - Training: {X_train.shape}, Test: {X_test.shape}")

# Define diet and microbe feature columns
diet_cols = ['VIOSCREEN_FIBER', 'FERMENTED_PLANT_FREQUENCY', 
            'VIOSCREEN_TOTSUGAR', 'VIOSCREEN_CAFFEINE', 'SUGARY_SWEETS_FREQUENCY']
# Use only present diet columns; treat rest as microbe features
diet_cols_present = [c for c in diet_cols if c in X.columns]
missing_diet = set(diet_cols) - set(diet_cols_present)
if missing_diet:
    warnings.warn(f"Missing diet columns: {sorted(list(missing_diet))}")
microbe_cols = [c for c in X.columns if c not in diet_cols_present]

# Models to train
models = {
    'Logistic (Diet)': LogisticRegression(max_iter=1000, random_state=42),
    'Logistic (Microbe)': LogisticRegression(max_iter=1000, random_state=42),
    'Logistic (Both)': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest (Both)': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost (Both)': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
}

results = {}

for name, model in models.items():
    try:
        if 'Diet' in name:
            # Use only the diet columns that are actually present
            if not diet_cols_present:
                warnings.warn(f"No diet columns present for {name}; skipping this model.")
                continue
            # Ensure diet columns exist in both train and test
            diet_cols_common = [c for c in diet_cols_present if c in X_train.columns and c in X_test.columns]
            X_tr, X_te = X_train[diet_cols_common], X_test[diet_cols_common]
        elif 'Microbe' in name:
            # Ensure microbe columns exist in both train and test
            microbe_cols_common = [c for c in microbe_cols if c in X_train.columns and c in X_test.columns]
            X_tr, X_te = X_train[microbe_cols_common], X_test[microbe_cols_common]
        else:
            X_tr, X_te = X_train, X_test

        print(f"Training {name} with features: {X_tr.shape}")

        # Pipeline with median imputation for missing values
        imputer = SimpleImputer(strategy='median')
        pipeline = make_pipeline(imputer, model)
        pipeline.fit(X_tr, y_train)
        
        y_pred = pipeline.predict(X_te)
        
        try:
            y_proba = pipeline.predict_proba(X_te)
            if y_proba.shape[1] > 2:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_proba[:, 1])
        except Exception as e:
            print(f"Warning while calculating ROC-AUC for {name}: {e}")
            auc = None
        
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = {'Accuracy': acc, 'ROC-AUC': auc}
        print(f"{name}: Accuracy={acc:.3f}, ROC-AUC={auc if auc is not None else 'NA'}")
    
    except Exception as e:
        print(f"Error training {name}: {e}")
        continue

if results:
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_results.csv')
    print("\n✅ Model results saved: model_results.csv")
else:
    print("❌ No models were successfully trained")
    exit(1)

# Plot confusion matrix for best model (XGBoost)
try:
    best_pipeline = make_pipeline(
        SimpleImputer(strategy='median'), 
        xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
    )
    best_pipeline.fit(X_train, y_train)
    # save trained pipeline, label encoder, and train columns for downstream use (app & reporting)
    try:
        joblib.dump(best_pipeline, 'best_pipeline.joblib')
        # save the label encoder used earlier (le)
        joblib.dump(le, 'label_encoder.joblib')
        train_columns = list(X_train.columns)
        with open('train_columns.json', 'w') as fh:
            json.dump(train_columns, fh)
        print("✅ Saved best_pipeline.joblib, label_encoder.joblib, train_columns.json")
    except Exception as e:
        print(f"Warning: failed to save pipeline artifacts: {e}")
    y_pred_best = best_pipeline.predict(X_test)

    # record the column order used to train the best pipeline
    train_columns = list(X_train.columns)

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (XGBoost)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("✅ Confusion matrix saved: confusion_matrix.png")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")

# === SHAP Explanations ===
try:
    print("\nComputing SHAP values...")
    
    # Get the trained XGBoost model from the pipeline
    xgb_model = best_pipeline.named_steps['xgbclassifier']
    
    # Prepare data for SHAP - use the imputer from the pipeline
    imputer = best_pipeline.named_steps['simpleimputer']
    X_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=train_columns,
        index=X_test.index
    )
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_imputed)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Multi-class: use first class for summary plot
        if len(np.unique(y_encoded)) <= 2:
            shap_values_array = shap_values
        else:
            shap_values_array = shap_values[0]
    else:
        shap_values_array = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_array, X_imputed, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ SHAP summary plot saved: shap_summary.png")
    
except Exception as e:
    print(f"SHAP computation failed: {e}")
    # Fallback: compute permutation importances
    try:
        print("Computing permutation importances as SHAP fallback...")
        
        # Use the already trained best_pipeline with properly aligned data
        perm = permutation_importance(
            best_pipeline, X_test, y_test, 
            n_repeats=3, random_state=42, n_jobs=1
        )
        
        perm_ser = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False).head(20)
        perm_ser.to_csv('permutation_importance.csv')
        
        plt.figure(figsize=(8, 6))
        perm_ser.plot(kind='barh', color='coral')
        plt.title('Top 20 Permutation Importances (fallback)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('permutation_importance.png')
        plt.close()
        print("✅ Permutation importance fallback completed")
        
    except Exception as e2:
        print(f"Permutation importance failed: {e2}")

# Feature Importance Plot
try:
    clf = best_pipeline.named_steps['xgbclassifier']
    fi = clf.feature_importances_
    
    # Get feature names - use training columns (already aligned)
    importance = pd.Series(fi, index=train_columns).sort_values(ascending=False).head(20)
    importance.to_csv('feature_importance.csv')

    plt.figure(figsize=(8, 6))
    importance.plot(kind='barh', color='teal')
    plt.title('Top 20 Feature Importances (XGBoost)')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("✅ Feature importance plot saved: feature_importance.png")
    
except Exception as e:
    print(f"Feature importance plotting failed: {e}")

print("✅ Modeling & SHAP analysis complete.")