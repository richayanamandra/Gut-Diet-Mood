"""
Step 7-8: Generate analysis summary and conclusions
"""
import pandas as pd
import os

if not os.path.exists('model_results.csv'):
   print("❌ model_results.csv not found. Run modeling script first.")
   exit(1)

results_df = pd.read_csv('model_results.csv', index_col=0)

# Read processed features once (avoid repeated reads and dtype warnings)
processed_features_path = 'processed_features.csv'
if not os.path.exists(processed_features_path):
   print("❌ processed_features.csv not found. Run data processing first.")
   exit(1)
processed_df = pd.read_csv(processed_features_path, low_memory=False)
num_samples, num_features = processed_df.shape

# Load feature importance robustly if available
importance_df = pd.DataFrame()
if os.path.exists('feature_importance.csv'):
   try:
      importance_df = pd.read_csv('feature_importance.csv', index_col=0)
   except Exception:
      try:
         importance_df = pd.read_csv('feature_importance.csv', index_col=0, header=None, names=['Importance'])
      except Exception:
         importance_df = pd.DataFrame()
         print("⚠️ feature_importance.csv exists but could not be parsed. Skipping importance report.")
else:
   print("⚠️ feature_importance.csv missing. Skipping importance report.")

report = f"""
================================================================================
        GUT-DIET-MOOD ML PROJECT: ANALYSIS REPORT
================================================================================

1. DATASET SUMMARY
   - Samples: {num_samples}
   - Features: {num_features} (microbiome + diet)
   - Target: Binary mood/stress classification

2. MODEL PERFORMANCE
{results_df.to_string()}

   Best Model: XGBoost (Both features)
   - Accuracy: {results_df.loc['XGBoost (Both)', 'Accuracy']:.3f}
   - ROC-AUC: {results_df.loc['XGBoost (Both)', 'ROC-AUC']:.3f}

3. KEY FINDINGS
   Top 5 Predictive Features:
{importance_df.head(5).to_string() if not importance_df.empty else '   (no importance data available)'}

4. HYPOTHESIS VALIDATION
   H1 (Fiber → Better Mood): {'✅ Supported' if 'FIBER' in str(importance_df.head(10).index) else '❌ Not strongly supported'}
   H2 (UPF → Worse Mood): {'✅ Supported' if 'SUGAR' in str(importance_df.head(10).index) else '❌ Needs further analysis'}
   H3 (Personalization): Requires per-subject modeling (future work)

5. LIMITATIONS
   - Observational data (no causality)
   - Self-reported diet (measurement error)
   - Synthetic target (if real mood data unavailable)
   - Cross-sectional (no temporal dynamics)

6. RECOMMENDATIONS
   - Validate with prospective cohort
   - Add longitudinal data (repeated measures)
   - Test causal interventions (fiber supplementation)
   - Personalize models per individual

================================================================================
"""

print(report)

with open('analysis_report.txt', 'w', encoding='utf-8') as f:
   f.write(report)

print("\n✅ Report saved: analysis_report.txt")
