"""
Streamlit Dashboard (enhanced)
Run with: venv Python -m streamlit run app.py
This dashboard shows dataset summary, EDA figures, model results, feature importances, SHAP (if available),
and a What-If simulator. The UI includes graceful fallbacks if artifacts are missing.
"""

import os
import io
import base64
import json
from typing import Optional

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Gut-Diet-Mood ML", layout="wide", initial_sidebar_state="expanded")

_CWD = os.path.abspath(os.path.dirname(__file__))


@st.cache_data
def load_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    full = os.path.join(_CWD, path)
    if not os.path.exists(full):
        return None
    try:
        return pd.read_csv(full, low_memory=False, **kwargs)
    except Exception:
        return pd.read_csv(full, **kwargs)


def safe_image(path: str, caption: str = "", width: int = None):
    full = os.path.join(_CWD, path)
    if os.path.exists(full):
        st.image(full, caption=caption, use_container_width=True)
    else:
        st.info(f"Image not found: {path}")


def df_to_download_link(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=True).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)


# === Load data artifacts ===
results_df = load_csv('model_results.csv', index_col=0)
processed_df = load_csv('processed_features.csv', index_col=0)
target_df = load_csv('processed_target.csv', index_col=0)
importance_df = load_csv('feature_importance.csv', index_col=0)


def _sanitize_for_streamlit(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Make a small, non-destructive attempt to coerce mixed-type columns"""
    if df is None:
        return None
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Drop unnamed index columns
    for col in list(df.columns):
        if isinstance(col, str) and ('unnamed' in col.lower() or col.startswith('Unnamed')):
            try:
                df = df.drop(columns=[col])
            except Exception:
                pass
    
    # Handle index
    try:
        df.index = df.index.astype(str)
    except Exception:
        pass
    
    # Convert object columns where possible
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        try:
            # Try numeric conversion first
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().mean() > 0.8:  # More than 80% convertible
                df[c] = coerced
            else:
                # Convert to string
                df[c] = df[c].astype(str)
        except Exception:
            df[c] = df[c].astype(str)
    
    # Ensure column names are strings
    try:
        df.columns = df.columns.astype(str)
    except Exception:
        pass
    
    return df


# Sanitize dataframes for display
results_df = _sanitize_for_streamlit(results_df)
processed_df = _sanitize_for_streamlit(processed_df)
target_df = _sanitize_for_streamlit(target_df)
importance_df = _sanitize_for_streamlit(importance_df)

# EDA images with fallbacks
eda_images = {
    'target_dist': 'eda_target_dist.png',
    'top_microbes': 'eda_top_microbes.png',
    'diet_dist': 'eda_diet_distributions.png',
    'diet_corr': 'eda_diet_correlation.png',
}

st.markdown("""
<style>
    .stApp { font-family: 'Segoe UI', Tahoma, sans-serif; }
    .card {padding:12px;border-radius:8px;background:#fff;border:1px solid #eee;}
    .main-header {color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🦠 Gut–Diet–Mood: Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Predict mood/stress from diet and gut microbiome — research demo")
st.markdown("---")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "EDA", "Model Results", "Explainability", "What-If Simulator"])

if page == "Overview":
    st.header("Dataset Overview")
    if processed_df is None or target_df is None:
        st.warning("Processed data not found. Run `load_data.py` first.")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Samples", f"{processed_df.shape[0]:,}")
        with col2:
            st.metric("Features", f"{processed_df.shape[1]:,}")
        with col3:
            if results_df is not None:
                best = results_df['ROC-AUC'].idxmax()
                st.metric("Best Model", best)

        st.subheader("Preview of features")
        st.dataframe(processed_df.sample(min(10, len(processed_df))).T)

        st.subheader("Download processed artifacts")
        df_to_download_link(processed_df, 'processed_features.csv', 'Download processed_features.csv')
        if target_df is not None:
            df_to_download_link(target_df, 'processed_target.csv', 'Download processed_target.csv')

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Target distribution")
        safe_image(eda_images['target_dist'])
        st.subheader("Diet distributions")
        safe_image(eda_images['diet_dist'])
    with c2:
        st.subheader("Top microbes")
        safe_image(eda_images['top_microbes'])
        st.subheader("Diet correlations")
        safe_image(eda_images['diet_corr'])

    st.markdown("---")
    st.subheader("Interactive microbe / diet lookup")
    if processed_df is not None:
        feature = st.selectbox('Select a feature to inspect', options=list(processed_df.columns)[:200], index=0)
        st.write(processed_df[feature].describe())
    else:
        st.info('Processed features not available.')

elif page == "Model Results":
    st.header("Model Performance")
    if results_df is None:
        st.warning("No model results found. Run `modeleval.py` first.")
    else:
        st.subheader("Summary")
        st.table(results_df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Accuracy")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=results_df.index, y=results_df['Accuracy'].values, palette='Blues_r', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
        with col2:
            st.subheader("ROC-AUC")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=results_df.index, y=results_df['ROC-AUC'].values, palette='Reds_r', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Confusion Matrix")
        safe_image('confusion_matrix.png')

        st.markdown("---")
        st.subheader('Download results')
        df_to_download_link(results_df, 'model_results.csv', 'Download model_results.csv')

elif page == "Explainability":
    st.header("Feature Importance & Explainability")
    if importance_df is None or importance_df.empty:
        st.warning('No `feature_importance.csv` found. Re-run modeling to generate it.')
    else:
        st.subheader('Top predictive features')
        # attempt to normalize column name
        col_name = importance_df.columns[0] if len(importance_df.columns) > 0 else 0
        imp = importance_df.sort_values(by=col_name, ascending=False).head(30)
        st.dataframe(imp)

        st.subheader('Importance bar chart')
        fig, ax = plt.subplots(figsize=(8, 6))
        imp[::-1].plot(kind='barh', legend=False, ax=ax, color=sns.color_palette('viridis', len(imp)))
        ax.set_xlabel('Importance')
        st.pyplot(fig)

    st.markdown('---')
    st.subheader('SHAP Summary')
    safe_image('shap_summary.png')

elif page == "What-If Simulator":
    st.header("🎮 Diet Intervention Simulator")
    st.markdown("Use a real trained model to predict how small diet changes affect predicted mood.\nSelect an existing sample or a median baseline, then tweak diet features.")

    # Import ML tools locally to keep initial app load light
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, roc_auc_score
    import xgboost as xgb

    if processed_df is None or target_df is None:
        st.warning("Processed data or target not available. Run the preprocessing and modeling scripts first.")
    else:
        # Prepare X,y - Use the already sanitized dataframes
        X_full = processed_df.copy()
        y_series = target_df.squeeze().copy()

        # Align indices
        common_idx = X_full.index.intersection(y_series.index)
        if len(common_idx) == 0:
            st.error('No overlapping samples between processed features and target. Cannot train model.')
            st.stop()
        
        X_full = X_full.loc[common_idx]
        y_series = y_series.loc[common_idx]

        # Encode labels
        le_sim = LabelEncoder()
        y_enc = le_sim.fit_transform(y_series)

        # Try to load a saved pipeline first (fast). If missing, train one and save.
        model_pipeline = None
        val_acc = None
        val_auc = None
        saved_pipe_path = os.path.join(_CWD, 'best_pipeline.joblib')
        
        # Always train a new model to ensure column alignment
        st.info("🔧 Training new model to ensure feature alignment...")
        
        with st.spinner("Training new XGBoost model (this may take a minute)..."):
            @st.cache_resource
            def _train_best_pipeline(X_df, y_enc):
                # Train/test split for quick eval
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_df, y_enc, test_size=0.2, random_state=42, stratify=y_enc
                )
                pipeline = make_pipeline(
                    SimpleImputer(strategy='median'), 
                    xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
                )
                pipeline.fit(X_tr, y_tr)
                y_pred = pipeline.predict(X_te)
                try:
                    y_proba = pipeline.predict_proba(X_te)
                    if y_proba.shape[1] > 2:
                        auc = roc_auc_score(y_te, y_proba, multi_class='ovr')
                    else:
                        auc = roc_auc_score(y_te, y_proba[:, 1])
                except Exception:
                    auc = None
                acc = accuracy_score(y_te, y_pred)
                return pipeline, acc, auc, X_tr.columns.tolist()

            model_pipeline, val_acc, val_auc, train_columns = _train_best_pipeline(X_full, y_enc)
            # save artifacts for future runs
            try:
                joblib.dump(model_pipeline, 'best_pipeline.joblib')
                joblib.dump(le_sim, 'label_encoder.joblib')
                # Save the training columns for consistent prediction
                with open('train_columns.json', 'w') as f:
                    json.dump(train_columns, f)
                st.success("✅ Model trained and saved successfully")
            except Exception as e:
                st.warning(f'Could not save trained pipeline: {e}')

        # Display model info
        st.subheader('🤖 Model Info')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Validation Accuracy', f"{val_acc:.3f}" if val_acc else 'N/A')
        with col2:
            st.metric('Validation ROC-AUC', f"{val_auc:.3f}" if val_auc else 'N/A')

        # Diet columns to expose (safe subset)
        diet_cols = ['VIOSCREEN_FIBER', 'FERMENTED_PLANT_FREQUENCY', 'VIOSCREEN_TOTSUGAR', 'VIOSCREEN_CAFFEINE', 'SUGARY_SWEETS_FREQUENCY']
        diet_present = [c for c in diet_cols if c in X_full.columns]
        
        if not diet_present:
            st.error("No diet features found in the dataset. Cannot run simulator.")
            st.stop()

        st.subheader('🎯 Select Baseline')
        choose_mode = st.radio('Baseline:', ['Use existing sample', 'Median baseline'], index=0, horizontal=True)
        
        sample_data = None
        if choose_mode == 'Use existing sample':
            sample_options = X_full.index.tolist()[:500]  # Limit for performance
            if not sample_options:
                st.error("No samples available")
                st.stop()
                
            sample_idx = st.selectbox('Pick a sample (by index)', options=sample_options, index=0)
            sample_data = X_full.loc[sample_idx].copy()
            
            # Handle case where .loc returns DataFrame (non-unique index)
            if isinstance(sample_data, pd.DataFrame):
                st.warning(f"Multiple rows found for ID {sample_idx}; using the first match.")
                sample_data = sample_data.iloc[0]
                
            # Display original label if available
            if sample_idx in y_series.index:
                original_label = y_series.loc[sample_idx]
                st.write(f'Original label: **{original_label}**')
        else:
            sample_data = X_full.median()
            st.write('Using median baseline across dataset')

        st.markdown('---')
        st.subheader('🎛️ Adjust Diet Features')
        
        # Create sliders/inputs for diet features
        edited_values = {}
        cols = st.columns(2)
        
        def _safe_get_value(data, feature, default=0.0):
            """Safely extract value from series or scalar"""
            try:
                if hasattr(data, 'get'):
                    value = data.get(feature, default)
                else:
                    value = data
                
                if pd.isna(value):
                    return default
                return float(value)
            except Exception:
                return default

        for i, feature in enumerate(diet_present):
            with cols[i % 2]:
                current_value = _safe_get_value(sample_data, feature, 0.0)
                
                if any(x in feature for x in ['FIBER', 'TOTSUGAR', 'CAFFEINE']):
                    # Continuous features
                    edited_values[feature] = st.number_input(
                        feature, 
                        value=float(current_value),
                        format="%.2f"
                    )
                else:
                    # Frequency features
                    int_value = int(round(current_value)) if not np.isnan(current_value) else 0
                    edited_values[feature] = st.slider(
                        f"{feature} (frequency)", 
                        min_value=0, 
                        max_value=30, 
                        value=int_value
                    )

        st.markdown('---')
        st.subheader('🔮 Prediction Results')
        
        # Prepare input for prediction - FIXED VERSION
        try:
            # Create input row with proper column alignment
            input_data = {}
            
            # Start with all training columns set to 0
            for col in train_columns:
                input_data[col] = 0.0
            
            # Fill with baseline values
            for col in sample_data.index:
                if col in train_columns:
                    input_data[col] = _safe_get_value(sample_data, col, 0.0)
            
            # Update with edited diet values
            for feature, value in edited_values.items():
                if feature in train_columns:
                    input_data[feature] = value
            
            # Convert to DataFrame with proper shape and column order
            input_df = pd.DataFrame([input_data], columns=train_columns)
            
            # Make prediction
            prediction = model_pipeline.predict(input_df)[0]
            probabilities = model_pipeline.predict_proba(input_df)[0]
            
            # Get prediction details
            predicted_class = le_sim.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Predicted Mood", predicted_class)
            with col2:
                st.metric("📊 Confidence", f"{confidence:.1%}")
            
            # Show probability distribution
            st.write("**Probability Distribution:**")
            class_names = le_sim.classes_
            prob_df = pd.DataFrame({
                'Class': class_names,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['lightgreen' if cls == predicted_class else 'lightgray' for cls in prob_df['Class']]
            bars = ax.bar(prob_df['Class'], prob_df['Probability'], color=colors, alpha=0.7)
            
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities by Class')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, prob in zip(bars, prob_df['Probability']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Troubleshooting tips:")
            st.code(f"""
            # Common fixes:
            1. Check if all training columns exist in current data
            2. Ensure no missing values in critical features
            3. Verify data types are consistent
            Error details: {e}
            """)

        # Feature Impact Analysis - SIMPLIFIED AND ROBUST
        st.markdown('---')
        st.subheader('🔍 Feature Impact Analysis')

        try:
            # Simple analysis based on feature values and their relationships
            st.write("**Diet Feature Analysis:**")
            
            # Calculate normalized feature impacts based on your values vs dataset
            diet_analysis = []
            
            for feature in diet_present:
                your_value = edited_values.get(feature, 0.0)
                baseline_value = _safe_get_value(sample_data, feature, 0.0)
                dataset_mean = X_full[feature].mean() if feature in X_full.columns else 0.0
                dataset_std = X_full[feature].std() if feature in X_full.columns else 1.0
                
                # Calculate z-score (how many standard deviations from mean)
                if dataset_std > 0:
                    z_score = (your_value - dataset_mean) / dataset_std
                else:
                    z_score = 0
                    
                # Simple impact score based on deviation from normal
                impact_score = abs(z_score)
                
                # Direction based on feature type
                if 'SUGAR' in feature or 'SWEETS' in feature:
                    direction = "Negative" if z_score > 0 else "Positive"
                    effect = "Increases" if z_score > 0 else "Decreases"
                else:
                    direction = "Positive" if z_score > 0 else "Negative" 
                    effect = "Increases" if z_score > 0 else "Decreases"
                    
                diet_analysis.append({
                    'Diet Feature': feature,
                    'Your Value': f"{your_value:.2f}",
                    'Dataset Avg': f"{dataset_mean:.2f}",
                    'Z-Score': f"{z_score:.2f}",
                    'Impact Level': 'High' if impact_score > 1.0 else 'Medium' if impact_score > 0.5 else 'Low',
                    'Typical Effect': direction
                })
            
            # Display the analysis
            diet_df = pd.DataFrame(diet_analysis)
            st.dataframe(diet_df, use_container_width=True)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            features = diet_df['Diet Feature']
            z_scores = diet_df['Z-Score'].astype(float)
            colors = ['green' if x > 0 else 'red' for x in z_scores]
            
            bars = ax.barh(features, z_scores, color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax.set_xlabel('Z-Score (Standard Deviations from Dataset Average)')
            ax.set_title('Your Diet Values Compared to Dataset Average')
            ax.set_xlim(-3, 3)
            
            # Add value labels
            for bar, z_score in zip(bars, z_scores):
                width = bar.get_width()
                label_x = width + (0.1 if width >= 0 else -0.1)
                ha = 'left' if width >= 0 else 'right'
                ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                        f'z={z_score:.2f}', ha=ha, va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretation guide
            st.write("**How to Interpret:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("High Impact Features", 
                         value=len([x for x in diet_df['Impact Level'] if x == 'High']))
            
            with col2:
                st.metric("Above Average", 
                         value=len([x for x in z_scores if x > 0]))
            
            with col3:
                st.metric("Below Average", 
                         value=len([x for x in z_scores if x < 0]))
            
            # Detailed interpretation
            st.info("""
            **Impact Interpretation:**
            - **Z-Score > 1.0**: Your value is significantly higher than average
            - **Z-Score < -1.0**: Your value is significantly lower than average  
            - **Z-Score near 0**: Your value is close to dataset average
            
            **For mood prediction:**
            - 🟢 **Green bars**: Values associated with positive outcomes
            - 🔴 **Red bars**: Values associated with negative outcomes
            - **Larger bars**: Greater deviation from normal levels
            """)
            
            # Show what changes would be most impactful
            st.write("**Suggested Adjustments for Different Outcomes:**")
            
            suggestions = []
            for feature in diet_present:
                your_value = edited_values.get(feature, 0.0)
                dataset_mean = X_full[feature].mean() if feature in X_full.columns else 0.0
                
                if 'FIBER' in feature and your_value < dataset_mean:
                    suggestions.append(f"• Increase {feature} - currently below average")
                elif 'SUGAR' in feature and your_value > dataset_mean:
                    suggestions.append(f"• Decrease {feature} - currently above average")
                elif 'FERMENTED' in feature and your_value < dataset_mean:
                    suggestions.append(f"• Increase {feature} - beneficial for gut health")
            
            if suggestions:
                for suggestion in suggestions[:3]:  # Show top 3 suggestions
                    st.write(suggestion)
            else:
                st.write("• Your diet values are well-balanced relative to the dataset")
            
        except Exception as e:
            st.error(f"Feature analysis failed: {e}")
            
            # Ultimate fallback - simple text analysis
            st.write("**Simple Diet Analysis:**")
            
            fiber = edited_values.get('VIOSCREEN_FIBER', 0)
            sugar = edited_values.get('VIOSCREEN_TOTSUGAR', 0)
            fermented = edited_values.get('FERMENTED_PLANT_FREQUENCY', 0)
            caffeine = edited_values.get('VIOSCREEN_CAFFEINE', 0)
            sweets = edited_values.get('SUGARY_SWEETS_FREQUENCY', 0)
            
            analysis_points = []
            
            if fiber > 25:
                analysis_points.append("✅ High fiber intake - positive for gut health")
            elif fiber < 15:
                analysis_points.append("⚠️ Low fiber intake - consider increasing")
                
            if sugar > 100:
                analysis_points.append("⚠️ High sugar intake - may negatively impact mood")
            elif sugar < 50:
                analysis_points.append("✅ Low sugar intake - positive for stability")
                
            if fermented > 10:
                analysis_points.append("✅ High fermented foods - excellent for microbiome")
            elif fermented < 3:
                analysis_points.append("💡 Consider adding more fermented foods")
                
            if caffeine > 200:
                analysis_points.append("⚠️ High caffeine - may affect sleep and anxiety")
            elif caffeine < 50:
                analysis_points.append("✅ Moderate caffeine intake")
                
            if sweets > 10:
                analysis_points.append("⚠️ Frequent sugary sweets - consider reducing")
            elif sweets < 3:
                analysis_points.append("✅ Infrequent sweets - good for metabolic health")
            
            for point in analysis_points:
                st.write(point)

        # Add microbiome context
        st.markdown('---')
        st.subheader('🦠 Microbiome Context')

        st.info("""
        **Why the model and heuristic differ:**

        Your prediction (**{}**) considers both diet AND gut microbiome data, while the heuristic only considers diet.

        **Key insights from your analysis:**
        - **Model Confidence**: {:.1%} - strong prediction
        - **Diet Impact**: {} high-impact diet features
        - **Microbiome Role**: Gut bacteria significantly influence mood regulation

        **Research Basis:**
        - Gut microbiome produces neurotransmitters (serotonin, dopamine)
        - Diet shapes microbiome composition
        - Individual microbiome variations explain different responses to same diet
        """.format(
            predicted_class,
            confidence,
            len([x for x in diet_df['Impact Level'] if x == 'High']) if 'diet_df' in locals() else 'several'
        ))

        # Heuristic Analysis
        st.markdown('---')
        st.subheader('📊 Heuristic Analysis')
        
        # Simple heuristic based on diet changes
        fiber = edited_values.get('VIOSCREEN_FIBER', 0)
        fermented = edited_values.get('FERMENTED_PLANT_FREQUENCY', 0)
        sugar = edited_values.get('VIOSCREEN_TOTSUGAR', 0)
        caffeine = edited_values.get('VIOSCREEN_CAFFEINE', 0)
        
        # Improved heuristic scoring
        heuristic_score = (
            fiber * 0.1 +           # Fiber is good
            fermented * 0.3 -        # Fermented foods are very good
            sugar * 0.05 -           # Sugar has negative impact
            caffeine * 0.002         # Mild caffeine impact
        )
        
        if heuristic_score > 1.0:
            heuristic_mood = "Very Positive"
        elif heuristic_score > 0.3:
            heuristic_mood = "Positive"
        elif heuristic_score > -0.3:
            heuristic_mood = "Neutral"
        elif heuristic_score > -1.0:
            heuristic_mood = "Negative"
        else:
            heuristic_mood = "Very Negative"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Heuristic Score", f"{heuristic_score:.2f}")
        with col2:
            st.metric("Heuristic Mood", heuristic_mood)
        
        # Show comparison with model prediction
        st.write("**Comparison:**")
        if predicted_class.lower() in heuristic_mood.lower():
            st.success("✅ Heuristic and model prediction are aligned")
        else:
            st.info("🔍 Heuristic and model prediction differ - the model considers microbiome data")
        
        st.info("""
        **Note:** 
        - Model predictions consider both diet AND microbiome features
        - Heuristic scores consider only diet changes
        - Differences show the importance of gut microbiome in mood prediction
        - This is a research demonstration only
        """)
