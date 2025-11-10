import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import networkx as nx
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration & Setup ---
plt.style.use('dark_background')
sns.set_style("darkgrid")

st.set_page_config(page_title="Gut-Diet-Mood Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🧬 Gut-Diet-Mood Analysis Project")
st.markdown("Exploring the **Microbiome, Diet, and Mood** relationship with a focus on data visualization and model explainability.")

warnings.filterwarnings('ignore') # Suppress warnings

# --- Helper Function for Data Info ---
def display_data_info(df):
    """Displays a clean summary of DataFrame columns and types."""
    st.subheader("Dataset Information (Structure)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Total Samples", df.shape[0])
        st.metric("Total Features", df.shape[1])
        st.markdown("**Data Types Overview:**")
        # Convert dtype objects to strings to avoid Arrow conversion issues
        dt_counts = df.dtypes.value_counts().rename_axis('Data Type').to_frame('Count').reset_index()
        dt_counts['Data Type'] = dt_counts['Data Type'].astype(str)
        st.dataframe(dt_counts, width='stretch')

    with col2:
        st.markdown("**Column Details (First 5):**")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.count(),
            'Dtype': df.dtypes
        }).reset_index(drop=True).head(5)
        # Cast dtype objects to strings to avoid Arrow/pyarrow errors
        info_df['Dtype'] = info_df['Dtype'].astype(str)
        st.dataframe(info_df, width='stretch')

# --- 1. Data Loading and Processing (Cached) ---
@st.cache_data
def load_and_process_data():
    # Load data
    df = pd.read_csv("project_catalog.csv")
    
    # Clean data (for correlation and initial processing)
    drop_cols = [
        'HMP ID', 'GOLD ID', 'Project Status', 'NCBI Submission Status',
        'IMG/HMP ID', 'Funding Source', 'Strain Repository ID'
    ]
    df_clean = df.drop(columns=drop_cols, errors='ignore').copy()
    
    # Encode categorical variables
    label_enc = LabelEncoder()
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = label_enc.fit_transform(df_clean[col].astype(str))
    
    # Generate synthetic diet and mood scores
    np.random.seed(42)
    df_mood = pd.DataFrame({
        'Organism Name': df['Organism Name'],
        'diet_score': np.random.randint(1, 10, size=len(df)),
        'mood_score': np.random.randint(1, 10, size=len(df))
    })
    
    # Merge data
    merged_df = pd.merge(df, df_mood, on='Organism Name', how='inner')
    
    # Identify numeric features for scaling (from merged_df, excluding non-features like scores or clusters)
    numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    features_for_scaling = [col for col in numeric_cols if col not in ['diet_score', 'mood_score']]

    # Scale numeric features
    scaler = StandardScaler()
    merged_scaled = scaler.fit_transform(merged_df[features_for_scaling].fillna(0)) # Fill NaN for scaling/PCA
    
    # Run PCA 
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(merged_scaled)
    
    # Run clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(merged_scaled)
    merged_df['Cluster'] = clusters
    
    return merged_df, df_clean, merged_scaled, pca_result

# --- 2. Cached Model Training ---
@st.cache_resource
def train_mood_prediction_model(merged_df):
    """Trains and caches the Random Forest Regressor model."""
    features = ['Gene Count', 'diet_score', 'Cluster']
    if not all(f in merged_df.columns for f in features):
        return None, None, None, None, None, None

    X = merged_df[features].dropna()
    y = merged_df['mood_score'].loc[X.index]
    
    # Split data for performance metrics (Section 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model for Metrics (Trained on 80% split)
    rf_reg_split = RandomForestRegressor(random_state=42, n_estimators=200)
    rf_reg_split.fit(X_train, y_train)
    
    # Model for SHAP (Trained on full data for robust feature influence, Section 5)
    rf_reg_full = RandomForestRegressor(random_state=42, n_estimators=200)
    rf_reg_full.fit(X, y) 

    return rf_reg_split, X_test, y_test, rf_reg_full, X, y

# --- Load Data and Train Model ---
try:
    merged_df, df_clean, scaled, pca_result = load_and_process_data()
except FileNotFoundError:
    st.error("🚨 Error: The required file 'project_catalog.csv' was not found.")
    st.markdown("Please ensure the data file is in the same directory as this Streamlit application.")
    st.stop()

# Train/Load the cached model
rf_reg_split, X_test_split, y_test_split, rf_reg_full, X_full, y_full = train_mood_prediction_model(merged_df)

if rf_reg_split is None:
    st.warning("⚠️ Model training skipped: Required features ('Gene Count', 'diet_score', 'Cluster') not found after data processing.")
    st.stop()


# --- Sidebar Navigation ---
st.sidebar.header("🔍 Analysis Sections")
section = st.sidebar.radio(
    "Select a view:",
    ["1. Data Overview & EDA",
     "2. PCA & Clustering",
     "3. Gut-Mood Prediction Model",
     "4. Behavioral Insights",
     "5. Network & Explainability"]
)

# --- 3. Content Rendering ---

# 1. Data Overview & EDA
if section == "1. Data Overview & EDA":
    st.header("📊 Data Overview & EDA")
    
    col_data, col_info = st.columns([2, 1])
    
    with col_data:
        st.subheader("Raw Data Sample")
        st.dataframe(merged_df.head(), width='stretch')
    
    with col_info:
        display_data_info(merged_df)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Domain Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(y=merged_df['Domain'], ax=ax, palette='YlGnBu_r') 
        ax.set_title('Distribution of Domains', color='white')
        st.pyplot(fig)
    
    with col4:
        st.subheader("Top Body Site Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        merged_df['HMP Isolation Body Site'].value_counts().head(10).plot(kind='bar', ax=ax, color=sns.color_palette("Set2"))
        ax.set_title('Top 10 Samples per Body Site', color='white')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_clean.corr(), cmap='coolwarm', ax=ax, annot=False, fmt=".2f", linewidths=0.5, linecolor='gray')
    ax.set_title('Overall Feature Correlation Matrix (Cleaned Data)', color='white')
    st.pyplot(fig)

# 2. PCA & Clustering
elif section == "2. PCA & Clustering":
    st.header("🔬 PCA & Clustering Analysis")
    st.markdown("Visualizing high-dimensional data reduction and intrinsic groupings.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PCA Visualization (2 Components)")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                             c=merged_df['Cluster'], cmap='plasma', alpha=0.8)
        plt.colorbar(scatter, label='K-Means Cluster')
        ax.set_title('PCA Visualization of Microbiome Patterns', color='white')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Cluster Distribution")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.countplot(x='Cluster', data=merged_df, ax=ax, palette='magma')
        ax.set_title('Distribution of K-Means Clusters (n=5)', color='white')
        st.pyplot(fig)

# 3. Gut-Mood Prediction Model
elif section == "3. Gut-Mood Prediction Model":
    st.header("🧠 Gut-Mood Prediction Model (Random Forest)")
    
    # Use cached model and test sets
    y_pred = rf_reg_split.predict(X_test_split) 
    
    st.subheader("Model Performance Metrics (Trained on 80% Data)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score (Test Set)", round(r2_score(y_test_split, y_pred), 3), delta="Higher is better")
    
    with col2:
        st.metric("RMSE (Test Set)", round(np.sqrt(mean_squared_error(y_test_split, y_pred)), 3), delta_color="inverse", delta="Lower is better")
    
    with col3:
        st.info(f"**Trained on:** {len(X_test_split)} samples\n\n**Algorithm:** Random Forest Regressor")

    # Feature importance
    st.subheader("Feature Importance")
    importances = pd.Series(rf_reg_split.feature_importances_, index=X_test_split.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind='bar', ax=ax, color='cyan')
    ax.set_title('Feature Importance in Mood Prediction', color='white')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

# 4. Behavioral Insights
elif section == "4. Behavioral Insights":
    st.header("📈 Behavioral Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gene Count vs Mood Score by Body Site")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=merged_df, x='Gene Count', y='mood_score', 
                        hue='HMP Isolation Body Site', alpha=0.8, ax=ax, palette='hsv')
        ax.set_title('Microbiome Size (Gene Count) vs Mood Score', color='white')
        ax.legend(title='Body Site', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Average Mood Score per Cluster")
        mood_cluster = merged_df.groupby('Cluster')['mood_score'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=mood_cluster.index, y=mood_cluster.values, palette='cool', ax=ax)
        ax.set_title('Average Mood Score per Microbiome Cluster', color='white')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Mood Score (Avg)')
        st.pyplot(fig)

# 5. Network & Explainability
else:
    st.header("🔗 Network Analysis & Model Explainability (SHAP)")
    
    # Network Graph
    st.subheader("Gut-Diet-Mood Relationship Network")
    subset = merged_df.sample(50, random_state=42)
    G = nx.Graph()
    
    for i, row in subset.iterrows():
        G.add_node(row['Organism Name'], type='bacteria')
        G.add_node(row['HMP Isolation Body Site'], type='site')
        G.add_node(f"Diet:{row['diet_score']}", type='diet')
        G.add_node(f"Mood:{row['mood_score']}", type='mood')
        G.add_edges_from([
            (row['Organism Name'], row['HMP Isolation Body Site']),
            (row['Organism Name'], f"Diet:{row['diet_score']}"),
            (row['Organism Name'], f"Mood:{row['mood_score']}")
        ])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Map organism nodes to their cluster (from the subset) and assign fixed colors for other node types
    organism_cluster_map = subset.set_index('Organism Name')['Cluster'].to_dict()
    node_colors = []
    for node in G.nodes():
        node_str = str(node)
        if node_str in organism_cluster_map:
            # map cluster integer to a matplotlib color
            cluster_idx = int(organism_cluster_map[node_str])
            color = plt.cm.tab10(cluster_idx % 10)
        elif node_str.startswith('Diet:'):
            color = 'green'
        elif node_str.startswith('Mood:'):
            color = 'red'
        else:
            # body site or other node
            color = 'orange'
        node_colors.append(color)

    nx.draw(G, pos, with_labels=False, node_size=100, node_color=node_colors,
            edge_color='gray', alpha=0.8, ax=ax)
    ax.set_title("Gut-Diet-Mood Relationship Network (Sample)", color='white')
    st.pyplot(fig)

    # SHAP Analysis - Using the full model (rf_reg_full)
    st.subheader("Model Explainability (SHAP Values)")
    
    explainer = shap.TreeExplainer(rf_reg_full)
    shap_values = explainer.shap_values(X_full)
    
    # 1. SHAP Bar Summary Plot
    st.markdown("**Feature Impact (Average Magnitude):**")
    fig_shap_bar = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_full, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(fig_shap_bar)
    plt.close(fig_shap_bar)
    
    # 2. SHAP Beeswarm Summary Plot
    st.markdown("**Feature Impact by Instance (Beeswarm Plot):**")
    fig_beeswarm = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_full, show=False)
    plt.tight_layout()
    st.pyplot(fig_beeswarm)
    plt.close(fig_beeswarm)