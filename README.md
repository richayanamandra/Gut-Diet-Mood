The project, titled **GUT-DIET-MOOD ML PROJECT**, follows a standard machine learning workflow, including data loading, exploratory data analysis (EDA), model training/evaluation, and reporting, which is encapsulated in a Streamlit dashboard.

### Topics Covered in the Documents

| Topic                         | Description                                                                                                                                                                                                                                                                                                                                                                                                   |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **ML Workflow Scripts**       | The core of the project is defined by Python scripts for each step: `load_data.py` (loading AGP and metadata), `eda.py` (exploratory analysis), `modeleval.py` (model training and evaluation, including Logistic Regression, Random Forest, and XGBoost), and `report.py` (summary generation).                                                                                                              |
| **Data & Features**           | The input data combines gut microbiome data (likely OTU counts from a BIOM file) and diet features (e.g., `VIOSCREEN_FIBER`, `FERMENTED_PLANT_FREQUENCY`). The target variable is present in `processed_target.csv`, which seems to be a categorical variable like **SLEEP\_DURATION** (categories include "Less than 5 hours," "5-6 hours," "6-7 hours," "7-8 hours," "8 or more hours," and "Unspecified"). |
| **Model Performance**         | Model training utilizes 17,170 samples and 4,776 features. The models aim for a "Binary mood/stress classification," although the `processed_target.csv` lists `SLEEP_DURATION`. The reported metrics show that model performance, at least for the preliminary Logistic (Diet) model, is poor.                                                                                                               |
| **Visualization & Dashboard** | Multiple `.png` files provide visualizations for the analysis, and `app.py` sets up a Streamlit dashboard to present the dataset overview, EDA figures, model results, feature importances, SHAP analysis, and a "What-If Simulator."                                                                                                                                                                         |

### Key Findings and Model Results

The `analysis_report.txt` and `model_results.csv` files provide a direct summary of the dataset and the initial model performance:

**Dataset Summary:**

  * **Samples:** 17,170
  * **Features:** 4,776 (microbiome + diet)
  * **Target:** Binary mood/stress classification (though raw target is **SLEEP\_DURATION** categories).

**Model Performance Summary:**

| Model           | Accuracy | ROC-AUC |
| :-------------- | :------- | :------ |
| Logistic (Diet) | 0.4053   | 0.5047  |

The ROC-AUC of 0.5047 for the Logistic (Diet) model suggests performance is barely better than random chance (0.50), indicating that the initial features or simple model struggled to predict the target successfully.

### Key Files and Their Roles

| File Name              | Type          | Role in Project                                                              |
| :--------------------- | :------------ | :--------------------------------------------------------------------------- |
| `load_data.py`         | Python Script | Initial loading of OTU table and metadata.                                   |
| `eda.py`               | Python Script | Script for Exploratory Data Analysis.                                        |
| `modeleval.py`         | Python Script | Script for model training (Logistic, Random Forest, XGBoost) and evaluation. |
| `report.py`            | Python Script | Generates the final analysis summary and conclusions.                        |
| `app.py`               | Python Script | Streamlit dashboard code for interactive visualization.                      |
| `processed_target.csv` | CSV           | The target variable data, containing **SLEEP\_DURATION** categories.         |
| `model_results.csv`    | CSV           | Stores the quantitative results of the models (e.g., Accuracy, ROC-AUC).     |
