# 🩺 Diabetes Risk Predictor
**An End-to-End Machine Learning Pipeline: From Neon PostgreSQL to Streamlit Deployment**

## 🚀 Project Overview
This project is a clinical decision-support tool designed to predict the risk of diabetes in patients based on diagnostic measurements. The primary goal was to prioritize **Recall (Sensitivity) > 0.85**, ensuring that potential cases are flagged for clinical follow-up even at the cost of higher False Positives.

### 📊 Key Performance Metrics (Test Set)
*   **Recall:** 0.844 (Optimized via custom probability threshold of 0.42) ✅
*   **Accuracy:** 86.4% ✅
*   **AUC-ROC:** 0.938 ✅
*   **F1-Score:** 78.5% ✅

---

## 🛠️ Technical Stack
*   **Database:** Neon PostgreSQL (Cloud-hosted)
*   **Language:** Python 3.12+
*   **Libraries:** Scikit-Learn, Pandas, NumPy, SQLAlchemy, Imbalanced-Learn (SMOTE)
*   **Deployment:** Streamlit Community Cloud
*   **Model:** Logistic Regression (Optimized via GridSearchCV)

---

## ⚙️ Data Pipeline Features
1.  **Data Ingestion:** Securely fetching raw data from Neon SQL using SQLAlchemy.
2.  **Data Preparation:** 
    *   Imputation of "impossible zeros" (Glucose, BMI, etc.) using median values.
    *   Z-Score outlier detection.
    *   Ordinal Encoding for BMI categories and One-Hot Encoding for Blood Types.
3.  **Feature Engineering:** Created a custom `insulin_glucose_ratio` to capture insulin resistance signals.
4.  **Class Imbalance:** Applied **SMOTE** to the training set only to handle the 70/30 class split.
5.  **Scaling:** Utilized **RobustScaler** to mitigate the impact of medical outliers.

---

## 🖥️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com
