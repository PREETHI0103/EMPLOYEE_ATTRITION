# 💼 HR Analytics: Employee Attrition & Performance Prediction

## 📊 Domain  
Human Resource Analytics

## 🎯 Project Objectives
- 🚪 **Predict Employee Attrition**  
- ⭐ **Predict Employee Performance Rating** (Classify ratings 3 vs 4)  

---

## 🚀 Project Overview

This end-to-end ML project equips HR teams to proactively manage workforce challenges by leveraging historical employee data. The solution spans **data cleaning**, **EDA**, **feature engineering**, **class balancing**, **model training with cross-validation and comparison**, and deployment as a **Streamlit web app**.

---

## 🔍 Problem Statement

Organizations face costly issues including:

- 🔄 **High attrition rates** leading to increased recruitment & training expenses  
- 🎯 **Inconsistent performance evaluations** hindering fair promotions & employee engagement  

Our solution predicts:

- Which employees are at risk of leaving  
- Which employees are likely to achieve top performance ratings (3 vs 4)  

Empowering HR with data-driven insights for better workforce planning.  

---

## 🧠 Skills & Techniques Applied

- 🧹 Data Cleaning & Missing Value Handling  
- 📈 Exploratory Data Analysis (EDA) & Outlier Capping  
- 🛠 Feature Engineering & Encoding (Binary & One-Hot)  
- ⚖️ Handling Imbalanced Data (SMOTE & Random Undersampling)  
- 🔍 Model Training, Hyperparameter Tuning, & Cross-Validation  
- 🏆 Model Comparison & Best Model Selection  
- 💾 Model & Artifact Serialization using **joblib**  
- 🌐 Streamlit App Deployment with Custom CSS Styling  

---

## 📋 Detailed Workflow

### 1️⃣ Data Understanding & Cleaning  
- Identified & handled missing/duplicate data  
- Classified features as:  
  - Continuous Numerical  
  - Numeric but Categorical  
  - Pure Categorical  
- Removed redundant and constant-value columns  
- Verified feature consistency and uniqueness  

### 2️⃣ Exploratory Data Analysis (EDA)   
  - Visualized distributions with boxplots, histograms  
  - Detected and capped outliers for robust modeling  
  - Analyzed relationships between target (Attrition / Performance) and features  
  - Performed Chi-Square tests for categorical dependencies  
  - Computed correlations for numeric features  

### 3️⃣ Feature Engineering & Encoding  
- Applied encoding techniques:  
  - Binary Encoding for binary features  
  - One-Hot Encoding for multi-class categorical variables  

### 4️⃣ Handling Class Imbalance  
- **Attrition Prediction:** Applied **SMOTE** oversampling to balance minority class  
- **Performance Rating Prediction:** Used **Random Undersampling** for balanced training  

### 5️⃣ Feature Scaling & Data Splitting  
- Scaled features using **MinMaxScaler** to normalize feature ranges  
- Split data into train/test sets ensuring stratified distribution  

### 6️⃣ Model Training, Cross-Validation & Comparison  
- Trained multiple candidate models:  
  - Random Forest, XGBoost, SVM, K-Nearest Neighbors, Logistic Regression, Decision Tree  
- Used **Stratified K-Fold Cross-Validation** for robust performance estimation  
- Evaluated models on:  
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Compared models and selected the best performer based on highest F1-Score and balanced metrics  

### 7️⃣ Model Serialization  
- Saved the best performing models and preprocessing artifacts using **joblib** for efficient loading:  
  - Model files (`.joblib`)  
  - Scaler  
  - Encoders  
  - Selected top features list  

### 8️⃣ Streamlit Web App Deployment  
- Developed an interactive app featuring:  
  - **Attrition Prediction Module:** Risk prediction with user-friendly input forms & results display  
  - **Performance Rating Prediction Module:** Predicts if an employee will get rating 3 or 4  
- Backend uses serialized `.joblib` models for instant inference  
- Applied sleek custom CSS for an engaging UI experience  

---

## 🧰 Tech Stack

| Category           | Tools & Libraries                                   |
|--------------------|----------------------------------------------------|
| Programming        | Python (Pandas, NumPy, Scikit-learn, XGBoost)      |
| Data Imbalance     | Imbalanced-learn (SMOTE, Random Undersampling)      |
| Model Serialization| joblib                                              |
| Visualization      | Matplotlib, Seaborn                                 |
| Web App            | Streamlit                                          |

---

## ✅ Key Results Summary

- 🚀 Achieved **F1-Scores above 90%** on balanced test sets for both prediction tasks  
- 🔄 Stable and consistent cross-validation results proving model robustness  
- ⚡ Real-time, deployable app enabling HR to make informed, proactive decisions  

---

## 🧑‍💻 Author

Built with ❤️ by **[PREETHI S]**

---

## 🏷️ Tags

`#HRAnalytics` `#EmployeeAttrition` `#EmployeePerformance` `#MachineLearning` `#DataScience` `#Python` `#ScikitLearn` `#XGBoost` `#SMOTE` `#ImbalancedLearning` `#Joblib` `#Streamlit` `#ModelDeployment` `#PredictiveModeling` `#CrossValidation` `#FeatureEngineering` `#DataPreprocessing` `#HumanResources` `#AIForHR`


