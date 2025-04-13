💼 HR Analytics: Employee Attrition & Performance Prediction
📊 Domain: Human Resource Analytics
🎯 Objectives:

Predict Employee Attrition

Predict Employee Performance Rating

This end-to-end machine learning project focuses on building predictive models to help HR departments make proactive decisions by analyzing historical employee data. The project includes data cleaning, EDA, feature engineering, model building, evaluation, and deployment via a user-friendly Streamlit web app.

🔍 Problem Statement
Organizations face significant challenges due to:

High attrition, which increases recruitment and training costs

Inconsistent performance evaluation, which affects promotions and engagement

This project solves both problems by:

Predicting which employees are likely to leave the company

Predicting which employees are likely to perform well (ratings 3 vs 4)

Empowering HR with data-driven decision tools

🧠 Skills Applied
Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

Outlier Detection & Capping

Feature Engineering & Selection

Class Balancing (SMOTE, Under-sampling)

Machine Learning & Cross Validation

Model Deployment with Streamlit

Serialization of Models and Artifacts

🚀 Project Workflow
📁 1. Data Understanding & Cleaning
Handled missing and duplicate values

Classified columns into:

Continuous Numerical

Numeric but Categorical

Pure Categorical

Dropped unnecessary or constant columns

Checked uniqueness and consistency across features

📊 2. Exploratory Data Analysis (EDA)
🔹 Univariate Analysis
Boxplots and distribution plots to detect outliers

Capped extreme values in continuous features

Analyzed skewness before and after capping

Studied distribution of categorical columns

🔹 Bivariate Analysis
Target (Attrition / Performance Rating) vs:

Continuous numerical features

Numeric-but-categorical features

Categorical features

Performed Chi-Square Tests for independence

Assessed correlation with targets for all feature types

🧬 3. Feature Engineering & Encoding
Created meaningful features (e.g., tenure buckets, engagement score)

Applied:

Binary Encoding for binary features

One-Hot Encoding for multi-class features

⚖️ 4. Handling Imbalanced Data
For Attrition prediction: applied SMOTE Oversampling

For Performance Rating prediction: applied Random Undersampling

📏 5. Feature Scaling & Splitting
Scaled features using MinMaxScaler

Split datasets into train and test sets for model training

🌲 6. Model Building & Evaluation
✔ Attrition Prediction
Trained models: Random Forest, XGBoost, SVM, KNN, Logistic Regression, Decision Tree

Selected Top 15 features using feature importance

Evaluated models using:

Accuracy, Precision, Recall, F1-Score, ROC-AUC

Cross-validated for robustness

📈 Performance Rating Prediction (3 vs 4)
Similar workflow with undersampled dataset

Top 15 features identified from Random Forest

Trained and compared all models on balanced data

💾 7. Model Saving
Saved key artifacts:

Final trained models for both tasks

Scaler

Encoders

Top 15 feature sets

Full encoded feature list

🎯 8. Streamlit App Deployment
Developed a sleek and intuitive Streamlit Web App for both use cases:

🔍 Attrition Prediction Module

Predicts whether an employee is at risk of leaving

Visual display of prediction and input summary

📊 Performance Rating Prediction Module

Predicts if an employee will be rated 3 or 4

Built for proactive appraisal planning

🔧 Backend uses serialized .pkl files for model inference
💡 CSS custom styling applied for a beautiful UI

🧰 Tech Stack
Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn)

Streamlit (Web App)

Pickle (Model and artifact saving)

✅ Results Summary
Achieved high F1-Scores > 90% on balanced data for both tasks

Reliable cross-validation performance

Deployable, real-time app for HR decision-making

