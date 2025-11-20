# Employee-Attrition-Prediction
A machine learning model for  Employee Attrition Prediction
This project predicts Employee Attrition using machine learning techniques. It includes a full end-to-end pipeline: data preprocessing, feature selection, imbalance handling, model building, hyperparameter tuning, and a Streamlit web application for real-time predictions.

📌 Project Overview

Employee attrition is a major concern for organizations, and this application helps identify employees who are at risk of leaving.
This repository contains:

🧹 Preprocessing & Feature Engineering

📊 Exploratory Data Analysis

⚖️ Imbalance Handling

🤖 Model Training with Multiple Algorithms

🌲 Final Random Forest Model with Threshold Tuning

💾 Saved Model, Scaler, Encoder & Feature List

🖥️ Interactive Streamlit Dashboard

High-Risk Employees Table

High-Satisfaction Employees Table

Full Employee Attrition Predictor

🏗️ Project Workflow
1️⃣ Data Understanding & Cleaning

Checked shape, info, describe(), duplicate rows, valid zeros, and outliers.

Removed irrelevant/unwanted columns.

2️⃣ Encoding & Target Analysis

Encoded categorical variables.

Split into feature matrix (X) and target (y).

Identified that attrition dataset is imbalanced.

3️⃣ Train-Test Split & Feature Selection

Applied multiple techniques to select strong predictors:

Correlation Analysis

Chi-Square Test

MI Classifier

Random Forest Feature Importance

Selected final 15 best features.

4️⃣ Scaling

Applied StandardScaler only on continuous numerical features.

Excluded one-hot and ordinal encoded features.

Verified scaling correctness.

5️⃣ Handling Data Imbalance

Tried:

SMOTETomek

RandomSampler

Final choice: RandomSampler with custom threshold tuning, because it gave better recall & precision for class 1.

6️⃣ Model Building

Tried:

Logistic Regression

KNN

Decision Tree

Random Forest

XGBoost

Ensemble approach

Best performer: Random Forest
Tuned using GridSearchCV.

Final improvement: Threshold adjustment → increased precision & recall for class 1.

7️⃣ Saving Artifacts

Saved using joblib:

Encoder

Scaler

Selected feature list

Final model

8️⃣ Streamlit Application

A clean UI with two sections:

🔼 Upper Section

Two tables:

High-Risk Employees

High-Satisfaction Employees

A "Show" button reveals the top 5 employees in each category.

🔽 Lower Section

Employee Attrition Predictor form:

User inputs all required feature values

On "Predict" → Model outputs:
🟡 Probability employee will leave
🟢 Not likely to leave / 🔴 Likely to leave


├── model/
│   ├── final_rf_model.pkl
│   ├── scaler.pkl
│   ├── le_overtime.pkl  (encoder)
│   ├── model_columns.pkl
│
├── app/
│   ├── emp_attrition.py              
│
├── data
│   ├── Employee-Attrition - Employee-Attrition.csv
│
├── README.md
├── requirements.txt

▶️ How to Run the Streamlit App

cd <Employee_Attrition>
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the app
streamlit run emp_attrition.py

🧪 Model Evaluation Summary

Best model → Random Forest

Improved class-1 performance using:

Random Sampling

Threshold tuning

Final model gives balanced accuracy, precision, and recall.

🛠️ Tech Stack

Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn
Streamlit
Joblib

⭐ Future Enhancements

Add SHAP explainability
Add database integration
Add employee insights dashboard






High-Satisfaction Employees Table

Full Employee Attrition Predictor
