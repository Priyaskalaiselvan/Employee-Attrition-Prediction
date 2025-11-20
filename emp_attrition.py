import streamlit as st
import pandas as pd
import joblib

model = joblib.load("final_rf_model.pkl")
scaler = joblib.load("final_scaler.pkl")
le_overtime = joblib.load("le_overtime.pkl")
columns = joblib.load("model_columns.pkl")

columns=list(columns)


high_risk_df = pd.DataFrame({
    "Emp_id": [1, 4, 19, 27, 31],
    'Age': [41,37,28,36,34],
    'DailyRate': [1102,1373,103,1218,699],
    'DistanceFromHome': [1,2,24,9,6],
    'JobLevel': [2,1,1,1,1],
    'JobSatisfaction': [4,3,3,1,1],
    'MonthlyIncome': [5993,2090,2028,3407,2960],
    'MonthlyRate': [19479,2396,12947,6986,17102],
    'OverTime': ['yes','yes','yes','no','no'],
    'StockOptionLevel': [0,0,0,0,0],
    'PercentSalaryHike': [11,15,14,23,11],
    'TotalWorkingYears':[8,7,6,10,8] ,
    'YearsAtCompany':[6,0,4,5,4] ,
    'YearsInCurrentRole':[4,0,2,3,2] ,
    'YearsWithCurrManager': [5,0,3,3,3],
    'MaritalStatus_Single': ['s','s','s','s','s']
    
})

high_satisfaction_df = pd.DataFrame({
    "Emp_id": [2, 7, 8, 10, 12],
    'Age': [49,27,32,59,38],
    'DailyRate': [279,591,1005,1324,216],
    'DistanceFromHome': [8,2,2,3,23],
    'JobLevel': [2,1,1,1,3],
    'JobSatisfaction': [2,2,4,1,3],
    'MonthlyIncome': [5130,3468,3068,2670,9526],
    'MonthlyRate': [24907,16632,11864,9964,8787],
    'OverTime': ['no','no','no','yes','no'],
    'StockOptionLevel': [1,1,0,3,1],
    'PercentSalaryHike': [23,12,13,20,21],
    'TotalWorkingYears':[10,6,8,12,10] ,
    'YearsAtCompany':[10,2,7,1,1] ,
    'YearsInCurrentRole':[7,2,7,0,0] ,
    'YearsWithCurrManager': [7,2,6,0,0],
    'MaritalStatus_Single': ['m','m','s','m','s']
    
})

st.set_page_config(page_title="Employee Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center;'>Employee Attrition Analysis</h1>",unsafe_allow_html=True)
st.markdown("Employee Insights Tables")
left, right = st.columns(2)
with left:
    st.subheader("🚨 High-Risk Employees")
    if st.button("Show"):
        st.dataframe(high_risk_df)
with right:
    st.subheader("😊 Employees With High Satisfaction")
    if st.button("Show "):
        st.dataframe(high_satisfaction_df)

st.markdown("---")

#st.markdown("## 🔮 Employee Attrition Prediction")

st.title("Employee Attrition Prediction")
st.header("📃Enter the following details to know about employee attrition")

Age = st.slider("Age", 18, 60, 30)
DailyRate = st.number_input("Daily Rate", 100, 1500, 500)
DistanceFromHome = st.slider("Distance From Home", 1, 50, 10)
JobLevel = st.slider("Job Level", 1, 5, 2)
JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
MonthlyIncome = st.number_input("Monthly Income", 1000, 50000, 15000)
MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 15000)
OverTime = st.selectbox("OverTime", ["Yes", "No"])
StockOptionLevel = st.slider("Stock Option Level", 0, 80, 5)
PercentSalaryHike = st.slider("Percent Salary Hike", 0, 30, 15)
TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
YearsInCurrentRole = st.slider("Years in Current Role", 0, 20, 3)
YearsWithCurrManager = st.slider("Years With Current Manager", 0, 20, 3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

OverTime_encoded = le_overtime.transform([OverTime])[0]
MaritalStatus_Single = 1 if MaritalStatus == "Single" else 0

input_dict = {
    'Age': Age,
    'DailyRate': DailyRate,
    'DistanceFromHome': DistanceFromHome,
    'JobLevel': JobLevel,
    'JobSatisfaction': JobSatisfaction,
    'MonthlyIncome': MonthlyIncome,
    'MonthlyRate': MonthlyRate,
    'OverTime': OverTime_encoded,
    'StockOptionLevel': StockOptionLevel,
    'PercentSalaryHike': PercentSalaryHike,
    'TotalWorkingYears': TotalWorkingYears,
    'YearsAtCompany': YearsAtCompany,
    'YearsInCurrentRole': YearsInCurrentRole,
    'YearsWithCurrManager': YearsWithCurrManager,
    'MaritalStatus_Single': MaritalStatus_Single
}

input_df = pd.DataFrame([input_dict])



# Reorder columns to match training
input_df = input_df[columns]

# Scale ONLY the columns the scaler was trained on
scaled_input = scaler.transform(input_df[scaler.feature_names_in_])

# Convert scaled output back into a DataFrame
scaled_df = pd.DataFrame(scaled_input, columns=scaler.feature_names_in_)

# Add the unscaled categorical columns
for col in input_df.columns:
    if col not in scaler.feature_names_in_:
        scaled_df[col] = input_df[col]

# Reorder final dataframe to match model training order
scaled_df = scaled_df[columns]

# Predict
if st.button("🔮 Predict Attrition"):
    prob = model.predict_proba(scaled_df)[0][1]
    prediction = 1 if prob >= 0.4 else 0

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Employee is LIKELY to Leave (Attrition)\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Employee is NOT likely to leave\nProbability: {prob:.2f}")




