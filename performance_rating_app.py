import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🌟 Performance Rating Predictor", layout="centered")

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{b64_string}"

background_image_url = get_base64_image("performance-rating-1200x480.jpg")

st.markdown(f"""<style>
.stApp {{background: linear-gradient(rgba(255,255,255,0.65), rgba(255,255,255,0.65)), url("{background_image_url}");
background-size: cover; background-repeat: no-repeat; background-position: center; background-attachment: fixed;}}
.main > div {{background-color: rgba(255, 255, 255, 0.95);padding: 2rem;border-radius: 16px;box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);}}

h1 {{color: #222222 !important;font-size: 2.4rem !important;font-weight: 800 !important;text-align: center;margin-bottom: 1rem;}}

.stMarkdown p {{font-size: 18px !important;font-weight: 700 !important;color: #000000 !important;text-align: center;margin-bottom: 2rem;}}
label, .stSelectbox label, .stSlider label, .stNumberInput label {{font-weight: 900 !important;color: #111111 !important;font-size: 17px !important;}}

.stButton button {{background-color: #ff4b4b;color: white;font-weight: bold;padding: 0.5rem 1.2rem;font-size: 16px;border-radius: 10px;
transition: background-color 0.3s ease;}}

.stButton button:hover {{background-color: #d63a3a;}}</style>""", unsafe_allow_html=True)


model = joblib.load("performance_rf_model.pkl")
scaler = joblib.load("performance_scaler.pkl")
top15_features = joblib.load("performance_top15_features.pkl")

st.title("📈 Employee Performance Rating Predictor")
st.markdown("Predict whether an employee is likely to receive a performance rating of 3 or 4.", unsafe_allow_html=True)

with st.form("performance_form"):
  st.markdown("### 🧾 <span style='color:#4b91ff'>Employee Performance Inputs</span>", unsafe_allow_html=True)

  PercentSalaryHike = st.slider("💸 Percent Salary Hike", 0, 100, 15)
  HourlyRate = st.slider("⏰ Hourly Rate", 30, 100, 50)
  MonthlyIncome = st.number_input("💰 Monthly Income", min_value=1000, max_value=20000, value=5000)
  DailyRate = st.number_input("📊 Daily Rate", min_value=100, max_value=1500, value=800)
  MonthlyRate = st.number_input("📈 Monthly Rate", min_value=1000, max_value=30000, value=10000)
  Age = st.slider("🎂 Age", 18, 60, 30)
  TotalWorkingYears = st.slider("📅 Total Working Years", 0, 40, 10)
  DistanceFromHome = st.slider("🚗 Distance From Home (km)", 0, 50, 10)
  YearsAtCompany = st.slider("🏢 Years at Company", 0, 40, 5)
  YearsWithCurrManager = st.slider("👔 Years with Current Manager", 0, 20, 5)
  NumCompaniesWorked = st.slider("🏢 Number of Companies Worked", 0, 10, 2)
  YearsSinceLastPromotion = st.slider("📉 Years Since Last Promotion", 0, 15, 3)
  YearsInCurrentRole = st.slider("🔨 Years in Current Role", 0, 15, 3)
  TrainingTimesLastYear = st.slider("📚 Trainings Attended Last Year", 0, 10, 2)
  EnvironmentSatisfaction = st.slider("🌿 Environment Satisfaction", 1, 4, 3)

  submit = st.form_submit_button("🔍 Predict Performance")

if submit:
  input_dict = {"PercentSalaryHike": PercentSalaryHike,
      "HourlyRate": HourlyRate,
      "MonthlyIncome": MonthlyIncome,
      "DailyRate": DailyRate,
      "MonthlyRate": MonthlyRate,
      "Age": Age,
      "TotalWorkingYears": TotalWorkingYears,
      "DistanceFromHome": DistanceFromHome,
      "YearsAtCompany": YearsAtCompany,
      "YearsWithCurrManager": YearsWithCurrManager,
      "NumCompaniesWorked": NumCompaniesWorked,
      "YearsSinceLastPromotion": YearsSinceLastPromotion,
      "YearsInCurrentRole": YearsInCurrentRole,
      "TrainingTimesLastYear": TrainingTimesLastYear,
      "EnvironmentSatisfaction": EnvironmentSatisfaction}

  input_df = pd.DataFrame([input_dict])
  input_scaled = pd.DataFrame(scaler.transform(input_df), columns=top15_features)
  prediction = model.predict(input_scaled)[0]

  st.markdown("---")
  if prediction == 4:
      st.markdown("<div class='result-box' style='background-color:#e6ffed; color:#006600;'>✅ <b>Prediction:</b> The employee is likely to receive a <u>Performance Rating of 4</u>.</div>", unsafe_allow_html=True)
  else:
      st.markdown("<div class='result-box' style='background-color:#fff7e6; color:#cc7a00;'>ℹ️ <b>Prediction:</b> The employee is likely to receive a <u>Performance Rating of 3</u>.</div>", unsafe_allow_html=True)
