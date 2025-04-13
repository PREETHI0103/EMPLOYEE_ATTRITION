import streamlit as st
import pandas as pd
import joblib
import base64

st.set_page_config(page_title="🌟 Employee Attrition Predictor", layout="centered")

background_image_url = "https://i.imgur.com/ha4HytZ.png"

st.markdown(f"""<style>.stApp {{background: linear-gradient(rgba(255,255,255,0.65), rgba(255,255,255,0.65)),url("{background_image_url}");
background-size: cover; background-repeat: no-repeat; background-position: center; background-attachment: fixed;}}

.main > div {{background-color: rgba(255, 255, 255, 0.95);padding: 2rem;border-radius: 16px;box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);}}

h1 {{color: #222222 !important;font-size: 2.4rem !important;font-weight: 800 !important;text-align: center;margin-bottom: 1rem;}}

.stMarkdown p {{font-size: 18px !important;font-weight: 700 !important;color: #000000 !important;text-align: center;margin-bottom: 2rem;}}
label, .stSelectbox label, .stSlider label, .stNumberInput label {{font-weight: 900 !important;color: #111111 !important;font-size: 17px !important;}}

.stButton button {{background-color: #ff4b4b;color: white;font-weight: bold;padding: 0.5rem 1.2rem;font-size: 16px;border-radius: 10px;
transition: background-color 0.3s ease;}}

.stButton button:hover {{background-color: #d63a3a;}}</style>""", unsafe_allow_html=True)

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
top15_features = joblib.load("top15_features.pkl")
model_features = joblib.load("model_features.pkl")
ohe = joblib.load("onehot_encoder.pkl")


st.title("🔍 Employee Attrition Predictor")
st.markdown( "<i>Fill in the details below to predict whether an employee is likely to leave the company.</i>",unsafe_allow_html=True)

with st.form("attrition_form"):
  st.markdown("### 🔧 <span style='color:#ff4b4b'>Employee Details</span>", unsafe_allow_html=True)

  StockOptionLevel = st.selectbox("📦 **Stock Option Level**", [0, 1, 2, 3])
  MonthlyIncome = st.number_input("💰 **Monthly Income**", min_value=1000, max_value=20000, value=5000)
  JobSatisfaction = st.slider("😊 **Job Satisfaction**", 1, 4, 3)
  YearsWithCurrManager = st.slider("🧑‍💼 **Years with Current Manager**", 0, 20, 5)
  RelationshipSatisfaction = st.slider("👥 **Relationship Satisfaction**", 1, 4, 3)
  JobInvolvement = st.slider("🔨 **Job Involvement**", 1, 4, 3)
  YearsAtCompany = st.slider("🏢 **Years at Company**", 0, 40, 5)
  TotalWorkingYears = st.slider("📅 **Total Working Years**", 0, 40, 10)
  MonthlyRate = st.number_input("📈 **Monthly Rate**", min_value=1000, max_value=30000, value=10000)
  Age = st.slider("🎂 **Age**", 18, 60, 30)
  DailyRate = st.number_input("📊 **Daily Rate**", min_value=100, max_value=1500, value=800)

  BusinessTravel = st.selectbox("✈️ **Business Travel**", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
  Department = st.selectbox("🏢 **Department**", ["Research & Development", "Sales", "Human Resources"])
  EducationField = st.selectbox("🎓 **Education Field**", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
  JobRole = st.selectbox("🧑‍💻 **Job Role**", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                                                "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
  MaritalStatus = st.selectbox("💍 **Marital Status**", ["Single", "Married", "Divorced"])

  submit = st.form_submit_button("🔍 Predict Attrition")

if submit:
  input_dict = {"StockOptionLevel": StockOptionLevel,
    "MonthlyIncome": MonthlyIncome,
    "JobSatisfaction": JobSatisfaction,
    "YearsWithCurrManager": YearsWithCurrManager,
    "RelationshipSatisfaction": RelationshipSatisfaction,
    "JobInvolvement": JobInvolvement,
    "YearsAtCompany": YearsAtCompany,
    "TotalWorkingYears": TotalWorkingYears,
    "MonthlyRate": MonthlyRate,
    "Age": Age,
    "DailyRate": DailyRate,
    "BusinessTravel": BusinessTravel,
    "Department": Department,
    "EducationField": EducationField,
    "JobRole": JobRole,
    "MaritalStatus": MaritalStatus}

  input_df = pd.DataFrame([input_dict])

  multi_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
  encoded_cats = ohe.transform(input_df[multi_cols])
  input_df = input_df.drop(columns=multi_cols).reset_index(drop=True)
  input_encoded = pd.concat([input_df, encoded_cats], axis=1)

  for col in model_features:
      if col not in input_encoded.columns:
          input_encoded[col] = 0
  input_encoded = input_encoded[model_features]


  input_scaled = pd.DataFrame(scaler.transform(input_encoded), columns=model_features)
  input_final = input_scaled[top15_features]

  prediction = model.predict(input_final)[0]

  st.markdown("---")
  if prediction == 1:
      st.markdown(
          "<div class='result-box' style='background-color:#ffe6e6; color:#cc0000;'>⚠️ <b>Prediction:</b> The employee is <u>likely to leave</u> the company.</div>",
          unsafe_allow_html=True)
  else:
      st.markdown(
          "<div class='result-box' style='background-color:#e6ffed; color:#006600;'>✅ <b>Prediction:</b> The employee is <u>likely to stay</u> with the company.</div>",
          unsafe_allow_html=True)
