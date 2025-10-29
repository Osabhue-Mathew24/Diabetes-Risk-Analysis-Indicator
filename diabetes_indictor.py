import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============== PAGE CONFIGURATION ==============
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# ============== TITLE SECTION ==============
st.markdown("<h1 style='color: #0077b6; text-align: center; font-size: 60px; font-family: Monospace'>DIABETES RISK ANALYSIS PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='margin: -30px; color: #00b4d8; text-align: center; font-family: Serif;'>Built by MATTHEW OSABHUE</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

image_path = "indictorimage.png"
if os.path.exists(image_path):
    try:
        st.image(image_path, use_container_width=True)
    except TypeError:
        st.image(image_path, use_column_width=True)
else:
    st.warning("⚠️ Image file not found. Please check 'indictorimage.png' in your app folder.")
st.divider()

# ============== BACKGROUND SECTION ==============
st.markdown("<h2 style='color: #0096c7; text-align: center; font-family: montserrat;'>Background of Study</h2>", unsafe_allow_html=True)
st.markdown("""
Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels. 
Early prediction and detection of diabetes risk is critical for prevention and effective management. 
This project uses machine learning to estimate an individual's risk score based on key health indicators such as glucose levels, HbA1c, BMI, age, and family history. 
By analyzing these metrics, healthcare practitioners and individuals can better understand potential risks and take preventive measures.
""")
st.divider()

# ============== SIDEBAR USER INPUT ==============
user_icon = "usericon.png"
if os.path.exists(user_icon):
    try:
        st.sidebar.image(user_icon, use_container_width=True, caption="Welcome User")
    except TypeError:
        st.sidebar.image(user_icon, use_column_width=True, caption="Welcome User")
else:
    st.sidebar.warning("⚠️ 'usericon.png' not found.")

st.sidebar.markdown("### Enter Your Health Data")

stage = st.sidebar.selectbox('Diabetes Stage (if known)', ['Normal', 'Prediabetes', 'Diabetes'], index=0)
hba1c = st.sidebar.number_input('HbA1c (%)', min_value=3.5, max_value=15.0, value=5.6, step=0.1)
gluc_fast = st.sidebar.number_input('Fasting Glucose (mg/dL)', min_value=60.0, max_value=300.0, value=95.0, step=1.0)
gluc_post = st.sidebar.number_input('Postprandial Glucose (mg/dL)', min_value=70.0, max_value=400.0, value=120.0, step=1.0)
age = st.sidebar.number_input('Age (years)', min_value=10, max_value=190, value=35, step=1)
fh_diabetes = st.sidebar.selectbox('Family History of Diabetes', ['No', 'Yes'], index=0)

st.sidebar.markdown("#### Do you know your BMI?")
know_bmi = st.sidebar.radio('Select an option:', ['Yes', 'No'], index=1)

if know_bmi == 'Yes':
    bmi = st.sidebar.number_input('Enter your BMI', min_value=10.0, max_value=60.0, value=24.5, step=0.1)
else:
    st.sidebar.markdown("### Calculate BMI")
    weight = st.sidebar.number_input('Weight (kg)', min_value=20.0, max_value=200.0, value=70.0, step=0.5)
    height = st.sidebar.number_input('Height (m)', min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    bmi = round(weight / (height ** 2), 2)
    st.sidebar.info(f"Your calculated BMI is: **{bmi}**")

# ============== USER INPUT DATAFRAME ==============
user_input = {
    'stage': [stage],
    'hba1c': [hba1c],
    'gluc_post': [gluc_post],
    'gluc_fast': [gluc_fast],
    'fh_diabetes': [fh_diabetes],
    'age': [age],
    'bmi': [bmi]
}
input_data = pd.DataFrame(user_input)

st.subheader("User Input Summary")
st.dataframe(input_data, use_container_width=True)
st.divider()

# ============== LOAD OR TRAIN MODEL ==============
model_path = "Diabetes_indicator.pkl"
dataset_path = "diabetes_dataset.csv"

if not os.path.exists(model_path):
    st.warning("⚠️ Model file not found. Attempting to train automatically...")

    # --- If dataset missing, generate dummy dataset ---
    if not os.path.exists(dataset_path):
        st.info("📊 No dataset found. Creating a small dummy dataset for demo use.")
        np.random.seed(42)
        dummy_data = pd.DataFrame({
            'stage': np.random.randint(0, 3, 200),
            'hba1c': np.random.uniform(4.5, 10.0, 200),
            'gluc_post': np.random.uniform(80, 300, 200),
            'gluc_fast': np.random.uniform(70, 250, 200),
            'fh_diabetes': np.random.randint(0, 2, 200),
            'age': np.random.randint(18, 80, 200),
            'bmi': np.random.uniform(18, 40, 200),
            'diabetes': np.random.randint(0, 2, 200)
        })
        dummy_data.to_csv(dataset_path, index=False)
        st.success("✅ Dummy dataset created.")

    # --- Load dataset and train model ---
    data = pd.read_csv(dataset_path)
    if 'fh_diabetes' in data.columns:
        data['fh_diabetes'] = data['fh_diabetes'].map({'No': 0, 'Yes': 1}).fillna(data['fh_diabetes'])
    if 'stage' in data.columns:
        data['stage'] = data['stage'].map({'Normal': 0, 'Prediabetes': 1, 'Diabetes': 2}).fillna(data['stage'])

    target_col = 'diabetes'
    if target_col not in data.columns:
        st.error(f"❌ Target column '{target_col}' not found. Add it to your dataset.")
        st.stop()

    x = data.drop(columns=[target_col])
    y = data[target_col]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(xtrain, ytrain)

    joblib.dump(model, model_path)
    st.success("✅ Model retrained and saved successfully!")
else:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")

# ============== ENCODE INPUTS & PREDICT ==============
input_data['fh_diabetes'] = input_data['fh_diabetes'].map({'No': 0, 'Yes': 1})
input_data['stage'] = input_data['stage'].map({'Normal': 0, 'Prediabetes': 1, 'Diabetes': 2})

predict_button = st.button("🔍 Predict Diabetes Risk Score")

if predict_button:
    try:
        expected_features = getattr(model, "feature_names_in_", input_data.columns)
        missing_cols = set(expected_features) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[expected_features]

        if hasattr(model, "predict_proba"):
            risk_score = model.predict_proba(input_data)[:, 1][0]
        else:
            risk_score = model.predict(input_data)[0]

        st.success(f"🎯 **Predicted Diabetes Risk Score:** {risk_score:.2%}")

        if risk_score < 0.4:
            st.info("🟢 **Low Risk** — Maintain a healthy lifestyle and balanced diet.")
        elif risk_score < 0.7:
            st.warning("🟠 **Moderate Risk** — Monitor glucose levels and consult a healthcare provider.")
        else:
            st.error("🔴 **High Risk** — Medical attention and lifestyle adjustments recommended.")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Risk Level Interpretation")
st.markdown("""
- **Low Risk (<40%)**: Maintain a healthy lifestyle and diet.  
- **Moderate Risk (40 - 69%)**: Monitor glucose levels and consult your doctor.  
- **High Risk (≥70%)**: Requires medical attention and lifestyle changes.
""")

st.divider()
st.caption("© 2025 | Diabetes Risk Analysis Predictor by Matthew Osabhue")
