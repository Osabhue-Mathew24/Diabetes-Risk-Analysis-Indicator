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

st.image('pngwing.com (1)dia_indicator.png', use_container_width=True)
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
st.sidebar.image('pngwing.com_user_icon.png', caption='Welcome User')
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

# ============== DATAFRAME OF USER INPUT ==============
user_input = {
    'stage': [stage],
    'hba1c': [hba1c],
    'gluc_post': [gluc_post],
    'gluc_fast': [gluc_fast],
    'fh_diabetes': [fh_diabetes],
    'age': [age],
    'bmi': [bmi]
}
input_df = pd.DataFrame(user_input)
st.subheader("User Input Summary")
st.dataframe(input_df, use_container_width=True)
st.divider()

# ============== LOAD OR TRAIN MODEL ==============
model_path = r"C:\Users\DELL\OneDrive\Desktop\Machine_Learning\dataset_project\Diabetes_indicator.pkl"

if not os.path.exists(model_path):
    st.warning("⚠️ Model file not found. Retraining model automatically...")

    # Try to load dataset for training
    data_path = r"C:\Users\DELL\OneDrive\Desktop\Machine_Learning\dataset_project\diabetes_dataset.csv"
    if not os.path.exists(data_path):
        st.error(f"❌ Dataset not found at {data_path}. Please add the dataset for training.")
        st.stop()

    # === TRAIN A SIMPLE MODEL ===
    df = pd.read_csv(data_path)
    st.info("Training a new model using dataset...")

    # Encode categorical features
    if 'fh_diabetes' in df.columns:
        df['fh_diabetes'] = df['fh_diabetes'].map({'No': 0, 'Yes': 1})
    if 'stage' in df.columns:
        df['stage'] = df['stage'].map({'Normal': 0, 'Prediabetes': 1, 'Diabetes': 2})

    # Define features and target
    target_col = 'diabetes'  # change if your dataset target column differs
    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' not found in dataset.")
        st.stop()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    st.success(f"✅ Model retrained and saved as {model_path}")
else:
    model = joblib.load(model_path)

# ============== ENCODE INPUTS & PREDICT ==============
input_df['fh_diabetes'] = input_df['fh_diabetes'].map({'No': 0, 'Yes': 1})
input_df['stage'] = input_df['stage'].map({'Normal': 0, 'Prediabetes': 1, 'Diabetes': 2})

predict_button = st.button("🔍 Predict Diabetes Risk Score")

if predict_button:
    try:
        # Ensure feature alignment
        expected_features = getattr(model, "feature_names_in_", input_df.columns)
        missing_cols = set(expected_features) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[expected_features]

        # Prediction
        if hasattr(model, "predict_proba"):
            risk_score = model.predict_proba(input_df)[:, 1][0]
        else:
            risk_score = model.predict(input_df)[0]

        # Display result
        st.success(f"🎯 **Predicted Diabetes Risk Score:** {risk_score:.2%}")

        if risk_score < 0.4:
            st.info("🟢 **Low Risk** — Maintain a healthy lifestyle, exercise, and balanced diet.")
        elif risk_score < 0.7:
            st.warning("🟠 **Moderate Risk** — Monitor glucose levels and consult a healthcare provider.")
        else:
            st.error("🔴 **High Risk** — Medical attention and lifestyle adjustments recommended.")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Risk Level Interpretation")
st.markdown("""
- **Low Risk (<40%)**: Maintain a healthy lifestyle, regular exercise, and balanced diet.  
- **Moderate Risk (40–69%)**: Monitor blood glucose levels and consult your healthcare provider.  
- **High Risk (≥70%)**: Requires medical attention and lifestyle adjustments.
""")

st.divider()
st.caption("© 2025 | Diabetes Risk Analysis Predictor by Matthew Osabhue")
