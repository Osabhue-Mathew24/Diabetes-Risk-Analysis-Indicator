Overview

The Diabetes Risk Predictor App is a machine learning–powered web application designed to estimate an individual’s risk of developing diabetes based on key health indicators.
It provides a simple, interactive interface built with Streamlit, enabling users to input their health data and instantly receive a predicted diabetes risk score, along with a clear interpretation.

Objective

The goal of this project is to apply data-driven insights to help individuals and healthcare practitioners make early, informed decisions about diabetes management and prevention.

Key Features

Interactive User Interface built with Streamlit|
Dynamic Risk Score Prediction powered by a trained ML model|
Automatic BMI Calculation (for users who don’t know theirs)|
Health Risk Categorization: Low, Moderate, or High Risk|
Clean Visuals and Intuitive Layout|

Model Information

The app uses a Random Forest Classifier trained on key predictors such as:

stage – Diabetes stage (Normal, Prediabetes, Diabetes) |
hba1c – Glycated Hemoglobin (%) |
gluc_fast – Fasting Blood Glucose (mg/dL) |
gluc_post – Postprandial Glucose (mg/dL) |
fh_diabetes – Family history of diabetes (Yes/No) |
age – Age of the individual |
bmi – Body Mass Index |

Tech Stack

Python 3.10+

Streamlit – App interface

Pandas / NumPy – Data handling

Scikit-learn – Model training

Plotly – Visualization

Joblib – Model serialization


The model outputs a risk probability score, which is used to classify individuals into low, moderate, or high risk categories.
