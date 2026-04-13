# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Title of App
# -----------------------------
st.title("Diabetes Prediction App")

st.write("Enter patient details to predict diabetes outcome")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("diabetes.csv")

# -----------------------------
# Data Preprocessing
# -----------------------------
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Define features and target
X = df.drop('Outcome', axis=1)
Y = df['Outcome']

# Train model
model = LogisticRegression()
model.fit(X, Y)

# -----------------------------
# User Inputs
# -----------------------------
Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose", 0, 200, 120)
BloodPressure = st.number_input("Blood Pressure", 0, 150, 70)
SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
Insulin = st.number_input("Insulin", 0, 900, 80)
BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age = st.number_input("Age", 1, 120, 30)

# Create input array
input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                        Insulin, BMI, DiabetesPedigreeFunction, Age]])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Diabetes (Probability: {probability:.2f})")
