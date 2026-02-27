import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Salary Prediction", page_icon="💼")

st.title("💼 Salary Prediction App")

# Load dataset
dataset = pd.read_csv("Salary_Data.csv")

X = dataset[["YearsExperience"]].values
y = dataset["Salary"].values

# Train model inside app
model = LinearRegression()
model.fit(X, y)

# Input
years = st.number_input("Enter Years of Experience", 0.0, 50.0, 1.0)

if st.button("Predict Salary"):
    prediction = model.predict([[years]])
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")

st.caption("Built with Streamlit & Scikit-learn")
