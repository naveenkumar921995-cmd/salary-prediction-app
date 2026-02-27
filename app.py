import streamlit as st
import pickle
import numpy as np

# Load model (keep model file in same folder)
model = pickle.load(open("linear_regression_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="centered")

# Title
st.title("💼 Salary Prediction App")

st.markdown("### Predict Salary based on Years of Experience")
st.write("This app uses a Machine Learning Linear Regression model.")

# Input
years_experience = st.number_input(
    "Enter Years of Experience:",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5
)

# Prediction button
if st.button("Predict Salary"):
    experience_input = np.array([[years_experience]])
    prediction = model.predict(experience_input)

    st.success(
        f"💰 Estimated Salary for {years_experience} years experience: "
        f"${prediction[0]:,.2f}"
    )

st.markdown("---")
st.caption("Built with Python, Scikit-Learn & Streamlit")
