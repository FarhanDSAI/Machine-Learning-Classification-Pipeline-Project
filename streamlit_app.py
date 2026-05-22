
import streamlit as st
import pandas as pd
import joblib

st.title("Adult Income Prediction")
model = joblib.load("model.pkl")

age = st.number_input("Age", 18, 100, 30)
hours = st.number_input("Hours per week", 1, 100, 40)

if st.button("Predict"):
    sample = pd.DataFrame([{
        "age": age,
        "workclass": "Private",
        "fnlwgt": 226802,
        "education": "11th",
        "educational-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "gender": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": hours,
        "native-country": "United-States"
    }])
    pred = model.predict(sample)[0]
    st.success(f"Prediction: {pred}")
