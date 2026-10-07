
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("titanic_decision_tree_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

st.divider()

# User inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["C", "Q", "S"]
)

# Prediction button
if st.button("Predict Survival"):

    # Create input data
    input_data = pd.DataFrame({
        "pclass": [pclass],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "sex_male": [1 if sex == "male" else 0],
        "embarked_Q": [1 if embarked == "Q" else 0],
        "embarked_S": [1 if embarked == "S" else 0]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🎉 Prediction: Passenger is likely to SURVIVE.")
    else:
        st.error("⚠️ Prediction: Passenger is likely NOT to survive.")
