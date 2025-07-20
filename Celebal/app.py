import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Page title
st.title("🔍 Diabetes Disease Progression Predictor")
st.write("This app predicts diabetes progression based on medical features.")

# User input via sliders
def user_input():
    age = st.slider("Age (standardized)", -0.1, 0.15, 0.0)
    sex = st.slider("Sex (standardized)", -0.1, 0.1, 0.0)
    bmi = st.slider("BMI", -0.1, 0.1, 0.0)
    bp = st.slider("Blood Pressure", -0.1, 0.1, 0.0)
    s1 = st.slider("S1 (TC)", -0.1, 0.1, 0.0)
    s2 = st.slider("S2 (LDL)", -0.1, 0.1, 0.0)
    s3 = st.slider("S3 (HDL)", -0.1, 0.1, 0.0)
    s4 = st.slider("S4 (TCH)", -0.1, 0.1, 0.0)
    s5 = st.slider("S5 (LTG)", -0.1, 0.1, 0.0)
    s6 = st.slider("S6 (GLU)", -0.1, 0.1, 0.0)

    data = {
        'age': age, 'sex': sex, 'bmi': bmi, 'bp': bp,
        's1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6
    }
    return pd.DataFrame(data, index=[0])

# Collect input
input_df = user_input()

# Predict button
if st.button("Predict Progression"):
    prediction = model.predict(input_df)
    st.success(f"📈 Predicted Progression Score: {prediction[0]:.2f}")

    # Feature importance visualization
    st.subheader("🔎 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': input_df.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    st.pyplot(fig)