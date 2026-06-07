import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("models/churn_model.pkl")
features = joblib.load("models/features.pkl")

st.title("Customer Churn Prediction")

st.write("Enter Customer Details")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

if st.button("Predict Churn"):

    data = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly
    }

    df = pd.DataFrame([data])

    # Match training features
    df = pd.get_dummies(df)

    df = df.reindex(
        columns=features,
        fill_value=0
    )

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    st.subheader(
        f"Churn Risk Score: {round(probability*100,2)}%"
    )

    if prediction == 1:
        st.error("Customer Likely To Churn")

        st.warning("""
Recommended Actions:

• Offer Discount

• Loyalty Rewards

• Customer Support Follow-up

• Upgrade Plan
""")

    else:
        st.success("Customer Likely To Stay")