# Customer Churn Prediction System

## Project Overview

This project predicts whether a telecom customer is likely to leave the company (churn) using Machine Learning.

The model analyzes customer information such as tenure, contract type, internet service, monthly charges, and payment method to predict churn risk.

---

## Problem Statement

Customer churn causes revenue loss for telecom companies.

The goal of this project is to identify customers who are likely to churn so that the company can take retention actions.

---

## Dataset

IBM Telco Customer Churn Dataset

Features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Churn

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Encoding
5. Train-Test Split
6. Random Forest Model Training
7. Model Evaluation
8. Model Deployment using Streamlit

---

## Data Cleaning Steps

- Removed customerID column
- Converted TotalCharges to numeric values
- Handled missing values
- Encoded categorical variables using One-Hot Encoding
- Converted target variable (Churn) into numerical format

---

## Model Used

Random Forest Classifier

Reason:
- Good accuracy
- Handles categorical data well
- Reduces overfitting
- Easy to interpret

---

## Features

- Predict customer churn
- Display churn risk score
- Real-time prediction using Streamlit
- Retention recommendations

---

## Project Structure

Customer-Churn-Project

├── data

├── models

├── app.py

├── train_model.py

├── requirements.txt

└── README.md

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python train_model.py
```

Run Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

- Power BI Dashboard
- Feature Importance Visualization
- Advanced ML Models
- Cloud Deployment

---

## Author

Kumar Suriya