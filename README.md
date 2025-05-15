#  Telecom Churn Prediction App

An interactive Streamlit app that predicts customer churn for a telecom company based on user inputs. The app is powered by a trained XGBoost model and visualizes predictive outcomes in real time.

---

##  Project Overview

- **Goal:** Predict whether a telecom customer will churn based on demographics, services, and billing data.
- **Tech Stack:** Python, Streamlit, XGBoost, Pandas, Scikit-learn
- **Features:**
  - Real-time churn prediction from user inputs
  - Trained on Telco Customer Churn dataset (IBM Sample Data)
  - Clean, intuitive UI using Streamlit

---

##  How to Use the App

1. Go to the live app: [🌐 Streamlit App Link](https://telecom-churn-app-dfncr2ayafhav8iposkhtq.streamlit.app/)
2. Fill in customer details (tenure, contract, charges, etc.)
3. Click "Predict Churn" to see if the customer is likely to churn

---

##  Model Details

- **Algorithm:** XGBoost Classifier
- **Accuracy:** ~76% (improved recall for churn via SMOTE)
- **Target Variable:** `Churn` (Yes/No)

---

##  Project Structure

