# 📉 Telecom Churn Prediction App

An interactive Streamlit app that predicts customer churn for a telecom company based on user inputs. The app is powered by a trained XGBoost model and visualizes predictive outcomes in real time — including SHAP-based explanations of each prediction.

---

##  Project Overview

- **Goal:** Predict whether a telecom customer will churn based on demographics, services, and billing data.
- **Tech Stack:** Python, Streamlit, XGBoost, SHAP, Pandas, Scikit-learn
- **Features:**
  - Real-time churn prediction from customer input
  - SHAP explainability showing which features drive churn
  - Clean, intuitive UI with Streamlit
  - Power BI dashboard for deep dive analysis

---

##  How to Use the App

1. Visit the live app:  
    [https://telecom-churn-app-dfncr2ayafhav8iposkhtq.streamlit.app/](https://telecom-churn-app-dfncr2ayafhav8iposkhtq.streamlit.app/)

2. Fill in customer details: tenure, contract, internet usage, charges, etc.

3. Click **"Predict Churn"** to:
   - See if the customer is likely to churn
   - View SHAP visual explanation of the decision
--
##  Model Details

- **Model:** XGBoost Classifier
- **Accuracy:** ~76%
- **Recall (churn):** Improved using `scale_pos_weight` and threshold tuning
- **Explainability:** SHAP summary and force plots
- **Target Variable:** `Churn` (Yes/No)
---
## Project Structure
telecom-churn-app/
┣ 📄 app.py ← Streamlit web app
┣ 📄 xgb_churn_model.pkl ← Trained model
┣ 📄 label_encoders.pkl ← Saved label encoders
┣ 📄 shap_explainer.pkl ← SHAP explainer
┣ 📄 README.md ← This file

---

##  Dataset
- **Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** ~7,000 customers
- **Features:** Tenure, contract type, charges, internet & streaming services, etc.
---

##  Local Setup (Optional)

```bash
git clone https://github.com/your-username/telecom-churn-app.git
cd telecom-churn-app
pip install -r requirements.txt
streamlit run app.py

