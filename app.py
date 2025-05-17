import streamlit as st
import pandas as pd
import joblib
import shap
import streamlit.components.v1 as components


#  Load model and encoders
model = joblib.load("xgb_churn_model.pkl")  # Load your trained model
encoders = joblib.load("label_encoders.pkl")  # Load label encoders used during training
explainer = joblib.load("shap_explainer.pkl")

st.title("📉 Telecom Churn Prediction App")

# Collect user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Prepare the input for prediction
input_dict = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_dict])

#  Use the encoders from training (don't re-fit!)
for col in input_df.select_dtypes(include='object').columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])
    else:
        st.warning(f"Missing encoder for column: {col}")

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    st.subheader("🔍 Prediction:")
    st.success("Customer is likely to **Churn**" if prediction == 1 else "Customer is **Not likely to Churn**")

# Explain prediction with SHAP
shap_values = explainer(input_df)

# Render SHAP HTML and show in Streamlit
st.subheader("🔍 SHAP Explanation:")
shap_html = shap.plots.force(explainer.expected_value, shap_values.values[0], input_df.iloc[0], matplotlib=False, show=False)

# Display using components
components.html(shap.getjs() + shap_html.html(), height=300)
