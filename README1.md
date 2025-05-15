 telecom-churn-app/
┣ 📄 app.py # Streamlit web app
┣ 📄 xgb_churn_model.pkl # Trained machine learning model
┣ 📄 README1.md # This file
---

##  Dataset

- [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,000+ customer records
- 20+ features: tenure, monthly charges, contract type, internet service, etc.

---

## 🛠 Local Setup (Optional)

```bash
git clone https://github.com/your-username/telecom-churn-app.git
cd telecom-churn-app
pip install -r requirements.txt
streamlit run app.py
