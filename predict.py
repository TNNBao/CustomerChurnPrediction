import streamlit as st
from model import preprocess, predict
def show():
    st.title("🔍 Customer Churn Prediction")

    st.markdown("Please fill out the information below to predict the likelihood of customer churn:")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=122, value=30)
    SeniorCitizen = 1 if age >= 60 else 0

    Partner = st.selectbox("Has partner?", ["Yes", "No"])
    Dependents = st.selectbox("Has dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (in months)", 0, 72, 12)

    # Phone service logic
    PhoneService = st.selectbox("Phone service", ["Yes", "No"])
    if PhoneService == "No":
        MultipleLines = "No phone service"
        st.selectbox("Multiple lines", ["No phone service"], index=0, disabled=True)
    else:
        MultipleLines = st.selectbox("Multiple lines", ["Yes", "No"])

    # Internet service logic
    InternetService = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    if InternetService == "No":
        OnlineSecurity = "No internet service"
        OnlineBackup = "No internet service"
        DeviceProtection = "No internet service"
        TechSupport = "No internet service"
        StreamingTV = "No internet service"
        StreamingMovies = "No internet service"

        st.selectbox("Online security", ["No internet service"], index=0, disabled=True)
        st.selectbox("Online backup", ["No internet service"], index=0, disabled=True)
        st.selectbox("Device protection", ["No internet service"], index=0, disabled=True)
        st.selectbox("Tech support", ["No internet service"], index=0, disabled=True)
        st.selectbox("Streaming TV", ["No internet service"], index=0, disabled=True)
        st.selectbox("Streaming movies", ["No internet service"], index=0, disabled=True)
    else:
        OnlineSecurity = st.selectbox("Online security", ["Yes", "No"])
        OnlineBackup = st.selectbox("Online backup", ["Yes", "No"])
        DeviceProtection = st.selectbox("Device protection", ["Yes", "No"])
        TechSupport = st.selectbox("Tech support", ["Yes", "No"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
        StreamingMovies = st.selectbox("Streaming movies", ["Yes", "No"])

    Contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    MonthlyCharges = st.number_input("Monthly charges", value=70.0)
    TotalCharges = st.number_input("Total charges", value=1500.0)

    if st.button("Predict"):
        try:
            preprocessed_data = preprocess(
                gender, SeniorCitizen, Partner, Dependents, tenure,
                PhoneService, MultipleLines, InternetService, OnlineSecurity,
                OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
                Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
            )

            churn_probability = predict(preprocessed_data)
            scaled_probability = (churn_probability / 33.0) * 50.0
            scaled_probability = min(scaled_probability, 100)

            st.success(f"The predicted churn probability is: **{scaled_probability:.2f}%**")

            if churn_probability >= 33:
                st.warning("⚠️ Warning: The customer is likely to churn.")
            else:
                st.info("✅ Good news: The customer is likely to stay.")
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")

