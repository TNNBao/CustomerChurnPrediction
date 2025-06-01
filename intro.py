import streamlit as st
import os

def show():
    st.title("🔍 Customer Churn Prediction")

    st.write("""
    Welcome to the Telco Customer Churn prediction system!
    
    This system uses Machine Learning to analyze customer data and predict the likelihood of them churning.

    **This application helps:**
    - Optimize customer retention strategy
    - Analyze user behavior
    - Improve service quality
    """)

    st.markdown("---")
    st.subheader("✅ Model used: **Logistic Regression**")
    st.markdown("""
    - Accuracy: **80.24%**
    - Precision: **0.62**
    - Recall: **0.61**
    - F1-score: **0.61**
    
    Optimal threshold: **33%**
    """)

    st.markdown("---")
    st.subheader("📈 Statistics")
    
    image_dir = "images"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for img in sorted(image_files):
        st.image(os.path.join(image_dir, img))
