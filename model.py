import numpy as np
import pandas as pd
import pickle
import os

def load_file():
    base_path = r'D:\CODE\Python_project\Customer_Churn_Prediction\model'

    with open(os.path.join(base_path, 'lr_model.pkl'), 'rb') as lr_f:
        lr_model = pickle.load(lr_f)
    with open(os.path.join(base_path, 'Scaler.pkl'), 'rb') as scaler_f:
        Scaler = pickle.load(scaler_f)

    with open(os.path.join(base_path, 'internet_service_OneHot.pkl'), 'rb') as file:
        internet_service_OneHot = pickle.load(file)
    with open(os.path.join(base_path, 'payment_OneHot.pkl'), 'rb') as file:
        payment_OneHot = pickle.load(file)
    with open(os.path.join(base_path, 'contract_OneHot.pkl'), 'rb') as file:
        contract_OneHot = pickle.load(file)

    with open(os.path.join(base_path, 'gender_Label.pkl'), 'rb') as file:
        gender_Label = pickle.load(file)
    with open(os.path.join(base_path, 'Partner_Label.pkl'), 'rb') as file:
        Partner_Label = pickle.load(file)
    with open(os.path.join(base_path, 'Dependents_Label.pkl'), 'rb') as file:
        Dependents_Label = pickle.load(file)
    with open(os.path.join(base_path, 'PhoneService_Label.pkl'), 'rb') as file:
        PhoneService_Label = pickle.load(file)
    with open(os.path.join(base_path, 'MultipleLines_Label.pkl'), 'rb') as file:
        MultipleLines_Label = pickle.load(file)
    with open(os.path.join(base_path, 'OnlineSecurity_Label.pkl'), 'rb') as file:
        OnlineSecurity_Label = pickle.load(file)
    with open(os.path.join(base_path, 'OnlineBackup_Label.pkl'), 'rb') as file:
        OnlineBackup_Label = pickle.load(file)
    with open(os.path.join(base_path, 'DeviceProtection_Label.pkl'), 'rb') as file:
        DeviceProtection_Label = pickle.load(file)
    with open(os.path.join(base_path, 'TechSupport_Label.pkl'), 'rb') as file:
        TechSupport_Label = pickle.load(file)
    with open(os.path.join(base_path, 'StreamingTV_Label.pkl'), 'rb') as file:
        StreamingTV_Label = pickle.load(file)
    with open(os.path.join(base_path, 'StreamingMovies_Label.pkl'), 'rb') as file:
        StreamingMovies_Label = pickle.load(file)
    with open(os.path.join(base_path, 'PaperlessBilling_Label.pkl'), 'rb') as file:
        PaperlessBilling_Label = pickle.load(file)

    return (lr_model, Scaler, internet_service_OneHot, payment_OneHot, contract_OneHot,
            gender_Label, Partner_Label, Dependents_Label, PhoneService_Label,
            MultipleLines_Label, OnlineSecurity_Label, OnlineBackup_Label,
            DeviceProtection_Label, TechSupport_Label, StreamingTV_Label,
            StreamingMovies_Label, PaperlessBilling_Label)

lr_model, Scaler, internet_service_OneHot, payment_OneHot, contract_OneHot, gender_Label, Partner_Label, Dependents_Label, PhoneService_Label, MultipleLines_Label, OnlineSecurity_Label, OnlineBackup_Label, DeviceProtection_Label, TechSupport_Label, StreamingTV_Label, StreamingMovies_Label, PaperlessBilling_Label = load_file()

def preprocess(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    lr_model, Scaler, internet_service_OneHot, payment_OneHot, contract_OneHot, gender_Label, Partner_Label, Dependents_Label, PhoneService_Label, MultipleLines_Label, OnlineSecurity_Label, OnlineBackup_Label, DeviceProtection_Label, TechSupport_Label, StreamingTV_Label, StreamingMovies_Label, PaperlessBilling_Label = load_file()

    gender_encoded = gender_Label.transform([gender])[0]
    Partner_encoded = Partner_Label.transform([Partner])[0]
    Dependents_encoded = Dependents_Label.transform([Dependents])[0]
    PhoneService_encoded = PhoneService_Label.transform([PhoneService])[0]
    MultipleLines_encoded = MultipleLines_Label.transform([MultipleLines])[0]
    OnlineSecurity_encoded = OnlineSecurity_Label.transform([OnlineSecurity])[0]
    OnlineBackup_encoded = OnlineBackup_Label.transform([OnlineBackup])[0]
    DeviceProtection_encoded = DeviceProtection_Label.transform([DeviceProtection])[0]
    TechSupport_encoded = TechSupport_Label.transform([TechSupport])[0]
    StreamingTV_encoded = StreamingTV_Label.transform([StreamingTV])[0]
    StreamingMovies_encoded = StreamingMovies_Label.transform([StreamingMovies])[0]
    PaperlessBilling_encoded = PaperlessBilling_Label.transform([PaperlessBilling])[0]

    internet_service_encoded = internet_service_OneHot.transform(pd.DataFrame([[InternetService]], columns=["InternetService"])).toarray()
    payment_method_encoded = payment_OneHot.transform(pd.DataFrame([[PaymentMethod]], columns=["PaymentMethod"])).toarray()
    contract_encoded = contract_OneHot.transform(pd.DataFrame([[Contract]], columns=["Contract"])).toarray()

    one_hot_encoded_data = np.concatenate([internet_service_encoded, payment_method_encoded, contract_encoded], axis=1)

    remain_data = np.array([[gender_encoded, SeniorCitizen, Partner_encoded, Dependents_encoded, tenure, PhoneService_encoded, MultipleLines_encoded, OnlineSecurity_encoded, OnlineBackup_encoded, DeviceProtection_encoded, TechSupport_encoded, StreamingTV_encoded, StreamingMovies_encoded, PaperlessBilling_encoded, MonthlyCharges, TotalCharges]])

    if one_hot_encoded_data.ndim == 1:
        one_hot_encoded_data = one_hot_encoded_data.reshape(1, -1)

    full_data = np.hstack((remain_data, one_hot_encoded_data))

    full_data_df = pd.DataFrame(full_data, columns=Scaler.feature_names_in_)

    scaler_data = Scaler.transform(full_data_df)

    return scaler_data

def predict(preprocessed_data):
    result = lr_model.predict_proba(preprocessed_data)
    return result[0][1]*100