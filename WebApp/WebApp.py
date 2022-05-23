import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = '123456789'


st.set_page_config(page_title="Laptop Price", layout="wide")
st.title('Predicting Laptops Price')

LaptopPriceDetection = mlflow.sklearn.load_model(
    model_uri=f"models:/LaptopPriceDetection/Staging"
)

col1, col2, col3, col4 = st.columns(4)


def user_input_features():
    with col1:
        company = st.selectbox(
            'Company:',
            ('HP',
             'Apple',
             'Acer',
             'Asus',
             'Dell',
             'Lenovo',
             'Chuwi',
             'MSI',
             'Microsoft',
             'Toshiba',
             'Huawei',
             'Xiaomi',
             'Vero',
             'Razer',
             'Mediacom',
             'Samsung',
             'Google',
             'Fujitsu',
             'LG'))
        TypeName = st.selectbox(
            'Laptop Type:',
            ('Ultrabook',
             'Notebook',
             'Netbook',
             'Gaming',
             '2 in 1 Convertible',
             'Workstation'))

        _Touchscreen = st.selectbox('Touchscreen (Yes/No):', ('Yes', 'No'))
        Touchscreen = 1 if _Touchscreen == 'Yes' else 0

    with col2:
        _ips = st.selectbox('IPS (Yes/No):', ('Yes', 'No'))
        ips = 1 if _ips == 'Yes' else 0
        cpu_vendor = st.selectbox('CPU Vendor:', ('Intel', 'AMD', 'Samsung'))
        cpu_type = st.text_input(
            'CPU Type (Core i7, Ryzen 1600 ...)', 'Core i7')

    with col3:
        storage_type = st.selectbox('Storage Type:', ('HDD', 'SSD', "Hybrid"))
        storage = st.slider('Storage size (GB)', 64, 2048, 256, 64)
        Ram = st.slider('Ram size (GB)', 4, 64, 8, 4)

        OpSys = st.selectbox(
            'Operating System:', ('Windows', 'Mac', 'Others/No OS/Linux'))

    with col4:
        gpu_vendor = st.selectbox(
            'GPU Vendor:', ('Nvidia', 'Intel', 'AMD', 'ARM'))
        gpu_type = st.text_input(
            'GPU type (Radeon Pro 455, GeForce 150MX ...)',
            "GeForce 150MX")
        Inches = st.slider(
            'Laptop Size in inches',
            min_value=14.0,
            max_value=17.5,
            value=15.5,
            step=0.5)
        weight = st.slider('Weight (KG)', 1.5, 3.0, 1.5, 0.1)

    data = [[company, TypeName, Inches, Touchscreen, ips, cpu_vendor, cpu_type,
             Ram, storage, storage_type, gpu_vendor, gpu_type, weight, OpSys]]

    return data


data = user_input_features()
if st.button("Predict"):
    price = LaptopPriceDetection.predict(data)
    st.success(f"Estimated Price: {round(price[0],2)} EURO")
