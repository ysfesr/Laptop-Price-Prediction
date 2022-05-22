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

with col1:
    company = st.selectbox('Company:',('Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
       'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer',
       'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'))
    TypeName = st.selectbox('Laptop Type:',('Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible',
       'Workstation'))
    Inches = st.number_input('Laptop inch size', step=0.1)
    _Touchscreen = st.selectbox('Touchscreen (Yes/No):',('Yes', 'No'))
    Touchscreen = 1 if _Touchscreen == 'Yes' else 0

with col2:
    _ips = st.selectbox('IPS (Yes/No):',('Yes', 'No'))
    ips = 1 if _ips == 'Yes' else 0
    cpu_vendor = st.selectbox('CPU Vendor:',('Intel', 'AMD', 'Samsung'))
    cpu_type = st.text_input('CPU Type (Core i7, Ryzen 1600 ...)')

with col3:
    Ram = st.slider('Ram size (GB)', 4, 64, 8, 4)
    storage = st.slider('Storage size (GB)', 64, 2048, 256, 64)
    storage_type = st.selectbox('Storage Type:',('HDD', 'SSD', "Hybrid"))

with col4:
    gpu_vendor = st.selectbox('GPU Vendor:',('Intel', 'AMD', 'Nvidia', 'ARM'))
    gpu_type = st.text_input('GPU type (Radeon Pro 455, GeForce 150MX...)')
    weight = st.number_input('Weight (KG))', step=0.05)
    OpSys = st.selectbox('Operating System:',('Mac', 'Others/No OS/Linux', 'Windows'))

def prediction():
    data = np.array([company, TypeName, Inches, Touchscreen, ips, cpu_vendor,cpu_type,\
        Ram, storage, storage_type, gpu_vendor, gpu_type, weight, OpSys])
    data = np.array(["Apple", "Ultrabook", 13.3, 0, 1, "Intel","Core i5",\
        9, 128, "SSD", "Intel", "Iris Plus Graphics 640", "1.37", "Mac"])

    return LaptopPriceDetection.predict('Apple', 'Ultrabook', 13.3, 0, 1, 'Intel', 'Core i5', 8, 128,
       'SSD', 'Intel', 'Iris Plus Graphics 640', 1.37, 'Mac')

if st.button("Predict"):
    cols = ['Company', 'TypeName', 'Inches', 'Touchscreen', 'Ips', 'Cpu_Vender',
       'Cpu_Type', 'Ram', 'Storage (GB)', 'Storage Type', 'Gpu_Vender',
       'Gpu_Type', 'Weight', 'OpSys']
    data = pd.DataFrame([['Apple', 'Ultrabook', 13.3, 0, 1, 'Intel', 'Core i5', 8, 128,'SSD', 'Intel', 'Iris Plus Graphics 640', 1.37, 'Mac']], columns=cols)
    price = LaptopPriceDetection.predict(data)
    st.success(f"Estimated Price: {price}")


