import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import mlflow

data = pd.read_csv("./datasets/laptops.csv", encoding='latin-1')

# Remove extra unnecessary details form Product Columns
data["Product"] = data["Product"].str.split("(").apply(lambda x: x[0])

# Extract CPU Vender, CPU Type and CPU Speed in Different Columns
data["Cpu_Speed"] = data["Cpu"].str.split(" ").apply(
    lambda x: x[-1]).str.replace("GHz", "")
data["Cpu_Vender"] = data["Cpu"].str.split(" ").apply(lambda x: x[0])
data["Cpu_Type"] = data["Cpu"].str.split(" ").apply(lambda x: x[1:4] if x[1] == "Celeron" and "Pentium" and "Xeon" else (
    x[1:3] if (x[1] == "Core" or x[0] == "AMD") else x[0]))
data["Cpu_Type"] = data["Cpu_Type"].apply(lambda x: ' '.join(x))

# Extract Memory type from Memory Column
split_mem = data['Memory'].str.split(' ', 1, expand=True)
data['Storage Type'] = split_mem[1]
data['Memory'] = split_mem[0]

data["Ram"] = data["Ram"].str.replace("GB", "")

df_mem = data['Memory'].str.split('(\\d+)', expand=True)
data['Memory'] = pd.to_numeric(df_mem[1])
data.rename(columns={'Memory': 'Memory (GB or TB)'}, inplace=True)


data['Memory (GB or TB)'] = data['Memory (GB or TB)'].apply(
    lambda x: 1024 if x == 1 else x)
data['Memory (GB or TB)'] = data['Memory (GB or TB)'].apply(
    lambda x: 2048 if x == 2 else x)
data.rename(columns={'Memory (GB or TB)': 'Storage (GB)'}, inplace=True)
data["Weight"] = data["Weight"].str.replace("kg", "")

# Extract GPU Vender, GPU Type in Different Columns
gpu_distribution_list = data["Gpu"].str.split(" ")
data["Gpu_Vender"] = data["Gpu"].str.split(" ").apply(lambda x: x[0])
data["Gpu_Type"] = data["Gpu"].str.split(" ").apply(lambda x: x[1:])
data["Gpu_Type"] = data["Gpu_Type"].apply(lambda x: ' '.join(x))

# Extract IPS and Touchscreen Feature form ScreenResolution Column
data['Touchscreen'] = data['ScreenResolution'].apply(
    lambda x: 1 if 'Touchscreen' in x else 0)
data['Ips'] = data['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# Catorizing The Operating System
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'MacOS'
    else:
        return 'Others/No OS/Linux'


data['OpSys'] = data['OpSys'].apply(cat_os)

# Fetching Out The Use Full Columns the Leaving The Rest
data = data.reindex(
    columns=[
        "Company",
        "TypeName",
        "Inches",
        "Touchscreen",
        "Ips",
        "Cpu_Vender",
        "Cpu_Type",
        "Ram",
        "Storage (GB)",
        "Storage Type",
        "Gpu_Vender",
        "Gpu_Type",
        "Weight",
        "OpSys",
        "Price_euros"])

# Transforming the Data Type of some of the Columns: Ram Storage Weight
data["Ram"] = data["Ram"].astype("int")
data["Storage (GB)"] = data["Storage (GB)"].astype("int")
data["Weight"] = data["Weight"].astype("float")

# Split the data
X = data.drop(columns=['Price_euros'])
y = data['Price_euros']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=2)

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 350
max_samples = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
max_features = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else None

# experiment = mlflow.set_experiment("LaptopPricePrediction")

with mlflow.start_run():
    Transformer = ColumnTransformer(
        transformers=[
            ('col_tnf', OneHotEncoder(
                sparse=False, handle_unknown='ignore'), [
                    0, 1, 5, 6, 9, 10, 11, 13])], remainder='passthrough')

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        max_depth=max_features)

    pipe = Pipeline([
        ('transformer', Transformer),
        ('Regressor', rf)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2_score = r2_score(y_test, y_pred)
    mean_absolute_error = mean_absolute_error(y_test, y_pred)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_samples", max_samples)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("max_depth", max_depth)

    mlflow.log_metric("r2_score", r2_score)
    mlflow.log_metric("mae", mean_absolute_error)

    mlflow.sklearn.log_model(pipe, "Predictor")

    print('R2 score', r2_score)
    print('MAE', mean_absolute_error)
