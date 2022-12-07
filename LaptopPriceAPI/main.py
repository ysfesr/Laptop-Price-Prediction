from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

app = FastAPI()

model = mlflow.sklearn.load_model(
    model_uri=f"models:/LaptopPriceDetection/Staging"
)


class UserRequestIn(BaseModel):
    Company: str
    TypeName: str
    Inches: float
    Touchscreen: bool
    Ips: bool
    Cpu_Vendor: str
    Cpu_Type: str
    Ram: int
    Storage: int
    Storage_Type: str
    Gpu_Vendor: str
    Gpu_Type: str
    Weight: float
    OpSys: str


@app.post("/predict")
def main(user_request: UserRequestIn):
    Company = user_request.Company
    TypeName = user_request.TypeName
    Inches = user_request.Inches
    Touchscreen = user_request.Touchscreen
    Ips = user_request.Ips
    Cpu_Vendor = user_request.Cpu_Vendor
    Cpu_Type = user_request.Cpu_Type
    Ram = user_request.Ram
    Storage = user_request.Storage
    Storage_Type = user_request.Storage_Type
    Gpu_Vendor = user_request.Gpu_Vendor
    Gpu_Type = user_request.Gpu_Type
    Weight = user_request.Weight
    OpSys = user_request.OpSys

    print("--------------------------------", Company)
    data = [[Company, TypeName, Inches, Touchscreen, Ips, Cpu_Vendor, Cpu_Type,
            Ram, Storage, Storage_Type, Gpu_Vendor, Gpu_Type, Weight, OpSys]]

    price = model.predict(data)[0]

    return {"Estimated Price": price}
