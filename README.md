# Laptops Price Prediction

The goal of this project is to build a supervised machine learning model to predict the price of a laptop based on its features. The model will be trained with a dataset containing features such as processor type, RAM, storage, etc. and corresponding prices. MLflow will be used to track the project’s progress, parameters and metrics. Scikit-learn will be used to create the machine learning model. Streamlit will be used to create an interactive web application that allows users to input the laptop features and get a prediction of the laptop’s price. Finally, FastAPI will be used to create an API that can be used to access the machine learning model and retrieve the predictions.
## Screenshots

![Home Page](/images/1.png)


## MLflow Project (./Project)
- Run docker compose first, which contains a database to store metadata and a Minio image to store artifacts
```bash
docker-compose up
```
- Create a bucket named "mlflow" in [minio](http://localhost:9001) to store our artifacts.
- Start the Mlflow server

```bash
mlflow server --backend-store-uri postgresql://root:root@localhost:5432/mlflow_db --default-artifact-root s3://mlflow/ --host 0.0.0.0
```

- Run the MLflow Project to train your models.
```bash
cd ./Project
mlflow run . --experiment-name <example-name> -P n_estimators=1000 -P max_samples=0.5 -P max_features=0.75 -P max_depth=30
```
## Streamlit Web App (./WebApp)
- Save the best-trained model to the Model Registry and name it "LaptopPriceDetection", then stage the model to "staging"
![Model Register](/images/2.png)
![Model Register](/images/3.png)

- Create a virtual environment for the web application, then run it.

```bash
cd ./WebApp
virtualenv venv
./venv/Scripts/activate
pip install -r requirements.txt
```
```bash
streamlit run ./WebApp.py
```
## API
- Create a conda virtual environment and activate it
```bash
cd ./LaptopAPI
conda env create -f conda.yaml
conda activate laptopapi
```
- launch the API
```bash
uvicorn main:app
```
## links
- **MLflow UI:**    http://localhost:5000
- **Web Application:**  http://localhost:8501
- **Minio:**  http://localhost:9001
- **API ENDPOINT:** http://localhost:8000

## Built With

- MLflow
- Scikit-learn
- Streamlit
- FastAPI


## Author

**Youssef EL ASERY**

- [Profile](https://github.com/ysfesr "Youssef ELASERY")
- [Linkedin](https://www.linkedin.com/in/youssef-elasery/ "Welcome")
- [Kaggle](https://www.kaggle.com/youssefelasery "Welcome")


## 🤝 Support

Contributions, issues, and feature requests are welcome!

Give a ⭐️ if you like this project!
