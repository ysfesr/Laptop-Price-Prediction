# Laptops Price Prediction

Build a supervised machine learning model to predict the price of a laptop based on its features

## Screenshots

![Home Page](/images/1.png)


## MLflow Project (./Project)
- First Run docker compose its contain a database to store metadata and an minio image to store artifacts
```bash
docker-compose up
```
- Create a bucket in [minio](http://localhost:9001) to store our artifacts (name it mlflow)
- Launch the Mlflow server

```bash
mlflow server --backend-store-uri postgresql://root:root@localhost:5432/mlflow_db --default-artifact-root s3://mlflow/ --host 0.0.0.0
```

- You can now train your models by running the Project
```bash
cd ./Project
mlflow run . --experiment-name <example-name> -P n_estimators=1000 -P max_samples=0.5 -P max_features=0.75 -P max_depth=30
```
## Streamlit Web App (./WebApp)
- Find the best-trained model and save it to the Model Registry, name the Model Registry LaptopPriceDetection and stage the model to "staging"
![Model Register](/images/2.png)
![Model Register](/images/3.png)

- Now create the virtual environment for the web application and run it

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
- Create a conda virtual environement and active it
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