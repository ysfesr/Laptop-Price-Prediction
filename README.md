# Laptops Price Prediction

Build a supervised machine learning model to predict the price of a laptop based on its features

## Screenshots

![Home Page](/images/1.png)


## Setup
- Run docker compose 
```bash
docker-compose up
```

- Create a bucket in [minio](http://localhost:9001) to store our artifacts (name it mlflow)


- Create a conda virtual environement and active it
```bash
conda env create -f Project/conda.yaml
conda activate laptopenv
```

- You can now train your models by running the Project
```bash
mlflow run .\Project\ --experiment-name example -P n_estimators=1000 -P max_samples=0.5 -P max_features=0.75 -P max_depth=30
```
- Find the best-trained model and save it to the Model Registry, name the Model Registry LaptopPricePrediction and stage the model to staging
![Model Register](/images/2.png)
![Model Register](/images/3.png)

- Now create the virtual environment for the web application and run it

```bash
virtualenv venv
./venv/Scripts/activate
pip install -r "/Web App/requirements.txt"
streamlit run "/Web App/WebApp.py"
```

## links
- **MLFlow:**    http://localhost:9090
- **Web Application:**  http://localhost:8501
- **Minio:**  http://localhost:9001

## Built With

- MLFlow
- Scikit-learn
- Streamlit


## Author

**Youssef EL ASERY**

- [Profile](https://github.com/ysfesr "Youssef ELASERY")
- [Linkedin](https://www.linkedin.com/in/youssef-elasery/ "Welcome")
- [Kaggle](https://www.kaggle.com/youssefelasery "Welcome")


## 🤝 Support

Contributions, issues, and feature requests are welcome!

Give a ⭐️ if you like this project!