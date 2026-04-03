from flask import Flask,render_template,request
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from flask_app.preprocessing_utility import normalize_text
import pickle
import os

app=Flask(__name__)

# load vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "rahulpatel16092005"
repo_name = "mlops-mini-project"

 # Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# load the latest model from DagsHub
def get_latest_model_version(model_name):
    client=MlflowClient()
    latest_version=client.get_latest_versions(model_name,stages=["Production"])
    if not latest_version:
        latest_version=client.get_latest_versions(model_name,stages=["None"])
    return latest_version[0] if latest_version else None
model_name = "my_model"
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version.version}"
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/')
def home():
    return render_template('index.html')   

@app.route('/predict',methods=['POST'])
def predct():

    text = request.form['text']
    normalized_text = normalize_text(text)

    
    text_vector = vectorizer.transform([normalized_text])

    model_prediction = model.predict(text_vector)

    return render_template('index.html', prediction=model_prediction[0])


app.run(debug=True)