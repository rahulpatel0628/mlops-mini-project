# 🚀 Emotion Detection from Text (End-to-End MLOps Project)

## 📌 Overview

This project implements a complete **end-to-end MLOps pipeline** for detecting emotions from textual data. It covers everything from data ingestion to model deployment using tools like MLflow, DVC, Docker, and CI/CD.

---

## 🎯 Key Features

* Complete ML Pipeline (Data → Model → Deployment)
* Experiment Tracking using MLflow + DagsHub
* Data & Pipeline Versioning using DVC + AWS S3
* Automated Workflow using DVC Pipelines
* Multiple Model Training & Comparison
* Hyperparameter Tuning
* Model Versioning with MLflow Registry
* REST API using Flask
* Dockerized Deployment
* CI/CD Automation with GitHub Actions

---

## ⚙️ Project Workflow

### 1. Data Ingestion

* Load raw text dataset

### 2. Data Preprocessing

* Text cleaning
* Tokenization
* Stopword removal

### 3. Feature Engineering

* Bag of Words (CountVectorizer)
* TF-IDF (TfidfVectorizer)

### 4. Model Training

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* Naive Bayes

### 5. Model Selection

* Best Model: Logistic Regression (BOW)
* Accuracy: ~80%

### 6. Hyperparameter Tuning

* C = 1.0
* solver = liblinear
* penalty = l2

### 7. Experiment Tracking

Using MLflow + DagsHub:

* Parameters
* Metrics
* Models
* Artifacts

### 8. Pipeline Automation (DVC)

Pipeline stages:

* data_ingestion
* preprocessing
* feature_engineering
* model_training
* evaluation
* registration

### 9. Data Versioning

* DVC integrated with AWS S3
* Ensures reproducibility

### 10. Model Registration

* Registered using MLflow
* Promoted to Staging

---

## 🌐 Flask API (Model Serving)

The trained model is served using a Flask API.

### Endpoint
```json
POST /predict
```
### Example Request
```json
{
"text": "I am feeling very happy today!"
}
```
### Example Response
```json
{
"emotion": "joy"
}
```
---

## 🐳 Docker (Containerization)

### Build Image
```bash
docker build -t emotiondetection .
```
### Run Container
```bash
docker run -p 8888:5000 emotiondetection
```
### Access App
```bash
http://localhost:8888
```
---
# 🚀 Deployment (AWS EC2 + Docker + CI/CD)

### Live Application
```bash
http://ec2-13-61-153-124.eu-north-1.compute.amazonaws.com/
```
### Deployment Architecture
```bash
GitHub → GitHub Actions → Docker Hub → AWS EC2 → Docker Container → Flask API
```
### Docker Hub
```bash
https://hub.docker.com/r/rahulpatel0628/emotiondetection
```
---

# AWS EC2 Deployment Steps

### 1. Launch EC2 Instance
- Ubuntu Server
- Open Ports: 22 (SSH), 80 (HTTP)
### 2. Install Docker
```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```
### 3. Pull Docker Image
```bash
docker pull rahulpatel0628/emotiondetection:latest
```
### 4. Run Container
```bash
docker rm -f emotiondetection || true

docker run -d -p 80:5000 --name emotiondetection \
--restart always \
-e DAGSHUB_PAT=<your_token> \
rahulpatel0628/emotiondetection:latest
```
---
# ⚙️ CI/CD Pipeline (GitHub Actions)

### On every push:
- Install dependencies
- Run unit tests
- Build Docker image
- Push Docker image to Docker Hub
- Deploy to AWS EC2
- Pull latest image
- Restart container

---

# 🧪 Testing

### Run model tests:
```bash
python -m unittest tests/test_model.py
```

### Run Flask tests:
```bash
python -m unittest tests/test_flask_app.py
```
---

## 📊 Results

* Accuracy: ~0.78
* Precision: ~0.76
* Recall: ~0.80
* F1 Score: ~0.78

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* MLflow
* DagsHub
* DVC
* AWS S3
* Flask
* Docker
* GitHub Actions
* AWS EC2

---

## 📁 Project Structure

data/
models/
notebooks/
reports/
src/
tests/
requirements.txt
Dockerfile
dvc.yaml
README.md

---

## ⚡ How to Run

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run pipeline:
```bash
dvc repro
```
---

## 🔄 End-to-End Flow
```bash
Data → Preprocessing → Feature Engineering → Model Training → MLflow → DVC Pipeline → Model Registry → Flask API → Docker → CI/CD → AWS EC2
```
---


## 💡 Conclusion

This project demonstrates a complete MLOps lifecycle focusing on reproducibility, scalability, automation, and production readiness.

---


