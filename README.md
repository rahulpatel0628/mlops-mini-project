# Emotion Detection from Text (MLOps Project)

## Overview

This project focuses on building an **end-to-end MLOps pipeline** for detecting emotions from textual data. It covers everything from data processing to model deployment readiness using modern industry tools.

---

## Key Features

* Complete ML Pipeline (Data → Model → Deployment Ready)
* Experiment Tracking using MLflow + DagsHub
* Data & Pipeline Versioning using DVC + AWS S3
* Automated Workflow using DVC Pipelines
* Multiple Model Comparison & Hyperparameter Tuning
* Model Versioning with MLflow Model Registry (Staging)

---

## Project Workflow

### 1️⃣ Data Ingestion

* Load raw text dataset

### 2️⃣ Data Preprocessing

* Text cleaning
* Tokenization
* Stopword removal

### 3️⃣ Feature Engineering

* Bag of Words (CountVectorizer)
* TF-IDF (TfidfVectorizer)

---

### 4️⃣ Baseline Models

* Built models using:

  * BOW
  * TF-IDF

* Compared performance

---

### 5️⃣ Model Training

Trained multiple models:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* Naive Bayes

---

### 6️⃣ Model Selection

* Best Model: **Logistic Regression with BOW**
* Accuracy: ~80%

---

### 7️⃣ Hyperparameter Tuning

Optimized:

* `C = 1.0`
* `solver = liblinear`
* `penalty = l2`

---

### 8️⃣ Experiment Tracking

* Used **MLflow with DagsHub**
* Logged:

  * Parameters
  * Metrics
  * Models
  * Artifacts

---

### 9️⃣ Pipeline Automation (DVC)

* Created modular pipeline:

  * data_ingestion
  * preprocessing
  * feature_engineering
  * model_training
  * evaluation
  * registration

---

### 🔟 Data Versioning

* DVC integrated with **AWS S3**
* Ensures reproducibility

---

### 1️⃣1️⃣ Model Registration

* Registered model using **run_id**
* Promoted to **Staging stage**

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* MLflow
* DagsHub
* DVC
* AWS S3
* Git

---

## 📁 Project Structure

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── data_ingestion.py
             └── data_preprocessing.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── feature_engineering.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── model_building.py
    │   │   └── model_evaluation.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


---

## ⚡ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
dvc repro
```

---

## 📊 Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | ~0.78 |
| Precision | ~0.76 |
| Recall    | ~0.80 |
| F1 Score  | ~0.78 |

---

## 🎯 Future Improvements

* Deploy using FastAPI
* Add CI/CD pipeline
* Implement model monitoring
* Add real-time inference

---

## 🙌 Conclusion

This project demonstrates a **complete MLOps lifecycle**, focusing on reproducibility, scalability, and real-world best practices.