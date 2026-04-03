import pandas as pd
import joblib
import logging
import mlflow
import mlflow.sklearn
import dagshub
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json
import os

TARGET = "sentiment"   # change if needed

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# -------------------- MLflow Setup --------------------
def setup_mlflow():
    try:
        dagshub.init(
            repo_owner='rahulpatel16092005',
            repo_name='mlops-mini-project',
            mlflow=True
        )

        mlflow.set_tracking_uri(
            "https://dagshub.com/rahulpatel16092005/mlops-mini-project.mlflow"
        )

        mlflow.set_experiment("DVC Pipeline Classification")
        logger.info("MLflow setup completed")

    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        raise


# -------------------- Load Data --------------------
def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# -------------------- Split Data --------------------
def split_X_y(df: pd.DataFrame):
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


# -------------------- Load Model --------------------
def load_model(path: Path):
    try:
        model = joblib.load(path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# -------------------- Evaluate Model --------------------
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba)
        }

        logger.info("Model evaluation completed")
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


# -------------------- MLflow Logging --------------------
def log_to_mlflow(model, metrics, test_df, root_path: Path):
    try:
        with mlflow.start_run() as run:

            mlflow.set_tag("model", "Classification Model")

            # log parameters
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            # log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # # log input dataset
            # test_input = mlflow.data.from_pandas(test_df, targets=TARGET)
            # mlflow.log_input(test_input, context="testing")

            # model signature
            signature = mlflow.models.infer_signature(
                model_input=test_df.iloc[:, :-1].sample(20, random_state=42),
                model_output=model.predict(test_df.iloc[:, :-1].sample(20))
            )

            temp_path = root_path / "models" / "mlflow_model"

            if os.path.exists(temp_path):
                import shutil
                shutil.rmtree(temp_path)

            mlflow.sklearn.save_model(
                sk_model=model,
                path=temp_path,
                signature=signature
            )

            mlflow.log_artifacts(temp_path, artifact_path="model")

            # log original model
            mlflow.log_artifact(root_path / "models" / "model.pkl")

            artifact_uri = mlflow.get_artifact_uri()

            logger.info("MLflow logging completed")
            return run.info.run_id, artifact_uri

    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")
        raise


# -------------------- Save Run Info --------------------
def save_run_info(path: Path, run_id: str, artifact_uri: str):
    try:
        info = {
            "run_id": run_id,
            "artifact_path": artifact_uri,
            "model_name": "model"
        }

        with open(path, "w") as f:
            json.dump(info, f, indent=4)

        logger.info("Run information saved")

    except Exception as e:
        logger.error(f"Error saving run info: {e}")
        raise


# -------------------- Main Pipeline --------------------
def main():
    try:
        setup_mlflow()

        root = Path(__file__).parent.parent.parent

        test_path = root / "data" / "features" / "test_bow.csv"
        model_path = root / "models" / "model.pkl"

        test_df = load_data(test_path)

        X_test, y_test = split_X_y(test_df)

        model = load_model(model_path)

        metrics = evaluate_model(model, X_test, y_test)

        run_id, artifact_uri = log_to_mlflow(
            model,
            metrics,
            test_df,
            root
        )

        save_run_info(
            root / "reports" / "experiment_info.json",
            run_id,
            artifact_uri
        )

    except Exception as e:
        logger.critical(f"Evaluation pipeline failed: {e}")


if __name__ == "__main__":
    main()