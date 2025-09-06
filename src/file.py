import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 

import dagshub
dagshub.init(repo_owner='aHp16', repo_name='MLFlow_Dagshub', mlflow=True,token=os.getenv("DAGSHUB_TOKEN"))

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Model hyperparameters
max_depth = 5
n_estimators = 10
mlflow.autolog()

# MLflow experiment
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Log metrics and params
    # mlflow.log_metric("accuracy", acc)
    # mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("n_estimators", n_estimators)

    # If you want to log a file, specify it explicitly instead of __file__
    # mlflow.log_artifact("your_script.py")

    mlflow.set_tags({"Author": "Manya", "Project": "ML_Flow Exp"})
    # mlflow.sklearn.log_model(rf, artifact_path="RandomForestClassifier_MODEL1")

    print("Accuracy:", acc)
