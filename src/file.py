import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")
wine =load_wine()
X=wine.data
y=wine.target

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.10,random_state=42)
max_depth=5
n_estimators=10
print(mlflow.get_tracking_uri())
with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)

    acc=accuracy_score(y_pred,y_test)
    mlflow.log_metric("accuarcy",acc)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_artifact(__file__)
    mlflow.set_tags({"Author":"Manya","Project":"Ml_Flow Exp"})
    mlflow.sklearn.log_model(rf,"RandomForestClassifer_MODEL1")

    print(acc)