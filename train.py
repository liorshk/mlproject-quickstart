import click
from ml.data_source.titanic import Titanic
from ml.model.metrics import Metrics

import mlflow

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

@click.command()
@click.option("--max-depth", default=3, help="Max depth for decision tree")
def train(max_depth):
    np.random.seed(0)

    data_train = Titanic('data/raw/train.csv')
    data_test = Titanic('data/raw/test.csv')

    X_train, y_train = data_train.get_features(), data_train.get_label()
    X_test, y_test = data_test.get_features(), data_test.get_label()

    params = {
        "max_depth": max_depth
    }

    model = DecisionTreeClassifier(**params)

    with mlflow.start_run() as run:
        
        mlflow.log_params(params)

        mlflow.set_tags({
            "training_nrows": X_train.shape[0],
            "training_label_ratio": y_train.value_counts(normalize=True)[1],
            "test_nrows": X_test.shape[0],
            "test_label_ratio": y_test.value_counts(normalize=True)[1]
        })
        
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = Metrics.classification(y_test, (y_pred_proba > 0.5).astype(int), y_pred_proba)
        mlflow.log_metrics(metrics)
        
        mlflow.sklearn.log_model(model, 'model')


if __name__ == "__main__":
    train()
