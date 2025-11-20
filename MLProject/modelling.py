#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CI Modelling — FIXED VERSION (NO PREPROCESSING)
Author: Yoga Fatiqurrahman
Tujuan:
- Workflow CI hanya load dataset hasil preprocessing
- Tidak ada SimpleImputer, StandardScaler, atau transformasi apa pun
"""

import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

def load_dataset(data_dir: str):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X, y)
    return model


def evaluate(model, X, y, prefix):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        f"{prefix}_accuracy": accuracy_score(y, preds),
        f"{prefix}_precision": precision_score(y, preds),
        f"{prefix}_recall": recall_score(y, preds),
        f"{prefix}_f1": f1_score(y, preds),
        f"{prefix}_roc_auc": roc_auc_score(y, probs)
    }, confusion_matrix(y, preds)


def main(args):
    run = mlflow.active_run()
    if not run:
        print("[ERROR] Tidak ada active_run dari MLflow Project.")
        return

    run_id = run.info.run_id

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.data_dir)

    model = train_model(X_train, y_train)

    val_metrics, val_cm = evaluate(model, X_val, y_val, "val")
    test_metrics, test_cm = evaluate(model, X_test, y_test, "test")

    client = mlflow.tracking.MlflowClient()

    client.log_param(run_id, "dataset_dir", args.data_dir)

    for k, v in val_metrics.items():
        client.log_metric(run_id, k, float(v))
    for k, v in test_metrics.items():
        client.log_metric(run_id, k, float(v))

    mlflow.sklearn.log_model(model, artifact_path="model")

    client.log_dict(run_id, {"val_cm": val_cm.tolist()}, "val_cm.json")
    client.log_dict(run_id, {"test_cm": test_cm.tolist()}, "test_cm.json")

    print("\n CI TRAINING SUCCESS ")
    print("VAL:", val_metrics)
    print("TEST:", test_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    main(args)