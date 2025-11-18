import sys
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib


def load_dataset(path="data/processed/dataset_basic_features.npz"):
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    return X, y


def train_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return model, acc, auc


def save_model(model, path="models/tennis_rf.pkl"):
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset()

    print("Training model...")
    model, acc, auc = train_model(X, y)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC AUC:  {auc:.4f}")

    save_model(model)