import joblib
import numpy as np

MODEL_PATH = "models/tennis_rf.pkl"

def load_model(path=MODEL_PATH):
    """
    Load the trained RandomForest model from disk.
    """
    return joblib.load(path)

def predict_from_features(features, model):
    """
    Given a feature vector and a trained model,
    return the probability that player A wins.
    """
    arr = np.array(features).reshape(1, -1)
    proba = model.predict_proba(arr)[0][1]
    return float(proba)