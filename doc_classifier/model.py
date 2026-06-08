"""
model.py
Helper functions to load a trained classifier and run predictions.
"""
import joblib
from typing import List


def load_model(path='doc_classifier/model.joblib'):
    return joblib.load(path)


def predict(texts: List[str], model_path='doc_classifier/model.joblib'):
    model = load_model(model_path)
    return model.predict(texts)


def predict_proba(texts: List[str], model_path='doc_classifier/model.joblib'):
    model = load_model(model_path)
    return model.predict_proba(texts)
