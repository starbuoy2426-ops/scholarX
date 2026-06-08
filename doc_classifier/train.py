"""
train.py
Train a simple text document classifier using TF-IDF + LogisticRegression.
Saves a trained pipeline to a joblib file.
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def train_and_save(csv_path, model_path):
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    X = df['text'].astype(str)
    y = df['label'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train document classifier')
    parser.add_argument('--data', default='doc_classifier/sample_data.csv', help='Path to CSV with text,label')
    parser.add_argument('--out', default='doc_classifier/model.joblib', help='Output model path')
    args = parser.parse_args()

    train_and_save(args.data, args.out)


if __name__ == '__main__':
    main()
