import string
import joblib

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = "document_classifier.pkl"
DATA_PATH = "data.csv"

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def preprocess(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


def load_data(path):
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("data.csv must contain 'text' and 'label' columns")
    df = df.dropna(subset=['text', 'label']).copy()
    df['text'] = df['text'].apply(preprocess)
    return df


def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])


def train_model(df):
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))
    return model


def main():
    df = load_data(DATA_PATH)
    model = train_model(df)

    sample_docs = [
        "Software Developer skilled in Python and Java",
        "Invoice amount due Rs 10000",
        "Machine Learning research paper on NLP"
    ]
    predictions = model.predict(sample_docs)
    for doc, label in zip(sample_docs, predictions):
        print(f"Document: {doc}")
        print(f"Predicted Category: {label}\n")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    text = ["Patient medical report showing symptoms of flu"]
    prediction = loaded_model.predict(text)
    print("Category:", prediction[0])


if __name__ == '__main__':
    main()
