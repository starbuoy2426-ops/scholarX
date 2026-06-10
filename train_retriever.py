import os
import pickle
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FOLDER = "data"
MODEL_FILE = "retriever.pkl"


def load_documents(folder: str) -> Tuple[List[str], List[str]]:
    if os.path.isdir(folder):
        base_folder = folder
        candidates = sorted(os.listdir(folder))
    else:
        base_folder = "."
        print(f"Warning: Folder not found: {folder}. Falling back to text files in current directory.")
        candidates = sorted(
            f for f in os.listdir(base_folder)
            if f.lower().endswith(".txt") and os.path.isfile(os.path.join(base_folder, f))
        )

    documents = []
    filenames = []

    for filename in candidates:
        path = os.path.join(base_folder, filename)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                documents.append(text)
                filenames.append(filename)

    if not documents:
        raise ValueError(f"No text files found in folder: {folder}")

    return documents, filenames


def train_retriever(folder: str = DATA_FOLDER, model_file: str = MODEL_FILE):
    documents, filenames = load_documents(folder)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    with open(model_file, "wb") as f:
        pickle.dump((vectorizer, X, documents, filenames), f)

    print("Retriever trained successfully.")
    return vectorizer, X, documents, filenames


def load_retriever(model_file: str = MODEL_FILE):
    with open(model_file, "rb") as f:
        vectorizer, X, documents, filenames = pickle.load(f)
    return vectorizer, X, documents, filenames


def retrieve(query: str, vectorizer: TfidfVectorizer, X, documents: List[str], filenames: List[str], top_k: int = 1):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, X)[0]

    best_indices = similarities.argsort()[::-1][:top_k]
    return [(filenames[i], documents[i], float(similarities[i])) for i in best_indices]


def chatbot():
    if not os.path.exists(MODEL_FILE):
        train_retriever()

    vectorizer, X, documents, filenames = load_retriever()

    print("Retrieval Chatbot Started")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        results = retrieve(query, vectorizer, X, documents, filenames, top_k=1)
        filename, context, score = results[0]
        response = f"Based on retrieved document '{filename}' (score={score:.3f}):\n\n{context}"

        print("\nBot:", response)
        print()


if __name__ == "__main__":
    chatbot()