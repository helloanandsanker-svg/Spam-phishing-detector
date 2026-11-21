import argparse
from pathlib import Path
import time

t_start = time.perf_counter()


def log_step(label: str, step_start: float) -> None:
    elapsed = time.perf_counter() - step_start
    total = time.perf_counter() - t_start
    print(f"[TIME] {label}: {elapsed:.2f}s (total: {total:.2f}s)")


t_imp = time.perf_counter()
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
log_step("Import libraries (sklearn/pandas/joblib)", t_imp)

from preprocess import clean_text

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "messages.csv"
MODELS_DIR = ROOT_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMS/Email Spam & Phishing Detector")
    parser.add_argument(
        "--model",
        choices=["logreg", "nb"],
        default="logreg",
        help="Classifier to train: 'logreg' (LogisticRegression) or 'nb' (fast MultinomialNB)",
    )
    return parser.parse_args()


def build_classifier(model_type: str):
    if model_type == "nb":
        return MultinomialNB(alpha=0.5)
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def main() -> None:
    args = parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Add a CSV with 'text' and 'label' columns."
        )

    t_step = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).copy()
    df["target"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(lambda x: 0 if x == "ham" else 1)
    )
    log_step("Load and prepare dataset", t_step)

    t_step = time.perf_counter()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    log_step("Clean and preprocess text", t_step)

    t_step = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    log_step("Split data and fit TF-IDF vectorizer", t_step)

    t_step = time.perf_counter()
    model = build_classifier(args.model)
    model.fit(X_train_vec, y_train)
    log_step(f"Train classifier ({args.model})", t_step)

    t_step = time.perf_counter()
    y_pred = model.predict(X_test_vec)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(
        "\nClassification Report:\n",
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate (Ham)", "Spam / Phishing"],
        ),
    )
    log_step("Evaluate on test set", t_step)

    t_step = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(model, MODELS_DIR / "spam_model.joblib")
    log_step(f"Save artifacts to {MODELS_DIR}", t_step)


if __name__ == "__main__":
    main()