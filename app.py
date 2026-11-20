import os
import re
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "tweets_dataset.csv")

# Load models and vectorizer
vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
models = {
    "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
    "Naive Bayes": joblib.load(os.path.join(MODELS_DIR, "naive_bayes.joblib")),
    "Linear SVM": joblib.load(os.path.join(MODELS_DIR, "linear_svm.joblib"))
}

# Preprocessing function matching the ML pipeline
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", " ", t)       # remove links
    t = re.sub(r"@\w+", " ", t)                 # remove mentions
    t = re.sub(r"#", " ", t)                    # keep hashtag word
    t = re.sub(r"[^a-z\s-]", " ", t)            # remove punctuation/digits
    return re.sub(r"\s+", " ", t).strip()

# Softmax for SVM decision function to estimate pseudo-probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    model_name = data.get("model", "Logistic Regression")

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    if model_name not in models:
        model_name = "Logistic Regression"

    cleaned = clean_text(text)
    model = models[model_name]
    transformed = vectorizer.transform([cleaned])

    # Predict class
    prediction = model.predict(transformed)[0]

    # Calculate probabilities/confidence
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(transformed)[0].tolist()
    else:
        # For LinearSVM, use decision_function with softmax
        dfunc = model.decision_function(transformed)[0]
        probs = softmax(dfunc)

    confidence_dict = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, probs)}

    # Highlight vocabulary words detected in text
    feature_names = set(vectorizer.get_feature_names_out())
    tokens = cleaned.split()
    detected_words = [t for t in tokens if t in feature_names]

    return jsonify({
        "original_text": text,
        "clean_text": cleaned,
        "model_used": model_name,
        "sentiment": prediction,
        "confidence": confidence_dict,
        "detected_features": detected_words
    })

@app.route("/api/metrics", methods=["GET"])
def metrics():
    # Performance benchmarks directly from evaluation
    return jsonify({
        "models": [
            {
                "name": "Logistic Regression",
                "test_accuracy": 0.8978,
                "cv_accuracy": 0.8719,
                "precision": {"negative": 0.90, "neutral": 0.89, "positive": 0.90},
                "recall": {"negative": 0.89, "neutral": 0.89, "positive": 0.91},
                "f1_score": {"negative": 0.90, "neutral": 0.89, "positive": 0.91},
                "best": True
            },
            {
                "name": "Naive Bayes",
                "test_accuracy": 0.8978,
                "cv_accuracy": 0.8719,
                "precision": {"negative": 0.90, "neutral": 0.89, "positive": 0.90},
                "recall": {"negative": 0.89, "neutral": 0.89, "positive": 0.91},
                "f1_score": {"negative": 0.90, "neutral": 0.89, "positive": 0.91},
                "best": False
            },
            {
                "name": "Linear SVM",
                "test_accuracy": 0.8911,
                "cv_accuracy": 0.8585,
                "precision": {"negative": 0.89, "neutral": 0.88, "positive": 0.90},
                "recall": {"negative": 0.88, "neutral": 0.88, "positive": 0.91},
                "f1_score": {"negative": 0.89, "neutral": 0.88, "positive": 0.90},
                "best": False
            }
        ],
        "dataset_summary": {
            "total_samples": 1800,
            "classes": {"positive": 600, "negative": 600, "neutral": 600},
            "train_size": 1350,
            "test_size": 450,
            "label_noise": "18%"
        },
        "top_words": {
            "negative": ["horrible", "awful", "waste", "useless", "worst", "hate", "disappointed"],
            "neutral": ["typical", "regular", "standard", "average", "fine", "usual", "okay"],
            "positive": ["best", "brilliant", "great", "awesome", "enjoyed", "love", "amazing"]
        }
    })

@app.route("/api/samples", methods=["GET"])
def samples():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        sample_rows = df.groupby("sentiment").head(4).to_dict(orient="records")
        return jsonify(sample_rows)
    return jsonify([])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

