# Social Media Sentiment Analysis

A supervised Machine Learning project and interactive web application that classifies social media posts and tweets into **positive**, **negative**, or **neutral** sentiments using **TF-IDF n-grams** and machine learning classifiers.

---

## 📌 Project Overview

Social media platforms produce massive volumes of short, informal text. This project implements an end-to-end NLP pipeline:
1. **Offline & Reproducible Synthetic Data Generation**: Generates 1,800 tweet-style posts (600 per class) with realistic hashtags, @mentions, URLs, and deliberate label noise (18%) to test model robustness.
2. **Text Preprocessing**: Normalizes case, strips URLs, handles mentions, removes punctuation/numbers, while retaining meaningful hashtag keywords.
3. **Feature Engineering**: TF-IDF vectorization with unigrams and bigrams (`ngram_range=(1, 2)`, `min_df=2`).
4. **Model Comparison**: Benchmarks **Logistic Regression**, **Multinomial Naive Bayes**, and **Linear SVM**.
5. **Interactive Web Application**: Full-stack web interface (Flask backend + responsive UI frontend) for real-time post classification, probability breakdowns, and visual metrics exploration.

---

## 🏆 Performance Benchmarks

Evaluated on 450 unseen test posts using a 75/25 stratified split and 5-fold cross-validation:

| Model | Test Accuracy | 5-Fold CV Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Notes |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **89.78%** | **87.19%** | **0.90** | **0.90** | **0.90** | **Selected Best Model** (Interpretable weights) |
| **Multinomial Naive Bayes** | **89.78%** | **87.19%** | **0.90** | **0.90** | **0.90** | Tied accuracy |
| **Linear SVM** | 89.11% | 85.85% | 0.89 | 0.89 | 0.89 | Fast and competitive |

### Top Learned Indicators
- **Positive**: `best`, `brilliant`, `great`, `awesome`, `enjoyed`, `love`, `amazing`
- **Neutral**: `typical`, `regular`, `standard`, `average`, `fine`, `usual`, `okay`
- **Negative**: `horrible`, `awful`, `waste`, `useless`, `worst`, `hate`, `disappointed`

---

## 📊 Result Visualisations

- **Figure 1**: Balanced class distribution (600 per class) and test set accuracy comparison across models (`fig1.png`).
- **Figure 2**: Confusion matrix of the best model (Logistic Regression) on 450 test posts (`fig2.png`).
- **Figure 3**: Top 7 highest weighted features per sentiment class (`fig3.png`).

---

## 📁 Repository Structure

```
├── app.py                      # Flask backend API & web server
├── sentiment_analysis.py       # Standalone ML training & evaluation script
├── tweets_dataset.csv          # Generated 1,800-sample tweets dataset
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
├── models/                     # Saved trained models & vectorizer
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   └── linear_svm.joblib
├── static/
│   ├── css/
│   │   └── style.css           # Modern custom styling
│   ├── js/
│   │   └── app.js              # Real-time inference frontend scripts
│   └── figures/                # Visualizations served in the web app
│       ├── fig1.png
│       ├── fig2.png
│       └── fig3.png
├── templates/
│   └── index.html              # Main responsive web application template
├── fig1.png                    # Distribution & accuracy comparison plot
├── fig2.png                    # Confusion matrix plot
└── fig3.png                    # Top feature weights plot
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+ installed
- pip package manager

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 3. Run the ML Pipeline
To re-run the full training, cross-validation, and generate new evaluation figures:
```bash
python sentiment_analysis.py
```

### 4. Launch the Web Application
Start the local web server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🌐 Web App Features

- **Real-Time Classification**: Type any tweet or select sample inputs to instantly predict sentiment.
- **Model Selector**: Switch dynamically between Logistic Regression, Naive Bayes, and Linear SVM.
- **Confidence Breakdown**: Live percentage progress bars displaying class probabilities.
- **Preprocess & Feature Tracing**: Inspect the cleaned tokenized text and detected TF-IDF vocabulary words.
- **Visual Analytics Tabs**: Seamlessly view Figure 1, Figure 2, and Figure 3 right inside the UI.

---

## 👤 Author
- **Pari Jain**

