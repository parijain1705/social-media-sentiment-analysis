# Social Media Sentiment Analysis - Machine Learning Lab Project
import re, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import joblib
import os

random.seed(42)
np.random.seed(42)

# 1. DATASET: synthetic tweets (no internet needed) ----------------------
POS = ["love", "amazing", "great", "awesome", "fantastic", "happy", "best", "brilliant", "excellent", "enjoyed"]
NEG = ["hate", "terrible", "awful", "worst", "disappointed", "horrible", "annoying", "broken", "useless", "waste"]
NEU = ["okay", "average", "fine", "normal", "usual", "decent", "regular", "standard", "so-so", "typical"]
SUBJ = ["the new phone", "this app", "the update", "the movie", "our delivery", "the concert",
        "this game", "the new cafe", "the service", "the latest episode"]
TAGS = ["#tech", "#life", "#mondaymood", "#review", "#trending", "#india", ""]
TEMPL = ["Honestly {s} is {w} {t}", "{w} experience with {s}! {t}", "Just tried {s}, it was {w} {t}",
         "{s} feels {w} today @brand {t}", "My take on {s}: {w} http://t.co/xyz {t}"]

def make_post(label):
    pool = {"positive": POS, "negative": NEG, "neutral": NEU}[label]
    w = random.choice(pool)
    if random.random() < 0.18: # noise: borrow a word from another class
        w = random.choice(POS + NEG + NEU)
    return random.choice(TEMPL).format(s=random.choice(SUBJ), w=w, t=random.choice(TAGS)).strip()

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("static/figures", exist_ok=True)

    rows = [(make_post(l), l) for l in ["positive", "negative", "neutral"] for _ in range(600)]
    df = pd.DataFrame(rows, columns=["text", "sentiment"]).sample(frac=1, random_state=1).reset_index(drop=True)
    df.to_csv("tweets_dataset.csv", index=False)
    print("Dataset shape:", df.shape)
    print(df.head(5).to_string(), "\n")
    print("Class distribution:\n", df["sentiment"].value_counts(), "\n")

    # 2. PREPROCESSING --------------------------------------------------------
    def clean(t):
        t = t.lower()
        t = re.sub(r"http\S+|www\S+", " ", t)        # remove links
        t = re.sub(r"@\w+", " ", t)                  # remove mentions
        t = re.sub(r"#", " ", t)                     # keep hashtag word
        t = re.sub(r"[^a-z\s-]", " ", t)             # remove punctuation/digits
        return re.sub(r"\s+", " ", t).strip()

    df["clean"] = df["text"].apply(clean)
    print("Before:", df["text"][0])
    print("After :", df["clean"][0], "\n")

    # 3. TRAIN / TEST SPLIT + TF-IDF ------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(df["clean"], df["sentiment"], test_size=0.25,
                                              stratify=df["sentiment"], random_state=42)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)
    print("Train:", Xtr.shape, " Test:", Xte.shape, "\n")

    # 4. MODELS -----------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42)
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, y_tr)
        pred = m.predict(Xte)
        cv = cross_val_score(m, Xtr, y_tr, cv=5).mean()
        results[name] = (accuracy_score(y_te, pred), cv, pred)
        print(f"{name:20s} Test Acc = {results[name][0]:.4f} | 5-fold CV = {cv:.4f}")

    best = max(results, key=lambda k: results[k][0])
    print("\nBest model:", best, "\n")
    print(classification_report(y_te, results[best][2]))

    # Save models and vectorizer
    joblib.dump(vec, "models/tfidf_vectorizer.joblib")
    joblib.dump(models["Logistic Regression"], "models/logistic_regression.joblib")
    joblib.dump(models["Naive Bayes"], "models/naive_bayes.joblib")
    joblib.dump(models["Linear SVM"], "models/linear_svm.joblib")
    print("Saved all trained models and vectorizer to models/")

    # 5. VISUALS -----------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    navy, teal, grey = "#1F3A5F", "#2A9D8F", "#B0B7C3"

    # Figure 1
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    counts = df["sentiment"].value_counts()
    ax[0].bar(counts.index, counts.values, color=[teal, navy, grey])
    ax[0].set_title("Class distribution")
    ax[0].set_ylabel("Posts")
    accs = [results[k][0] for k in results]
    ax[1].bar(list(results), accs, color=navy)
    ax[1].set_ylim(0.5, 1.0)
    ax[1].set_title("Model accuracy (test set)")
    ax[1].tick_params(axis="x", rotation=15)
    for i, a in enumerate(accs):
        ax[1].text(i, a + 0.01, f"{a:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("fig1.png", dpi=170)
    plt.savefig("static/figures/fig1.png", dpi=170)
    plt.close()

    # Figure 2
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_te, results[best][2], labels=models[best].classes_),
                           display_labels=models[best].classes_).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {best}")
    plt.tight_layout()
    plt.savefig("fig2.png", dpi=170)
    plt.savefig("static/figures/fig2.png", dpi=170)
    plt.close()

    # 6. TOP WORDS PER CLASS (Logistic Regression) ---------------------------
    lr = models["Logistic Regression"]
    names = np.array(vec.get_feature_names_out())
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for i, cls in enumerate(lr.classes_):
        idx = np.argsort(lr.coef_[i])[-7:]
        axes[i].barh(names[idx], lr.coef_[i][idx], color=[teal, navy, grey][i])
        axes[i].set_title(f"Top words: {cls}")
        print(f"Top words for {cls:9s}:", ", ".join(names[idx][::-1]))
    plt.tight_layout()
    plt.savefig("fig3.png", dpi=170)
    plt.savefig("static/figures/fig3.png", dpi=170)
    plt.close()

    # 7. PREDICT NEW POSTS ----------------------------------------------------
    new = ["Absolutely love this new update, fantastic work!",
           "Worst delivery ever, totally disappointed #angry",
           "The movie was okay, nothing special",
           "this app is awesome but sometimes broken"]
    print("\nPredictions on unseen posts:")
    for t in new:
        print(f" [{lr.predict(vec.transform([clean(t)]))[0]:8s}] {t}")

if __name__ == "__main__":
    main()
