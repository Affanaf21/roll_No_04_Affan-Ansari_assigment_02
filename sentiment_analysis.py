# ============================================================
# Assignment 2 - Sentiment Analysis on Spider-Man No Way Home Tweets
# Course: Data Analytics and Visualisation (CSC601)
# ============================================================

# ── STEP 1: Install & Import Libraries ──────────────────────
# Run this in Google Colab if needed:
# !pip install scikit-learn pandas matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, precision_score,
    recall_score, confusion_matrix, accuracy_score
)

# ── STEP 2: Load Dataset ────────────────────────────────────
df = pd.read_csv('tweets.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# ── STEP 3: Preprocessing ───────────────────────────────────
def preprocess(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'@\w+', '', text)             # remove mentions
    text = re.sub(r'#\w+', '', text)             # remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)         # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()     # remove extra spaces
    return text

df['clean_tweet'] = df['tweet'].apply(preprocess)
print("\nSample cleaned tweets:")
print(df[['tweet', 'clean_tweet']].head(3))

# ── STEP 4: Split Dataset (80 train / 20 test) ──────────────
X = df['clean_tweet']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size:  {len(X_test)}")

# ── STEP 5: TF-IDF Vectorization ────────────────────────────
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── STEP 6: Train Classifiers ───────────────────────────────
classifiers = {
    'Naive Bayes':       MultinomialNB(),
    'SVM':               SVC(kernel='linear', C=1.0, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}

for name, clf in classifiers.items():
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy  = accuracy_score(y_test, y_pred)

    results[name] = {
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'accuracy':  round(accuracy, 4),
        'y_pred':    y_pred
    }

    print(f"\n{'='*50}")
    print(f"Classifier: {name}")
    print(f"{'='*50}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# ── STEP 7: Comparison Table ────────────────────────────────
print("\n" + "="*55)
print("CLASSIFIER PERFORMANCE COMPARISON")
print("="*55)
print(f"{'Classifier':<25} {'Precision':>10} {'Recall':>10} {'Accuracy':>10}")
print("-"*55)
for name, r in results.items():
    print(f"{name:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['accuracy']:>10.4f}")

# ── STEP 8: Visualizations ──────────────────────────────────

# Plot 1: Sentiment Distribution
plt.figure(figsize=(6, 4))
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'], edgecolor='black')
plt.title('Sentiment Distribution of Spider-Man No Way Home Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=150)
plt.show()
print("Saved: sentiment_distribution.png")

# Plot 2: Precision & Recall Bar Chart
clf_names = list(results.keys())
precisions = [results[n]['precision'] for n in clf_names]
recalls    = [results[n]['recall']    for n in clf_names]

x = range(len(clf_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar([i - width/2 for i in x], precisions, width, label='Precision', color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], recalls,    width, label='Recall',    color='coral')

ax.set_xlabel('Classifier')
ax.set_ylabel('Score')
ax.set_title('Precision vs Recall for Each Classifier')
ax.set_xticks(list(x))
ax.set_xticklabels(clf_names)
ax.set_ylim(0, 1.1)
ax.legend()

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('precision_recall_comparison.png', dpi=150)
plt.show()
print("Saved: precision_recall_comparison.png")

# Plot 3: Confusion Matrix for best classifier
best_clf_name = max(results, key=lambda n: results[n]['accuracy'])
best_y_pred   = results[best_clf_name]['y_pred']
labels = ['negative', 'neutral', 'positive']

cm = confusion_matrix(y_test, best_y_pred, labels=labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,
            cmap='Blues', linewidths=0.5)
plt.title(f'Confusion Matrix — {best_clf_name} (Best Classifier)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print(f"Saved: confusion_matrix.png")

print(f"\n✅ Best Classifier: {best_clf_name}")
print(f"   Accuracy : {results[best_clf_name]['accuracy']:.4f}")
print(f"   Precision: {results[best_clf_name]['precision']:.4f}")
print(f"   Recall   : {results[best_clf_name]['recall']:.4f}")
