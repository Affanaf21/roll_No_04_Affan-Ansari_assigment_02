# Sentiment Analysis on Spider-Man No Way Home Tweets

## (1) Problem Statement
Social media platforms like Twitter generate massive amounts of opinionated text data every day. Analysing this data manually is impractical. This assignment addresses the problem of automatically classifying the sentiment of tweets related to the movie **Spider-Man: No Way Home** into three categories — **Positive**, **Neutral**, and **Negative** — using machine learning classifiers.

## (2) Objective
- Collect 100 tweets related to Spider-Man No Way Home and manually label them by sentiment.
- Preprocess the raw tweet text using standard NLP techniques.
- Train and evaluate multiple classifiers (Naive Bayes, SVM, Logistic Regression) on the labelled dataset.
- Compare classifier performance using Precision and Recall metrics to identify the best model.

## (3) Dataset
- **Source:** Manually collected tweets from Twitter/X related to "Spider-Man No Way Home"
- **Features:** `tweet` (raw text), `sentiment` (positive / neutral / negative)
- **Size:** 100 tweets — 48 Positive, 26 Neutral, 26 Negative
- **Split:** 80 tweets for training, 20 tweets for testing (80/20 split)

## (4) Methodology
1. **Data Preprocessing** — Lowercasing, URL removal, mention/hashtag removal, special character stripping, whitespace normalization
2. **Feature Extraction** — TF-IDF Vectorizer with `max_features=500` and `ngram_range=(1,2)`
3. **Model Building** — Three classifiers trained: Multinomial Naive Bayes, SVM (linear kernel), Logistic Regression
4. **Evaluation** — Precision, Recall, and Accuracy calculated for each classifier on the 20-tweet test set

## (5) Results

| Classifier           | Precision | Recall | Accuracy |
|----------------------|-----------|--------|----------|
| Naive Bayes          | 0.6397    | 0.5500 | 55.00%   |
| SVM (Linear)         | **0.7565**| **0.7000** | **70.00%** |
| Logistic Regression  | 0.6827    | 0.6000 | 60.00%   |

**Best Classifier: SVM** with Precision = 0.7565 and Recall = 0.7000

## (6) How to Run
```bash
pip install scikit-learn pandas matplotlib seaborn
python sentiment_analysis.py
```
> Make sure `tweets.csv` is in the same directory as `sentiment_analysis.py` before running.

## (7) Conclusion
The experiment successfully demonstrated sentiment analysis on movie-related tweets. Among the three classifiers, **SVM with a linear kernel performed the best**, achieving 70% accuracy, 0.7565 precision, and 0.7000 recall. This is because SVM is well-suited to high-dimensional TF-IDF feature spaces. Naive Bayes was the weakest performer due to its independence assumption not holding well for short tweet text. Future improvements include using larger datasets, BERT embeddings, and hyperparameter tuning via GridSearchCV.

## (8) Student's Details
- **Name:** Affan Ansari
- **Roll No:** 04
- **UIN:** 231A023
- **Year:** TE-AIDS
