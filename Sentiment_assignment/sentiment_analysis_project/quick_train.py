"""
Quick training script to create a baseline TF-IDF + LogisticRegression model
Saves:
 - models/best_sentiment_model.pkl
 - models/tfidf_extractor.pkl
 - models/model_metadata.json
"""

import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Use project's preprocessing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import TextPreprocessor

DATA_PATH = r'..\reviews_data_dump\reviews_badminton\data.csv'
# Adjust path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'reviews_data_dump', 'reviews_badminton', 'data.csv')

models_dir = os.path.join(script_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

print('Loading data...')
try:
    df = pd.read_csv(data_path)
except Exception as e:
    # try absolute path
    data_path2 = os.path.join(script_dir, 'data', 'preprocessed_reviews.csv')
    raise FileNotFoundError(f"Could not load dataset from {data_path}: {e}")

# Create sentiment label if not present
if 'Sentiment' not in df.columns:
    df['Sentiment'] = (df['Ratings'] >= 3).astype(int)

# Drop NaNs in review text
df = df.dropna(subset=['Review text']).reset_index(drop=True)

print(f'Total reviews: {len(df)}')

# Preprocess text
preprocessor = TextPreprocessor(use_lemmatization=True)
print('Cleaning texts (this may take a while)...')
df['cleaned_text'] = df['Review text'].apply(lambda x: preprocessor.clean_text(str(x), remove_stopwords=True, normalize=True))

# Remove very short texts
df = df[df['cleaned_text'].str.split().str.len() >= 2].reset_index(drop=True)

# Feature extraction TF-IDF
print('Fitting TF-IDF...')
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'].values)
y = df['Sentiment'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression
print('Training Logistic Regression...')
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1_score': float(f1_score(y_test, y_pred))
}
print('Evaluation metrics:')
print(metrics)

# Save model and extractor
model_path = os.path.join(models_dir, 'best_sentiment_model.pkl')
extractor_path = os.path.join(models_dir, 'tfidf_extractor.pkl')
metadata_path = os.path.join(models_dir, 'model_metadata.json')

joblib.dump(clf, model_path)
joblib.dump(tfidf, extractor_path)

metadata = {
    'model_type': 'LogisticRegression',
    'feature_extraction': 'TF-IDF',
    'f1_score': metrics['f1_score'],
    'accuracy': metrics['accuracy'],
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'feature_extractor_type': 'tfidf'
}
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f'Model saved to: {model_path}')
print(f'TF-IDF extractor saved to: {extractor_path}')
print(f'Metadata saved to: {metadata_path}')
print('Done.')
