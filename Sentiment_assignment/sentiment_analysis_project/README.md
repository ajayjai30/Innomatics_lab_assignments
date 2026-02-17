# Sentiment Analysis of Flipkart Product Reviews

## Project Overview
This project classifies customer reviews as positive or negative and analyzes pain points of customers who write negative reviews. We use machine learning and deep learning models to understand what features contribute to customer satisfaction or dissatisfaction.

## Dataset
- **Product**: YONEX MAVIS 350 Nylon Shuttle
- **Reviews**: 8,518 reviews from Flipkart
- **Features**: Reviewer Name, Rating, Review Title, Review Text, Place of Review, Date of Review, Up Votes, Down Votes

## Project Structure
```
sentiment_analysis_project/
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Python modules for preprocessing, feature extraction, training
├── models/                 # Trained models
├── app/                    # Flask/Streamlit web application
├── deployment/             # AWS EC2 deployment scripts
└── requirements.txt        # Python dependencies
```

## Workflow

### 1. Data Loading and Analysis
- Load reviews from CSV
- Explore data distribution, ratings, and review characteristics
- Identify positive/negative patterns

### 2. Data Cleaning
- Remove special characters and punctuation
- Remove stopwords
- Handle missing values
- Normalize text (lowercase, whitespace)

### 3. Text Embedding
- Bag-of-Words (BoW)
- TF-IDF
- Word2Vec (W2V)
- BERT embeddings

### 4. Model Training
- Logistic Regression
- Random Forest
- XGBoost
- LSTM/GRU (Deep Learning)
- BERT-based classifiers

### 5. Model Evaluation
- F1-Score as primary metric
- Precision, Recall, Accuracy
- Confusion Matrix
- ROC-AUC

### 6. Deployment
- Flask/Streamlit web app
- Real-time sentiment prediction
- AWS EC2 deployment

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the notebooks in order:
   - `01_eda.ipynb` - Exploratory Data Analysis
   - `02_preprocessing.ipynb` - Data Cleaning
   - `03_feature_extraction.ipynb` - Text embeddings
   - `04_model_training.ipynb` - Train and evaluate models

3. Launch the web app:
```bash
streamlit run app/streamlit_app.py
```

## Results
Results, model performance metrics, and insights will be documented as the project progresses.

## Author
Innomatics Sentiment Analysis Project
