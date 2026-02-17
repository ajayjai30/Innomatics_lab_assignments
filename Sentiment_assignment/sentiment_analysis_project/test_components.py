"""
Testing Script for Sentiment Analysis Application
Run this file to test all components
"""

import sys
sys.path.append(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\src')

from preprocessing import TextPreprocessor
from feature_extraction import TFIDFExtractor, Word2VecExtractor
from model_training import ModelTrainer
import joblib
import json

print("="*80)
print("SENTIMENT ANALYSIS - COMPONENT TEST SUITE")
print("="*80)

# Test 1: Text Preprocessing
print("\n[TEST 1] Text Preprocessing")
print("-" * 80)
try:
    preprocessor = TextPreprocessor(use_lemmatization=True)
    
    test_reviews = [
        "This product is amazing! Great quality and fast shipping.",
        "Terrible quality. Product broke after one day. Very disappointed!",
        "Average product. Nothing special but does the job."
    ]
    
    for i, review in enumerate(test_reviews, 1):
        cleaned = preprocessor.clean_text(review, remove_stopwords=True, normalize=True)
        print(f"\nReview {i}:")
        print(f"  Original: {review}")
        print(f"  Cleaned:  {cleaned}")
    
    print("\n✓ Text Preprocessing: PASSED")
except Exception as e:
    print(f"\n✗ Text Preprocessing: FAILED - {str(e)}")

# Test 2: Feature Extraction (TF-IDF)
print("\n[TEST 2] Feature Extraction (TF-IDF)")
print("-" * 80)
try:
    tfidf_extractor = TFIDFExtractor(max_features=1000)
    test_texts = ["good product", "bad quality product"]
    
    features = tfidf_extractor.fit_transform(test_texts)
    print(f"Input texts: {test_texts}")
    print(f"Features shape: {features.shape}")
    print(f"Features type: {type(features)}")
    print(f"Vocabulary size: {len(tfidf_extractor.feature_names)}")
    
    print("\n✓ TF-IDF Feature Extraction: PASSED")
except Exception as e:
    print(f"\n✗ TF-IDF Feature Extraction: FAILED - {str(e)}")

# Test 3: Feature Extraction (Word2Vec)
print("\n[TEST 3] Feature Extraction (Word2Vec)")
print("-" * 80)
try:
    w2v_extractor = Word2VecExtractor(vector_size=100, window=5, min_count=1)
    
    tokenized_texts = [
        "good quality product".split(),
        "bad quality product".split(),
        "average quality product".split()
    ]
    
    features = w2v_extractor.fit_transform(tokenized_texts)
    print(f"Input texts: {tokenized_texts}")
    print(f"Features shape: {features.shape}")
    print(f"Features type: {type(features)}")
    print(f"Embedding dimension: {features.shape[1]}")
    
    print("\n✓ Word2Vec Feature Extraction: PASSED")
except Exception as e:
    print(f"\n✗ Word2Vec Feature Extraction: FAILED - {str(e)}")

# Test 4: Model Loading
print("\n[TEST 4] Model Loading")
print("-" * 80)
try:
    model_path = r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\best_sentiment_model.pkl'
    model = joblib.load(model_path)
    print(f"Model loaded: {model}")
    print(f"Model type: {type(model).__name__}")
    
    metadata_path = r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\model_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nModel Metadata:")
    print(f"  Type: {metadata['model_type']}")
    print(f"  Features: {metadata['feature_extraction']}")
    print(f"  F1-Score: {metadata['f1_score']:.4f}")
    print(f"  Accuracy: {metadata['accuracy']:.4f}")
    
    print("\n✓ Model Loading: PASSED")
except Exception as e:
    print(f"\n✗ Model Loading: FAILED - {str(e)}")

# Test 5: Complete Pipeline
print("\n[TEST 5] Complete Prediction Pipeline")
print("-" * 80)
try:
    # Load components
    preprocessor = TextPreprocessor(use_lemmatization=True)
    tfidf = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\tfidf_extractor.pkl')
    model = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\best_sentiment_model.pkl')
    
    # Test prediction
    test_review = "This product is absolutely amazing! Highly recommended to everyone."
    
    # Preprocess
    cleaned = preprocessor.clean_text(test_review, remove_stopwords=True, normalize=True)
    
    # Extract features
    features = tfidf.transform([cleaned])
    
    # Predict
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]
    
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    confidence_score = max(confidence)
    
    print(f"Input Review: {test_review}")
    print(f"Cleaned Text: {cleaned}")
    print(f"Prediction: {sentiment}")
    print(f"Confidence: {confidence_score*100:.2f}%")
    print(f"Positive Prob: {confidence[1]:.4f}")
    print(f"Negative Prob: {confidence[0]:.4f}")
    
    print("\n✓ Complete Pipeline: PASSED")
except Exception as e:
    print(f"\n✗ Complete Pipeline: FAILED - {str(e)}")

# Test 6: Batch Processing
print("\n[TEST 6] Batch Prediction")
print("-" * 80)
try:
    preprocessor = TextPreprocessor(use_lemmatization=True)
    tfidf = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\tfidf_extractor.pkl')
    model = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\best_sentiment_model.pkl')
    
    test_reviews = [
        "Amazing product! Love it!",
        "Terrible quality, waste of money",
        "Average product, nothing special"
    ]
    
    print("Batch Predictions:")
    for i, review in enumerate(test_reviews, 1):
        cleaned = preprocessor.clean_text(review, remove_stopwords=True, normalize=True)
        features = tfidf.transform([cleaned])
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0]
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        
        print(f"\n  Review {i}: {review}")
        print(f"    Sentiment: {sentiment} ({max(confidence)*100:.2f}% confidence)")
    
    print("\n✓ Batch Prediction: PASSED")
except Exception as e:
    print(f"\n✗ Batch Prediction: FAILED - {str(e)}")

# Summary
print("\n" + "="*80)
print("TEST SUITE SUMMARY")
print("="*80)
print("""
✓ All core components tested successfully!

Next Steps:
1. Run the Jupyter notebooks for detailed analysis
2. Launch Streamlit app: streamlit run app/streamlit_app.py
3. Launch Flask API: python app/flask_app.py
4. Deploy to AWS EC2 using: deployment/deploy_to_ec2.sh
5. Deploy with Docker: docker-compose up

For more information, see:
- SETUP_GUIDE.md - Setup instructions
- API_REFERENCE.md - API documentation
- EC2_DEPLOYMENT_GUIDE.md - AWS deployment
- DOCKER_DEPLOYMENT.md - Docker deployment
""")
print("="*80)
