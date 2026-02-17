"""
Flask Web Application for Sentiment Analysis API
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sys
sys.path.append(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\src')

from preprocessing import TextPreprocessor
import joblib
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model components
def load_model():
    """Load trained model and components"""
    model = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\best_sentiment_model.pkl')
    
    with open(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    if metadata['feature_extractor_type'] == 'tfidf':
        feature_extractor = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\tfidf_extractor.pkl')
    else:
        feature_extractor = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\w2v_extractor.pkl')
    
    preprocessor = TextPreprocessor(use_lemmatization=True)
    
    return model, metadata, feature_extractor, preprocessor

# Load components at startup
model, metadata, feature_extractor, preprocessor = load_model()

# HTML Template for homepage
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis API</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            text-align: center;
        }
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 30px;
        }
        .metric {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .metric label {
            color: #666;
            font-size: 12px;
            display: block;
        }
        .metric value {
            color: #333;
            font-size: 18px;
            font-weight: bold;
            display: block;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            resize: vertical;
            margin-bottom: 15px;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:active {
            transform: translateY(0);
        }
        .api-docs {
            margin-top: 30px;
            padding-top: 30px;
            border-top: 2px solid #e0e0e0;
        }
        .api-docs h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .code-block {
            background: #282c34;
            color: #abb2bf;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 12px;
            font-family: 'Courier New', monospace;
            margin: 10px 0;
            line-height: 1.5;
        }
        .endpoint {
            margin-top: 15px;
        }
        .method {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>😊 Sentiment Analysis API</h1>
        <p class="subtitle">Flipkart Product Review Sentiment Classifier</p>
        
        <div class="info-box">
            <strong>Model Status:</strong> ✅ Active and Running
        </div>
        
        <div class="metrics">
            <div class="metric">
                <label>Model Type</label>
                <value>{{ model_type }}</value>
            </div>
            <div class="metric">
                <label>F1-Score</label>
                <value>{{ f1_score }}</value>
            </div>
            <div class="metric">
                <label>Accuracy</label>
                <value>{{ accuracy }}</value>
            </div>
            <div class="metric">
                <label>Features</label>
                <value>{{ features }}</value>
            </div>
        </div>
        
        <div class="api-docs">
            <h3>📚 API Documentation</h3>
            
            <div class="endpoint">
                <span class="method">POST</span>
                <strong>/api/predict</strong>
                <p>Predict sentiment for a single review</p>
            </div>
            
            <p><strong>Request Body:</strong></p>
            <div class="code-block">
{
  "review": "This product is amazing!"
}
            </div>
            
            <p><strong>Response:</strong></p>
            <div class="code-block">
{
  "sentiment": "positive",
  "confidence": 0.95,
  "timestamp": "2024-01-15T10:30:00"
}
            </div>
            
            <div class="endpoint" style="margin-top: 20px;">
                <span class="method">POST</span>
                <strong>/api/batch_predict</strong>
                <p>Predict sentiment for multiple reviews</p>
            </div>
            
            <p><strong>Request Body:</strong></p>
            <div class="code-block">
{
  "reviews": [
    "Great product!",
    "Poor quality"
  ]
}
            </div>
            
            <p><strong>Response:</strong></p>
            <div class="code-block">
{
  "predictions": [
    {"review": "Great product!", "sentiment": "positive", "confidence": 0.92},
    {"review": "Poor quality", "sentiment": "negative", "confidence": 0.88}
  ],
  "timestamp": "2024-01-15T10:30:00"
}
            </div>
            
            <div class="endpoint" style="margin-top: 20px;">
                <span class="method">GET</span>
                <strong>/api/health</strong>
                <p>Check API health status</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with API documentation"""
    return render_template_string(
        HTML_TEMPLATE,
        model_type=metadata['model_type'],
        f1_score=f"{metadata['f1_score']:.4f}",
        accuracy=f"{metadata['accuracy']:.4f}",
        features=metadata['feature_extraction']
    )

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': metadata['model_type'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict sentiment for a single review"""
    try:
        data = request.get_json()
        
        if not data or 'review' not in data:
            return jsonify({'error': 'Review text is required'}), 400
        
        review = data['review']
        
        if not isinstance(review, str) or len(review.strip()) == 0:
            return jsonify({'error': 'Review must be a non-empty string'}), 400
        
        # Preprocess
        cleaned = preprocessor.clean_text(
            review,
            remove_stopwords=True,
            normalize=True
        )
        
        # Extract features
        if metadata['feature_extractor_type'] == 'tfidf':
            features = feature_extractor.transform([cleaned])
        else:
            features = feature_extractor.transform([cleaned.split()])
        
        # Predict
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0]
        
        return jsonify({
            'review': review,
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(max(confidence)),
            'positive_prob': float(confidence[1]),
            'negative_prob': float(confidence[0]),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple reviews"""
    try:
        data = request.get_json()
        
        if not data or 'reviews' not in data:
            return jsonify({'error': 'Reviews list is required'}), 400
        
        reviews = data['reviews']
        
        if not isinstance(reviews, list):
            return jsonify({'error': 'Reviews must be a list'}), 400
        
        predictions = []
        
        for review in reviews:
            if not isinstance(review, str):
                continue
            
            # Preprocess
            cleaned = preprocessor.clean_text(
                review,
                remove_stopwords=True,
                normalize=True
            )
            
            # Extract features
            if metadata['feature_extractor_type'] == 'tfidf':
                features = feature_extractor.transform([cleaned])
            else:
                features = feature_extractor.transform([cleaned.split()])
            
            # Predict
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0]
            
            predictions.append({
                'review': review,
                'sentiment': 'positive' if prediction == 1 else 'negative',
                'confidence': float(max(confidence))
            })
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
