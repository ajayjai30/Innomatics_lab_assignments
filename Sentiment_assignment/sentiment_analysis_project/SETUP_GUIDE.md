# Project Setup and Quick Start Guide

## Complete Setup Instructions

This guide will help you set up and run the Sentiment Analysis project on your local machine.

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- Git
- ~5 GB of disk space for models and dependencies

### Step 1: Clone/Extract the Project

```bash
# Navigate to the project directory
cd c:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt

# Download NLTK data (required for text preprocessing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 4: Prepare Data

The preprocessed data and models should be located in:
- `data/preprocessed_reviews.csv` - Cleaned and preprocessed reviews
- `models/` - Trained models and feature extractors

If missing, run the notebooks in order:
1. `notebooks/01_eda.ipynb` - Exploratory Data Analysis
2. `notebooks/02_preprocessing.ipynb` - Data Preprocessing
3. `notebooks/03_feature_extraction.ipynb` - Feature Extraction
4. `notebooks/04_model_training.ipynb` - Model Training

## Running the Applications

### Option A: Streamlit Web App (Interactive UI)

```bash
# Activate virtual environment
cd c:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project
venv\Scripts\activate

# Run Streamlit app
streamlit run app/streamlit_app.py
```

The app will open at: `http://localhost:8501`

**Features:**
- Single review sentiment analysis
- Batch CSV upload for multiple reviews
- Real-time confidence scores
- Beautiful interactive interface

### Option B: Flask REST API

```bash
# Activate virtual environment
cd c:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project
venv\Scripts\activate

# Run Flask app
python app/flask_app.py
```

The API will be available at: `http://localhost:5000`

**Endpoints:**
- `GET /` - API homepage and documentation
- `GET /api/health` - Health check
- `POST /api/predict` - Single review prediction
- `POST /api/batch_predict` - Batch predictions

### Option C: Jupyter Notebooks (Development/Testing)

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook

# Open notebooks from the browser and run them sequentially
```

## Project Structure in Detail

```
sentiment_analysis_project/
│
├── data/
│   └── preprocessed_reviews.csv      # Cleaned and processed reviews
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Text Preprocessing
│   ├── 03_feature_extraction.ipynb    # Feature Extraction
│   └── 04_model_training.ipynb        # Model Training & Evaluation
│
├── src/
│   ├── preprocessing.py               # Text preprocessing functions
│   ├── feature_extraction.py          # Feature extraction methods
│   └── model_training.py              # Model training utilities
│
├── models/
│   ├── best_sentiment_model.pkl       # Trained sentiment classifier
│   ├── model_metadata.json            # Model performance metrics
│   ├── bow_extractor.pkl              # Bag-of-Words vectorizer
│   ├── tfidf_extractor.pkl            # TF-IDF vectorizer
│   └── w2v_extractor.pkl              # Word2Vec model
│
├── app/
│   ├── streamlit_app.py               # Streamlit web interface
│   └── flask_app.py                   # Flask REST API
│
├── deployment/
│   ├── deploy_to_ec2.sh               # AWS EC2 deployment script
│   ├── EC2_DEPLOYMENT_GUIDE.md        # EC2 deployment instructions
│   └── DOCKER_DEPLOYMENT.md           # Docker deployment guide
│
├── requirements.txt                   # Python dependencies
├── README.md                          # Project overview
└── SETUP_GUIDE.md                     # This file
```

## Testing the Applications

### Test Streamlit App

1. Open browser: `http://localhost:8501`
2. Try the sample reviews
3. Upload a CSV with multiple reviews
4. Check the confidence scores and sentiment predictions

### Test Flask API

```bash
# Test health endpoint
curl http://localhost:5000/api/health

# Test single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This product is amazing! Highly recommended."}'

# Test batch prediction
curl -X POST http://localhost:5000/api/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great product!", "Poor quality", "Decent item"]}'
```

## Common Issues and Solutions

### Issue 1: NLTK Data Not Downloaded
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Issue 2: Port Already in Use
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8501
kill -9 <PID>
```

### Issue 3: Out of Memory
- Use smaller batch sizes in feature extraction
- Process data in chunks
- Close other applications

### Issue 4: Missing Model Files
- Ensure you've run all notebooks before deploying
- Check that models are saved properly
- Verify file paths in app files

## Performance Benchmarks

- **Model Type**: Logistic Regression / Random Forest
- **Features**: TF-IDF or Word2Vec
- **F1-Score**: ~0.88-0.92
- **Inference Time**: ~50-100ms per review
- **Batch Processing**: ~1000 reviews/min

## Next Steps

1. **Fine-tune**: Try different models and hyperparameters
2. **Enhance**: Add more feature extraction methods (BERT, GPT)
3. **Deploy**: Use AWS EC2 or Docker for production
4. **Monitor**: Set up logging and monitoring
5. **Scale**: Use load balancing for multiple instances

## Further Documentation

- See [EC2_DEPLOYMENT_GUIDE.md](deployment/EC2_DEPLOYMENT_GUIDE.md) for AWS deployment
- See [DOCKER_DEPLOYMENT.md](deployment/DOCKER_DEPLOYMENT.md) for Docker deployment
- See [README.md](README.md) for project overview
- See individual notebooks for detailed analysis

## Support and Contact

For issues or questions:
1. Check the troubleshooting section
2. Review the Jupyter notebooks
3. Check the application logs

---

**Last Updated**: February 2024
**Python Version**: 3.9+
**Status**: Production Ready
