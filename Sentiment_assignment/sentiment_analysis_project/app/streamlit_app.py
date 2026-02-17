"""
Streamlit Web Application for Sentiment Analysis
Flipkart Product Review Sentiment Classifier
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\src')

from preprocessing import TextPreprocessor
import joblib
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .sentiment-positive {
        color: #27AE60;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #E74C3C;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("😊 Sentiment Analysis System")
st.markdown("### Flipkart Product Review Sentiment Classifier")
st.markdown("---")

# Load model and preprocessing components
@st.cache_resource
def load_model_components():
    """Load trained model and feature extractors"""
    try:
        # Load model
        model = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\best_sentiment_model.pkl')
        
        # Load metadata
        with open(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load feature extractor
        if metadata['feature_extractor_type'] == 'tfidf':
            feature_extractor = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\tfidf_extractor.pkl')
        else:
            feature_extractor = joblib.load(r'C:\Users\admin\Documents\Innomatics\Sentiment\sentiment_analysis_project\models\w2v_extractor.pkl')
        
        # Load preprocessor
        preprocessor = TextPreprocessor(use_lemmatization=True)
        
        return model, metadata, feature_extractor, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# Load components
model, metadata, feature_extractor, preprocessor = load_model_components()

if model is None:
    st.error("❌ Failed to load model components. Please ensure all model files are in the correct location.")
    st.stop()

# Sidebar information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown(f"**Model Type:** {metadata['model_type']}")
    st.markdown(f"**Feature Extraction:** {metadata['feature_extraction']}")
    st.markdown(f"**F1-Score:** {metadata['f1_score']:.4f}")
    st.markdown(f"**Accuracy:** {metadata['accuracy']:.4f}")
    st.markdown(f"**Precision:** {metadata['precision']:.4f}")
    st.markdown(f"**Recall:** {metadata['recall']:.4f}")
    
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. Enter a product review in the text box
    2. Click 'Analyze Sentiment'
    3. View the prediction and confidence score
    """)
    
    st.markdown("---")
    st.markdown("### 📝 About")
    st.markdown("""
    This app classifies Flipkart product reviews as **Positive** or **Negative**.
    
    - **Positive:** Rating >= 3 stars
    - **Negative:** Rating < 3 stars
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Review Text")
    user_input = st.text_area(
        "Paste your review here:",
        placeholder="Example: This product is amazing! Great quality and fast delivery.",
        height=200,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 🎯 Sample Reviews")
    sample_positive = "Great product! Excellent quality and very fast delivery. Highly recommended!"
    sample_negative = "Poor quality. Product broke after 2 days. Waste of money. Disappointed."
    
    if st.button("📌 Positive Sample"):
        st.session_state.user_input = sample_positive
    
    if st.button("📌 Negative Sample"):
        st.session_state.user_input = sample_negative

# Initialize session state
if 'user_input' in st.session_state:
    user_input = st.session_state.user_input

# Analyze button
col1, col2, col3 = st.columns([2, 1, 1])

with col2:
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

if analyze_button:
    if not user_input.strip():
        st.error("❌ Please enter a review text to analyze.")
    else:
        # Clean and preprocess the input
        cleaned_text = preprocessor.clean_text(
            user_input,
            remove_stopwords=True,
            normalize=True
        )
        
        # Generate features based on the feature extractor type
        if metadata['feature_extractor_type'] == 'tfidf':
            features = feature_extractor.transform([cleaned_text])
        else:  # Word2Vec
            tokenized_text = cleaned_text.split()
            features = feature_extractor.transform([tokenized_text])
        
        # Make prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_label = "😊 Positive" if prediction == 1 else "😞 Negative"
            st.metric(label="Sentiment", value=sentiment_label)
        
        with col2:
            confidence_score = max(confidence) * 100
            st.metric(label="Confidence", value=f"{confidence_score:.2f}%")
        
        with col3:
            if prediction == 1:
                st.success("Positive Review")
            else:
                st.warning("Negative Review")
        
        # Detailed breakdown
        st.markdown("### 📈 Probability Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Positive Probability:** {confidence[1]*100:.2f}%")
            st.progress(confidence[1])
        
        with col2:
            st.markdown(f"**Negative Probability:** {confidence[0]*100:.2f}%")
            st.progress(confidence[0])
        
        # Display processed text
        with st.expander("🔧 View Processed Text"):
            st.markdown("**Original Text:**")
            st.write(user_input)
            st.markdown("**Cleaned & Normalized Text:**")
            st.write(cleaned_text)

# Batch analysis section
st.markdown("---")
st.markdown("### 📂 Batch Analysis")

with st.expander("Upload CSV for Batch Analysis"):
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            st.markdown(f"**Loaded {len(batch_df)} reviews**")
            
            if 'Review text' not in batch_df.columns:
                st.error("CSV must contain a 'Review text' column")
            else:
                if st.button("🔄 Analyze All Reviews"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, review in enumerate(batch_df['Review text'].values):
                        # Preprocess
                        cleaned = preprocessor.clean_text(
                            str(review),
                            remove_stopwords=True,
                            normalize=True
                        )
                        
                        # Extract features
                        if metadata['feature_extractor_type'] == 'tfidf':
                            feat = feature_extractor.transform([cleaned])
                        else:
                            feat = feature_extractor.transform([cleaned.split()])
                        
                        # Predict
                        pred = model.predict(feat)[0]
                        conf = model.predict_proba(feat)[0]
                        
                        results.append({
                            'Review': review[:100],
                            'Sentiment': 'Positive' if pred == 1 else 'Negative',
                            'Confidence': max(conf)
                        })
                        
                        progress_bar.progress((idx + 1) / len(batch_df))
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
    <p>Sentiment Analysis System © 2024 | Innomatics Sentiment Analysis Project</p>
    </div>
""", unsafe_allow_html=True)
