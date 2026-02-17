"""
Text preprocessing functions for sentiment analysis
"""

import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """
    A class to preprocess text data for sentiment analysis
    """
    
    def __init__(self, use_lemmatization=True):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.use_lemmatization = use_lemmatization
        self.stop_words = set(stopwords.words('english'))
    
    def remove_special_characters(self, text):
        """Remove special characters and punctuation"""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def remove_html_tags(self, text):
        """Remove HTML tags"""
        text = re.sub(r'<.*?>', '', text)
        return text
    
    def remove_urls(self, text):
        """Remove URLs"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        return text
    
    def to_lowercase(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_whitespace(self, text):
        """Remove extra whitespace"""
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords"""
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def lemmatize(self, text):
        """Apply lemmatization"""
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized_tokens)
    
    def stem(self, text):
        """Apply stemming"""
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_tokens)
    
    def clean_text(self, text, remove_stopwords=True, normalize=True):
        """
        Complete text cleaning pipeline
        
        Parameters:
        -----------
        text : str
            Text to clean
        remove_stopwords : bool
            Whether to remove stopwords
        normalize : bool
            Whether to apply lemmatization/stemming
        
        Returns:
        --------
        str
            Cleaned text
        """
        # Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Convert to lowercase
        text = self.to_lowercase(text)
        
        # Remove special characters
        text = self.remove_special_characters(text)
        
        # Remove extra whitespace
        text = self.remove_whitespace(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Normalization
        if normalize:
            if self.use_lemmatization:
                text = self.lemmatize(text)
            else:
                text = self.stem(text)
        
        return text
    
    def process_dataframe(self, df, text_column, output_column='cleaned_text', 
                         remove_stopwords=True, normalize=True):
        """
        Process entire DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        text_column : str
            Column name containing text to clean
        output_column : str
            Column name for cleaned text
        remove_stopwords : bool
            Whether to remove stopwords
        normalize : bool
            Whether to apply normalization
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with cleaned text column
        """
        df[output_column] = df[text_column].apply(
            lambda x: self.clean_text(x, remove_stopwords, normalize) if pd.notna(x) else ''
        )
        return df


def create_binary_labels(df, rating_column='Ratings', threshold=3):
    """
    Create binary sentiment labels from ratings
    Ratings >= threshold: Positive (1)
    Ratings < threshold: Negative (0)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    rating_column : str
        Column containing ratings
    threshold : int
        Threshold for positive sentiment (default: 3)
    
    Returns:
    --------
    pd.Series
        Binary sentiment labels
    """
    return (df[rating_column] >= threshold).astype(int)


def handle_missing_values(df, text_column='Review text', strategy='drop'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    text_column : str
        Column to check for missing values
    strategy : str
        'drop' to remove rows, 'fill' to fill with empty string
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        df = df.dropna(subset=[text_column])
    elif strategy == 'fill':
        df[text_column] = df[text_column].fillna('')
    
    return df
