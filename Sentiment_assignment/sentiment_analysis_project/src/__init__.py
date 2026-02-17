"""
Initialize the src module for sentiment analysis
"""

__version__ = "1.0.0"
__author__ = "Innomatics Sentiment Analysis Project"

from .preprocessing import TextPreprocessor, create_binary_labels, handle_missing_values
from .feature_extraction import (
    BagOfWordsExtractor,
    TFIDFExtractor,
    Word2VecExtractor,
    BERTExtractor
)
from .model_training import ModelTrainer, train_and_evaluate_all_models

__all__ = [
    'TextPreprocessor',
    'create_binary_labels',
    'handle_missing_values',
    'BagOfWordsExtractor',
    'TFIDFExtractor',
    'Word2VecExtractor',
    'BERTExtractor',
    'ModelTrainer',
    'train_and_evaluate_all_models'
]
