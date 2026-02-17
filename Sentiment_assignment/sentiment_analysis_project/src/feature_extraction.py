"""
Feature extraction functions for text data
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch


class BagOfWordsExtractor:
    """Extract Bag-of-Words features from text"""
    
    def __init__(self, max_features=5000, lowercase=True):
        self.vectorizer = CountVectorizer(max_features=max_features, lowercase=lowercase)
        self.feature_names = None
    
    def fit(self, texts):
        """Fit BoW vectorizer"""
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self
    
    def transform(self, texts):
        """Transform texts to BoW features"""
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return X


class TFIDFExtractor:
    """Extract TF-IDF features from text"""
    
    def __init__(self, max_features=5000, lowercase=True):
        self.vectorizer = TfidfVectorizer(max_features=max_features, lowercase=lowercase)
        self.feature_names = None
    
    def fit(self, texts):
        """Fit TF-IDF vectorizer"""
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self
    
    def transform(self, texts):
        """Transform texts to TF-IDF features"""
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return X


class Word2VecExtractor:
    """Extract Word2Vec embeddings from text"""
    
    def __init__(self, vector_size=300, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
    
    def fit(self, sentences):
        """
        Train Word2Vec model
        
        Parameters:
        -----------
        sentences : list of list of str
            List of tokenized sentences
        """
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1  # Skip-gram model
        )
        return self
    
    def transform(self, sentences):
        """
        Transform sentences to average Word2Vec embeddings
        
        Parameters:
        -----------
        sentences : list of list of str
            List of tokenized sentences
        
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, vector_size)
        """
        embeddings = []
        
        for sentence in sentences:
            vectors = [self.model.wv[word] for word in sentence if word in self.model.wv]
            
            if vectors:
                embedding = np.mean(vectors, axis=0)
            else:
                embedding = np.zeros(self.vector_size)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def fit_transform(self, sentences):
        """Fit and transform in one step"""
        self.fit(sentences)
        return self.transform(sentences)


class BERTExtractor:
    """Extract BERT embeddings from text"""
    
    def __init__(self, model_name='bert-base-uncased', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
    
    def transform(self, texts, batch_size=32, pooling='mean'):
        """
        Transform texts to BERT embeddings
        
        Parameters:
        -----------
        texts : list of str
            List of texts to embed
        batch_size : int
            Batch size for processing
        pooling : str
            'mean' for mean pooling, 'cls' for CLS token
        
        Returns:
        --------
        np.ndarray
            Array of embeddings with shape (n_samples, embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize and encode
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get BERT outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Apply pooling
                if pooling == 'mean':
                    # Mean pooling
                    last_hidden = outputs.last_hidden_state
                    embeddings_batch = torch.mean(last_hidden * attention_mask.unsqueeze(-1), dim=1)
                else:  # cls token
                    embeddings_batch = outputs.last_hidden_state[:, 0, :]
                
                embeddings.append(embeddings_batch.cpu().numpy())
        
        return np.vstack(embeddings)


def extract_text_features(texts, method='tfidf', **kwargs):
    """
    Extract text features using specified method
    
    Parameters:
    -----------
    texts : list or array-like
        List of texts to extract features from
    method : str
        Feature extraction method: 'bow', 'tfidf', 'w2v', 'bert'
    **kwargs : dict
        Additional arguments for the specific extractor
    
    Returns:
    --------
    np.ndarray or sparse matrix
        Extracted features
    """
    
    if method == 'bow':
        extractor = BagOfWordsExtractor(**kwargs)
        return extractor.fit_transform(texts)
    
    elif method == 'tfidf':
        extractor = TFIDFExtractor(**kwargs)
        return extractor.fit_transform(texts)
    
    elif method == 'w2v':
        # For Word2Vec, texts should be tokenized
        extractor = Word2VecExtractor(**kwargs)
        return extractor.fit_transform(texts)
    
    elif method == 'bert':
        extractor = BERTExtractor(**kwargs)
        return extractor.transform(texts)
    
    else:
        raise ValueError(f"Unknown method: {method}")
