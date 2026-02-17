"""
Model training and evaluation functions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
import joblib


class ModelTrainer:
    """Train and evaluate sentiment classification models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def train_logistic_regression(self, X_train, y_train, **kwargs):
        """Train Logistic Regression model"""
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest model"""
        model = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=100,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """Train XGBoost model"""
        model = xgb.XGBClassifier(
            random_state=self.random_state,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : sklearn model
            Trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        self.results[model_name] = results
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results_df = []
        
        for model_name, model in self.models.items():
            results = self.evaluate_model(model_name, model, X_test, y_test)
            results_df.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc']
            })
        
        return pd.DataFrame(results_df)
    
    def save_model(self, model_name, filepath):
        """Save model to disk"""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"Model '{model_name}' saved to {filepath}")
        else:
            raise ValueError(f"Model '{model_name}' not found in trained models")
    
    def load_model(self, filepath):
        """Load model from disk"""
        model = joblib.load(filepath)
        return model
    
    def save_all_models(self, directory):
        """Save all trained models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(directory, f"{model_name}.pkl")
            joblib.dump(model, filepath)
            print(f"Model '{model_name}' saved to {filepath}")


def train_and_evaluate_all_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple models
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    
    Returns:
    --------
    ModelTrainer
        Trainer object with trained models
    """
    trainer = ModelTrainer()
    
    print("Training Logistic Regression...")
    trainer.train_logistic_regression(X_train, y_train)
    
    print("Training Random Forest...")
    trainer.train_random_forest(X_train, y_train, n_jobs=-1)
    
    print("Training XGBoost...")
    trainer.train_xgboost(X_train, y_train, verbose=0)
    
    print("\nEvaluating models...")
    results_df = trainer.evaluate_all_models(X_test, y_test)
    
    return trainer, results_df


def get_model_feature_importance(model, feature_names=None):
    """
    Get feature importance from tree-based models
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    feature_names : array-like, optional
        Feature names
    
    Returns:
    --------
    pd.DataFrame
        Feature importances sorted by value
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        if feature_names is not None:
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
        else:
            df = pd.DataFrame({
                'feature': range(len(importances)),
                'importance': importances
            })
        
        return df.sort_values('importance', ascending=False)
    else:
        raise ValueError("Model does not have feature importance attribute")


def hyperparameter_tuning_results(X_train, y_train, X_test, y_test):
    """
    Example function showing hyperparameter tuning approach
    """
    from sklearn.model_selection import GridSearchCV
    
    # Hyperparameters for Logistic Regression
    lr_params = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    print("Hyperparameter tuning for Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    gs_lr = GridSearchCV(lr, lr_params, cv=5, scoring='f1', n_jobs=-1)
    gs_lr.fit(X_train, y_train)
    
    print(f"Best parameters: {gs_lr.best_params_}")
    print(f"Best CV F1-Score: {gs_lr.best_score_:.4f}")
    
    return gs_lr.best_estimator_
