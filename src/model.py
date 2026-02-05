from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import numpy as np

try:
    from src.utils import process_tweet
except ImportError:
    from utils import process_tweet

class SentimentModel:
    def __init__(self, model_type='logistic'):
        """
        Initialize the model structure.
        model_type: 'logistic', 'naive_bayes', or 'bert'.
        """
        self.model_type = model_type
        self.pipeline = None

        if model_type == 'bert':
            from transformers import pipeline
            # Load pre-trained model from Hugging Face
            print("Loading BERT pipeline... this may take a moment.")
            self.pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        elif model_type == 'naive_bayes':
            clf = MultinomialNB()
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=process_tweet, token_pattern=None)),
                ('clf', clf)
            ])
            
        else: # logistic
            clf = LogisticRegression(random_state=42, max_iter=1000)
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=process_tweet, token_pattern=None)),
                ('clf', clf)
            ])

    def train(self, X_train, y_train):
        """Train the model (if applicable)."""
        if self.model_type == 'bert':
            print("BERT model is pre-trained. Skipping training.")
            return

        print(f"Training {self.model_type} model...")
        self.pipeline.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X):
        """
        Predict sentiment.
        Returns: array of 0s (Negative) and 1s (Positive).
        """
        if self.model_type == 'bert':
            # BERT pipeline accepts list of strings
            # Ensure X is a list
            if hasattr(X, 'tolist'):
                X = X.tolist()
            
            results = self.pipeline(X)
            # Map 'POSITIVE' -> 1, 'NEGATIVE' -> 0
            predictions = [1 if res['label'] == 'POSITIVE' else 0 for res in results]
            return np.array(predictions)
        
        else:
            return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        Returns: array of shape (n_samples, 2).
        """
        if self.model_type == 'bert':
            if hasattr(X, 'tolist'):
                X = X.tolist()
                
            results = self.pipeline(X)
            probs = []
            for res in results:
                score = res['score']
                if res['label'] == 'POSITIVE':
                    # [Probability of Neg, Probability of Pos]
                    probs.append([1-score, score])
                else:
                    # [Probability of Neg, Probability of Pos]
                    probs.append([score, 1-score])
            return np.array(probs)
            
        else:
            return self.pipeline.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return metrics."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return acc, report, conf_matrix, y_pred

    def save_model(self, path):
        """Save the model to disk."""
        if self.model_type == 'bert':
            # For BERT, we don't save the whole pipeline object with joblib efficiently.
            # We rely on Hugging Face cache.
            # But we can create a placeholder file to indicate it exists.
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write("bert-placeholder")
            print(f"BERT placeholder saved to {path}")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load the model from disk."""
        if not os.path.exists(path):
            print(f"Model file not found at {path}")
            return False

        if self.model_type == 'bert':
            # Pipeline is already loaded in __init__
            print("BERT pipeline ready (cached).")
            return True

        self.pipeline = joblib.load(path)
        print(f"Model loaded from {path}")
        return True
