import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import twitter_samples
import sys
import matplotlib.pyplot as plt

# Import local modules
from src.utils import download_nltk_data
from src.model import SentimentModel
from src.visualize import plot_sentiment_distribution, plot_confusion_matrix, plot_word_cloud

def load_data():
    """Load and label NLTK twitter samples."""
    print("Loading data...")
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    
    # Create labels: Positive = 1, Negative = 0
    # NOTE: In a real 'Neutral' case, we would need a dataset with neutral labels.
    # For this project using NLTK samples, we stick to Binary Classification.
    
    y_pos = [1] * len(pos_tweets)
    y_neg = [0] * len(neg_tweets)
    
    X = pos_tweets + neg_tweets
    y = y_pos + y_neg
    
    return X, np.array(y)

def main():
    # 1. Setup
    download_nltk_data()
    
    # 2. Load Data
    X, y = load_data()
    print(f"Loaded {len(X)} tweets.")
    
    # 3. Visualize Data Distribution (Initial)
    # Map labels back to strings for better pie chart labels
    y_labels = pd.Series(y).map({1: 'Positive', 0: 'Negative'})
    print("Close the plot window to continue...")
    plot_sentiment_distribution(y_labels, title='Dataset Sentiment Distribution')
    
    # 4. Word Clouds
    # Show word cloud for a subset to save time/memory or all
    print("Generating Word Clouds... Close windows to continue...")
    pos_subset = [x for x, label in zip(X, y) if label == 1]
    neg_subset = [x for x, label in zip(X, y) if label == 0]
    
    plot_word_cloud(pos_subset, title='Positive Tweets Word Cloud')
    plot_word_cloud(neg_subset, title='Negative Tweets Word Cloud')

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Model Training
    model = SentimentModel()
    model.train(X_train, y_train)
    
    # 7. Evaluation
    acc, report, conf_matrix, y_pred = model.evaluate(X_test, y_test)
    
    # 8. Visualizing Results
    print("Showing Confusion Matrix... Close window to continue...")
    plot_confusion_matrix(conf_matrix, classes=['Negative', 'Positive'])
    
    # 9. Interactive Mode
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("Type a sentence to check its sentiment. Type 'exit' to quit.")
    print("="*50)
    
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        prediction = model.predict([user_input])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"Predicted Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
