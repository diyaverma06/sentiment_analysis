import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

def plot_sentiment_distribution(y, title='Sentiment Distribution'):
    """Plot the distribution of sentiments."""
    fig = plt.figure(figsize=(6, 6))
    if isinstance(y, pd.Series):
        counts = y.value_counts()
    else:
        counts = pd.Series(y).value_counts()
    
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title(title)
    return fig

def plot_confusion_matrix(conf_matrix, classes, title='Confusion Matrix'):
    """Plot confusion matrix heatmap."""
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    return fig

def plot_word_cloud(text_data, title='Word Cloud'):
    """Generate and plot a word cloud."""
    # Assuming text_data is a list of strings (tweets)
    # Join them into one big string
    if isinstance(text_data, list) or isinstance(text_data, pd.Series):
        text = ' '.join(str(t) for t in text_data)
    else:
        text = text_data

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    return fig
