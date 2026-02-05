import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def download_nltk_data():
    """
    Check if we have the necessary NLTK datasets.
    If not, download them. This prevents errors when the app first runs on a new machine.
    """
    try:
        nltk.data.find('corpora/twitter_samples')
    except LookupError:
        print("Downloading twitter_samples...")
        nltk.download('twitter_samples')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords')

def process_tweet(tweet):
    """
    Clean and prepare a tweet for the model.
    1. Remove junk (tickers, RT, links, hashtags).
    2. Tokenize (split into words).
    3. Remove stopwords (common words like 'the', 'is') and punctuation.
    4. Stem words (convert 'running' -> 'run') to reduce vocabulary size.
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    # Remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    
    # Remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    
    # Remove hyperlinks (http or https)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # Remove hashtags (just the # sign, keeping the word)
    tweet = re.sub(r'#', '', tweet)
    
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # Drop common words that don't add meaning
                word not in string.punctuation):  # Drop punctuation marks
            stem_word = stemmer.stem(word)  # Reduce word to root form
            tweets_clean.append(stem_word)

    return tweets_clean
