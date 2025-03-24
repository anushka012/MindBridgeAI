# sentiment_analysis.py
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK's VADER sentiment analysis tool
nltk.download("vader_lexicon")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Returns sentiment category: positive, negative, or neutral."""
    sentiment_score = sia.polarity_scores(text)["compound"]
    if sentiment_score >= 0.05:
        return "positive"
    elif sentiment_score <= -0.05:
        return "negative"
    else:
        return "neutral"