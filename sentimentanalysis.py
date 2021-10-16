# -*- coding: utf-8 -*-
pip install vaderSentiment

pip install transformers

import nltk
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

df = pd.read_csv(r"my_data.csv") #read in your cleaned data

"""VADER ( Valence Aware Dictionary for Sentiment Reasoning)"""

def sentiment_scores_vader(texts):
    scores = []
    analyzer = SentimentIntensityAnalyzer()
    for text in texts:
        sentiment_dict = analyzer.polarity_scores(text)
        scores.append(sentiment_dict['compound'])
    return scores

"""Transformer"""

def sentiment_scores_transformer(texts):
    scores = []
    classifier = pipeline('sentiment-analysis')
    for text in texts:
        sentiment_dict = classifier(text)
        score = (sentiment_dict[0])['score']
        if (sentiment_dict[0])['label'] == "NEGATIVE":
          score *= -1
        scores.append(score)
    return scores

print(f"VADER:{sentiment_scores_vader(['i hate that horrible rude cat!'])}")
print(f"Transformer:{sentiment_scores_transformer(['i hate that horrible rude cat!'])}")

print(f"VADER:{sentiment_scores_vader(['i love my beautiful friend!'])}")
print(f"Transformer:{sentiment_scores_transformer(['i love my beautiful friend!'])}")

"""Textblob

see: https://investigate.ai/investigating-sentiment-analysis/comparing-sentiment-analysis-tools/
"""

def textblob(df):
  df['sentiment_textblob'] = df['text'].apply(lambda x: TextBlob(x).sentiment[0])

def textblob_bayes(df):
  blobber = Blobber(analyzer=NaiveBayesAnalyzer())
  df['sentiment_bayes'] = df['text'].apply(lambda x: blobber(x).sentiment[0])

textblob(df)

textblob_bayes(df)

"""**Visualizing & Comparing Your Results**"""

#make a histogram of sentiment scores
df.hist('sentiment_textblob')
print(df['sentiment_textblob'].mean())

#plot average monthly sentiment
df['date'] = pd.to_datetime(df['date'])
df['yearmon'] = pd.DatetimeIndex(df['date']).to_period('M')
mean_sentiment_byday = pd.DataFrame(df.groupby(['yearmon'])['sentiment_textblob'].mean())
mean_sentiment_byday.reset_index(level=0, inplace=True)
mean_sentiment_byday.plot(x='yearmon',y='sentiment_textblob')
