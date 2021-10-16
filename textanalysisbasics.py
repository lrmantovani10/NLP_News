import pandas as pd
import numpy as np
import re

df = pd.read_csv("my_data.csv")

#remove urls
def clean(text):
  output = re.sub('<[^>]+>', '', text)
  output = output.replace("\n"," ")
  return output

df['text'] = df['text'].map(clean)
df.to_csv('cleaned_data.csv', index=False)



"""**Very basic preliminary analysis**"""

#number of words per article
df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))
print(df[['text','word_count']].head())

df['word_count'].plot.bar()

#average sentence length
def avg_word(sentence):
  words = sentence.split()
  try:
    r = (sum(len(word) for word in words)/len(words))
  except:
    r = 0
  return r

df['avg_word'] = df['text'].apply(lambda x: avg_word(x))
df[['text','avg_word']].head()

df['avg_word'].plot.bar()

"""What is a stopword?"""

import nltk
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
stop = stopwords.words('english')
#count stopwords per article
df['stopwords'] = df['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
print(df[['text','stopwords']].head())

df['stopwords'].plot.bar()

#count numbers per article
df['numerics'] = df['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print(df[['text','numerics']].head())

print(df['numerics'].plot.bar())

"""**Basic Pre-processing**

Everything lowercase
"""

df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

"""Remove punctuation"""

df['text'] = df['text'].str.replace('[^\w\s]','')

"""Remove stopwords"""

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

"""Common and rare word removal"""

#top 10 most common words
common_words = pd.Series(' '.join(df['text']).split()).value_counts()[:10]
freq_m = list(common_words.index)
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_m))

#least commonly used words
rare_words = pd.Series(' '.join(df['text']).split()).value_counts()[-10:]

freq_l = list(rare_words.index)
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_l))

"""Lemmatization"""

from textblob import Word
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(df['text'])
df.to_csv('cleaned_data.csv')

"""**Text Processing**

N-grams
"""

from textblob import TextBlob
TextBlob(df['text'][0]).ngrams(2)

"""Term frequency = (Number of times term T appears in the particular row) /(number of terms in that row)"""

tf1 = (df['text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']

"""IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present"""

for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(df.shape[0]/(len(df[df['text'].str.contains(word)])))

"""TF-IDF = TF * IDF"""

tf1['tf-idf'] = tf1['tf'] * tf1['idf']

"""TF-IDF + pre-processing done quick"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
text_vect = tfidf.fit_transform(df['text'])
feature_names_tfidf = tfidf.get_feature_names()

dense = text_vect.todense()
denselist = dense.tolist()
df_tfidf = pd.DataFrame(denselist, columns=feature_names_tfidf)

df_tfidf.head()

"""Bag of Words"""

from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
text_bow = bow.fit_transform(df['text'])
feature_names_bow = bow.get_feature_names()

dense = text_bow.todense()
denselist = dense.tolist()
df_bow = pd.DataFrame(denselist, columns=feature_names_bow)

df_bow.head()

"""**Sentiment Analysis**

Basic probing
"""

df['text'][:10].apply(lambda x: TextBlob(x).sentiment)

"""Above, you can see that it returns a tuple representing polarity and subjectivity of each tweet from 1 to -1

**Visualizing Tf-idf scores.**

Matplotlib reference: https://matplotlib.org/3.3.3/gallery/statistics/barchart_demo.html#sphx-glr-gallery-statistics-barchart-demo-py
"""

df = pd.read_csv("cleaned_data.csv")

#compute tfidf scores for all tokens
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
text_vect = tfidf.fit_transform(df['text'])
feature_names_tfidf = tfidf.get_feature_names()

dense = text_vect.todense()
denselist = dense.tolist()
df_tfidf = pd.DataFrame(denselist, columns=feature_names_tfidf)

import statistics as stat 
mean_tfidf = df_tfidf.apply(stat.mean, axis=0)
mean_tfidf_sort = mean_tfidf.sort_values(ascending=False)

mean_tfidf_sort[:20]

top_tokens = mean_tfidf_sort[0:20] #get the top 20 tokens

import matplotlib.pyplot as plt

def plot_mean_tfidf(top_tokens):
  fig, ax1 = plt.subplots(figsize=(9, 7))  # Create the figure
  fig.subplots_adjust(left=0.115, right=0.88)

  pos = np.arange(len(top_tokens))

  rects = ax1.barh(pos, [top_tokens[k] for k in range(0,len(top_tokens))],
    align='center',
    height=0.5,
    tick_label=top_tokens.index)
  xlabel = ('Mean TF-IDF Score')
  ax1.set_xlabel(xlabel)
  return fig

fig = plot_mean_tfidf(top_tokens)

fig.savefig('mean_tfidf.png') #save the figure