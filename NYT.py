# -*- coding: utf-8 -*-

#NYT API docs: https://developer.nytimes.com/apis
import requests
import json
import pandas as pd
import datetime
from dateutil import rrule
import re


API_KEY = "NYT api key"
month_url = "https://api.nytimes.com/svc/archive/v1/2019/1.json?api-key=" + API_KEY
r = requests.get(month_url)

#all articles containing "artificial intelligence" from 2020
ai = requests.get(search_url)

#all comments from one article
#Note: comments not available for all articles
comments_url = "https://api.nytimes.com/svc/community/v3/user-content/url.json?api-key=" + API_KEY + "&offset=0&url=https%3A%2F%2Fwww.nytimes.com%2F2019%2F06%2F21%2Fscience%2Fgiant-squid-cephalopod-video.html"
c = requests.get(comments_url)

#one article
r.json()['response']['docs'][2]
#number of articles returned 
len(r.json()['response']['docs'])

#article title
r.json()['response']['docs'][1]['news_desk']

#function to insert page number, date range in url; date format YYYYMMDD
#also input search words & your api key
def make_search_url(pg_num, start_date, end_date, api_key):
    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q=artificial+intelligence&begin_date=" + str(start_date) + "&end_date=" + str(end_date) + "&page=" + str(pg_num) + "&api-key=" + str(api_key)
    return url

def make_comments_url(article_url, api_key):
    url = "https://api.nytimes.com/svc/community/v3/user-content/url.json?api-key=" + str(api_key) + "&offset=0&url=" + article_url
    return url

"""Helpful things to know about the API that you are using:

1) *Request limits*:

The NYT and the Guardian have limits to the number of requests you can make per second/minute and per day. You won't be able to make requests after you reach that limit. 

NYT: no more than 10 requests per minute or 4000 requests per day

Guardian: no more than 12 per second or 5000 per day

2) *Number of Results Returned*:

NYT: 10 results per page, up to 100 pages 

Guardian: defaut is 10 per page, but you can change this with the page-size parameter (example below)
"""

#Note: this will take a while to run 
#input api key
ai_docs = []
for page in range(0,100):
    url = make_search_url(page,20200101,20200530, API_KEY)
    req = requests.get(url)
    all_docs = req.json()['response']['docs']
    ai_docs = ai_docs + all_docs
    #print(i)
    #print(url)
    #print(len(ai_docs))
    if (len(all_docs) < 10):
        break 
    time.sleep(6) #pause 6 seconds

#initialize empty lists that will be columns of data frame
title = []
date = []
news_desk = []
abstract = []
url = []

#iterate through docs and extract data
for doc in ai_docs:
    title.append(doc['headline']['main'])
    date.append(doc['pub_date'])
    news_desk.append(doc['news_desk'])
    abstract.append(doc['abstract'])
    url.append(doc['web_url'])

#make dictionary & convert to pandas dataframe & save as csv
df = {'title': title,
     'date': date,
     'news_desk': news_desk,
     'abstract': abstract,
     'url' : url}
df = pd.DataFrame(df) 
df.to_csv('NYT_ai.csv')

#import csv as pandas dataframe
articles = pd.read_csv('NYT_ai.csv')










