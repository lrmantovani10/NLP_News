import requests
import json
import pandas as pd
import datetime
from dateutil import rrule
import re

# DOcumentation link: https://open-platform.theguardian.com/documentation/

"""We can also try the **Guardian API**... Make sure you are using the correct API key."""
GUARDIAN_KEY = "caaf6800-bac7-4fe3-bfb4-a6c233ba51e5"

title = []
date = []
text = []

#this function extracts the data we are interested in from the API response
def get_article_data(response, title, date, text):
  articles = response.json()['response']['results']
  for doc in articles:
    title.append(doc['webTitle'])
    date.append(doc['webPublicationDate'])
    text.append(doc['fields']['body'])

start_date = datetime.datetime.strptime("2018-02-01", '%Y-%m-%d')
from_date = "2018-01-01"
end_date = datetime.date.today()
#this loop requests 200 articles per month, uses the function above to get useful data, & saves the data
for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
    to_date = dt.strftime('%Y-%m-%d')
    guard_url = "https://content.guardianapis.com/search?show-fields=body&q='artificial%20AND%20intelligence%20OR%20neural%20AND%20networks'&from-date=" + from_date + "&to-date=" + to_date + "&page-size=200&api-key=" + GUARDIAN_KEY
    from_date = to_date
    articles = requests.get(guard_url)
    get_article_data(articles, title, date, text)

df = {'title': title,
     'date': date,
     'text' : text}
df = pd.DataFrame(df) 
df.to_csv('my_data.csv',index=False)

#print('First entry: '+df['text'][1])