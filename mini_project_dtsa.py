import streamlit as st

st.title('Twitter Sentiment Analysis')
with st.form("User Input:"):
    keyword= st.text_input("Enter The Topic")
    noOfTweet= st.number_input("Enter The Number of Tweets",1,5000)
    submitted = st.form_submit_button("Submit")

st.title('Select Algorithms of your choice: ')
classify1 = st.checkbox('Naive Bayes')
classify2 = st.checkbox('Logistic Regression')
classify3 = st.checkbox('SVM')
classify4 = st.checkbox('XGboost')
classify5 = st.checkbox('Decision Tree')

import nltk
nltk.download('vader_lexicon')

import nltk
nltk.download('stopwords')

# Import Libraries
from textblob import TextBlob
import seaborn as sns
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# Authentication
consumerKey = "WjFAh2SyEk35R64eED8qQUp19"
consumerSecret = "vpw5aC1p8Hs0AjMvj42ucMl0AwEM4ZOUubqtf6PEXIGe8l6EDj"
accessToken = "1253231215525662720-Ij0fmMyVJ9uTYipL2JPqbDOO1Npdp3"
accessTokenSecret = "A8w7jOKvav5r68FCPk9lmJFUnGGhCxZTQYArx0bzIGOp9"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

def percentage(part,whole):
 return 100 * float(part)/float(whole)

#Initialization
tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

#Sentiment Analysis
for tweet in tweets:
  tweet_list.append(tweet.text)
  analysis = TextBlob(tweet.text)
  score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
  neg = score['neg']
  neu = score['neu']
  pos = score['pos']
  comp = score['compound']
  polarity += analysis.sentiment.polarity

  #Appending to List
  if neg > pos:
    negative_list.append(tweet.text)
    negative += 1
  elif pos > neg:
    positive_list.append(tweet.text)
    positive += 1
  elif pos == neg:
    neutral_list.append(tweet.text)
    neutral += 1

positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')

#Number of Tweets (Total, Positive, Negative, Neutral)
tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
st.write("total number: ",len(tweet_list))
st.write("positive number: ",len(positive_list))
st.write("negative number: ", len(negative_list))
st.write("neutral number: ",len(neutral_list))

st.set_option('deprecation.showPyplotGlobalUse', False)

#define data
st.title("PieChart of The Sentiment Analysis")
data = [positive, neutral, negative]
labels = ['positive', 'neutral', 'negative']
#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:5]
#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%1.01f%%')
st.pyplot()

st.subheader("Fetched Tweets")
tweet_list

tweet_list.drop_duplicates(inplace = True)

#Cleaning Text (RT, Punctuation etc)

#Creating new dataframe and new features
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]

#Removing RT, Punctuation etc
remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
rt = lambda x: re.sub("(@[A-Za-z0–9]+) or ([0-9A-Za-z \t]) or (\w+:\/\/\S+)"," ",x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()

#Calculating Negative, Positive, Neutral and Compound values
tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
  score = SentimentIntensityAnalyzer().polarity_scores(row)
  neg = score['neg']
  neu = score['neu']
  pos = score['pos']
  comp = score['compound']
  if neg > pos:
    tw_list.loc[index, 'sentiment'] = "negative"
    tw_list.loc[index, 'neg'] = neg
  elif pos > neg:
    tw_list.loc[index, 'sentiment'] = "positive"
    tw_list.loc[index, 'pos'] = pos
  else:
    tw_list.loc[index, 'sentiment'] = "neutral"
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'compound'] = comp

#Creating new data frames for all sentiments (positive, negative and neutral)
tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]

def count_values_in_column(data,feature):
  total=data.loc[:,feature].value_counts(dropna=False)
  percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
  return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

#Count_values for sentiment
count_values_in_column(tw_list,"sentiment")

#Calculating tweet’s length and word count
tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))
round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()),2)
round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()),2)

#Removing Punctuation
def cleaning_URLs(text):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',text)
tw_list['text'] = tw_list['text'].apply(lambda x: cleaning_URLs(x))
tw_list['text'].tail()

def cleaning_numbers(text):
    return re.sub('[0-9]+', '', text)
tw_list['text'] = tw_list['text'].apply(lambda x: cleaning_numbers(x))
tw_list['text'].tail()

def remove_punct(text):
 text = "".join([char for char in text if char not in string.punctuation])
 text = re.sub('[0–9]+', '', text)
 return text
tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))

#Appliyng tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text
tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))

#Removing stopwords
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))

#Appliyng Stemmer
ps = nltk.PorterStemmer()
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text
tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))

#Cleaning Text
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text
clean = {"sentiment":{"neutral": 0,"positive": 1, "negative": 2}}
tw_list = tw_list.replace(clean)

#Appliyng Countvectorizer
countVectorizer = CountVectorizer(analyzer=clean_text)
countVector = countVectorizer.fit_transform(tw_list['text'])
st.write('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()

# Most Used Words
count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0,ascending=False).head(20)
st.subheader('Words That appeared Most Frequently:')
countdf[1:11]

#Function to ngram
def get_top_n_gram(corpus,ngram_range,n=None):
 vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
 bag_of_words = vec.transform(corpus)
 sum_words = bag_of_words.sum(axis=0)
 words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
 words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
 return words_freq[:n]

#n2_bigram
n2_bigrams = get_top_n_gram(tw_list['text'],(2,2),20)

#n3_trigram
n3_trigrams = get_top_n_gram(tw_list['text'],(3,3),20)

st.subheader("Tokenization of Tweets text")
tw_list['stemmed']

import nltk
nltk.download('wordnet')
lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
tw_list['stemmed'] = tw_list['stemmed'].apply(lambda x: lemmatizer_on_text(x))
tw_list['stemmed'].head()

X=tw_list.stemmed
y=tw_list.sentiment

# Separating the 80% data for training data and 20% for testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state =42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
vectorizer.fit_transform(X_train)

X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if classify1:
    st.subheader('Naive Bayes:')
    from nltk.classify.scikitlearn import SklearnClassifier
    from sklearn.naive_bayes import BernoulliNB
    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train, y_train)
    y_pred1 = BNBmodel.predict(X_test)
    # Print the evaluation metrics for the dataset.
    st.write(classification_report(y_test, y_pred1))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred1)

if classify2:
    st.subheader('Logisitic Regression:')
    from sklearn.linear_model import LogisticRegression
    lmodel=LogisticRegression()
    lmodel.fit(X_train, y_train)
    y_pred2 = lmodel.predict(X_test)
    # Print the evaluation metrics for the dataset.
    st.write(classification_report(y_test, y_pred2))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred2)

if classify3:
    st.subheader('Support Vector Machine:')
    from sklearn import svm
    SVCmodel = svm.SVC()
    SVCmodel.fit(X_train, y_train)
    y_pred3 = SVCmodel.predict(X_test)
    # Print the evaluation metrics for the dataset.
    st.write(classification_report(y_test, y_pred3))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred3)

if classify4:
    st.subheader('XGBoost:')
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    y_pred4 = xgb.predict(X_test)
    from sklearn.metrics import classification_report
    st.write(classification_report(y_test,y_pred4))


if classify5:
    st.subheader('Decision Tree:')
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    #Predict the response for test dataset
    y_pred5 = clf.predict(X_test)
    from sklearn.metrics import classification_report
    st.write(classification_report(y_test,y_pred5))

st.title('Comparision between The Classifiers')

from sklearn.metrics import accuracy_score

# creating the dataset
data = {}
if classify1:
    a = accuracy_score(y_test, y_pred1)
    data['NB']=a
if classify2:
    b = accuracy_score(y_test, y_pred2)
    data['LR']=b
if classify3:
    c = accuracy_score(y_test, y_pred3)
    data['SVM']=c
if classify4:
    d = accuracy_score(y_test, y_pred4)
    data['XGB']=d
if classify5:
    e = accuracy_score(y_test, y_pred5)
    data['DT']=e

st.write(data)
courses = list(data.keys())
values = list(data.values())

# creating the bar plot
chart_data = pd.DataFrame(values,courses)
st.bar_chart(chart_data)