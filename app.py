#
from pydoc import describe
#import load_model
from tensorflow.keras.models import load_model
#
#numpy is for mathematical stuff
import numpy as np
#pandas is for dataframe handling and time series 
import pandas as pd
#matplotlib is for graphs
import matplotlib.pyplot as plt
#yfinance is for stock data
import yfinance as yf
#
#datetime related, timedelta is for finding no days between two dates
from datetime import datetime,timedelta
#
#streamlit is framework
import streamlit as st
#
st.title("SPREDICTOR")
#
#Hiding made by streamlit comment
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)
#Hist
model100 = load_model('hist_keras_model.h5')
#Model input 100 days data --> OP - 101st days price
#Sent
#Hist
#Form for inputs
with st.form(key='form1'):
        
    ticker = st.text_input('Enter Stock Ticker','AAPL')
    #
    keywords = st.text_input("Enter Comma separated keywords",'Apple,AAPL,Iphone')
    submit_but = st.form_submit_button(label = 'Load data')#True or False
if submit_but:
    st.success("Loading data...")

#with st.spinner('Wait for it...'):
#    time.sleep(5)
#st.success('Done!')
keywords = keywords.split(',')#['Apple','AAPL','Iphone']
keywords = [' OR '.join(keywords)]#['Apple OR AAPL OR Iphone']


end100 = datetime.now() #today/28/may
start100 = end100 - timedelta(200) #28 may -200days
df100 = yf.download(ticker,start100,end100)
#Sent
end = datetime.now()
start = end - timedelta(30)
#
import snscrape.modules.twitter as sntwitter
#import pandas as pd
#
company_data = yf.download(ticker,start,end)#30- days
company_data = company_data.tail(3)#3days
print(company_data)
#company = [d1,d2,d3]
end = company_data.index[-1]
start = str(company_data.index[0])[:10]
time_series = pd.date_range(start, periods=6, freq="D") #tweet collection - 6 days

# Creating list to append tweet data to
tweets_list2 = []
#
# Using TwitterSearchScraper to scrape data and append tweets to list
#regular expression - cleaning and etc
import re
#Rule
eng = re.compile('^[a-zA-Z0-9. -_?]*$')
for k in time_series:#for each date in timeseries[date1,date2...]
    flag = 0 #flag to 0 means maximum tweet limit not reached
    #print(k)
    k1 = k.strftime("%Y-%m-%d")[:10] #starting time
    k2 = (k+pd.Timedelta("1d")).strftime("%Y-%m-%d")[:10] #ending time
    #print(f'APPL OR Apple since:{k1} until:{k2}')
    maxTweets = 20
    if flag == 1:
        continue
    try:
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keywords[0]}:{k1} until:{k2}').get_items()):#0,"worst stock"
            #print("im in")
            #print(k,i,tweet)
            if i>maxTweets:
                i = 0
                flag = 1
                break
            if eng.match(tweet.content):#hindi
                tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    except:
        continue
#
# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
# Display first 5 entries from dataframe
tweets_df = tweets_df2.copy(deep=False)
#
def clean(tweet):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    # we then use the sub method to replace anything matching
    tweet = whitespace.sub(' ', tweet)
    tweet = web_address.sub('', tweet)
    tweet = user.sub('', tweet)
    return tweet
#
#Text sentiment analysis --- Good/positive = 1, bad/regative = -1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#
sid_obj = SentimentIntensityAnalyzer()
#
#probs = []
sentiments = []#[+1,-0.9,-1...]['positive,negative...]

# use regex expressions (in clean function) to clean tweets
tweets_df2['Text'] = tweets_df2['Text'].apply(clean)

for tweet in tweets_df2['Text'].to_list():
    # make prediction
    score = sid_obj.polarity_scores(tweet) 
    sent = score['compound']
    if sent>=0:
        sent = 'POSITIVE'
    else:
        sent = 'NEGATIVE'
    # extract sentiment prediction
    try:
        sentiments.append(sent)  # 'POSITIVE' or 'NEGATIVE'
    except:
        sentiments.append(0)  # 'POSITIVE' or 'NEGATIVE'
        continue
# add probability and sentiment predictions to tweets dataframe
tweets_df2['sentiment'] = sentiments
#
encoding = {"sentiment":{"POSITIVE" : 1, "NEGATIVE":-1}}    
new_df = tweets_df2.replace(encoding)#29/june
#
from datetime import datetime
#
new_df['Datetime'] = new_df['Datetime'].dt.strftime("%Y-%m-%d")
new_df['Datetime'] = pd.to_datetime(new_df['Datetime'], errors='coerce')
new_df.set_index('Datetime',inplace=True)
agg_df = new_df.resample('D').agg({'sentiment':'mean'})
#
#
#NEWS
from newsapi import NewsApiClient
#import pandas as pd
#
news_list = []
newsapi = NewsApiClient(api_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#
def get_content(keyword):
    everything = newsapi.get_everything(q=keyword)
    x = pd.DataFrame(everything['articles'])
    x = x.drop(['source','author','url','urlToImage'],axis = 1)
    x['publishedAt'] = pd.to_datetime(x.publishedAt)#required datetime format
    x['publishedAt'] = x['publishedAt'].dt.strftime('%Y-%m-%d')#format
    x = x.set_index('publishedAt')
    x = x.sort_index()
    news_list.append(x)
    return x
#
#
company = yf.Ticker(ticker)
currency = company.info['currency']

try:
    name = company.info['longName']
    name = name.split()
    name = name[0][0:]
except:
    name = company.info['shortName']
#
#Critical -- kwd
kwd = f'{name} Stock prices'
print(kwd)
articles = get_content(kwd)#list of news
news_sent_list = []#list for sentiments of each news
#Sent_analyzer
#data news heading, date
def get_res(data):
    content = data.reset_index()
    content = get_sent(content)
    news_sent_list.append(content)
    return content

def get_sent(data):
    sentiments = []
    for newsh in data['title'].to_list():
        score = sid_obj.polarity_scores(newsh)
        sent = score['compound']
        try:
            sentiments.append(sent)  # 'POSITIVE' or 'NEGATIVE'
        except:
            sentiments.append(0)  # 'POSITIVE' or 'NEGATIVE'
            continue
    data['sentiment'] = sentiments
    data['publishedAt'] = data['publishedAt'].astype('datetime64')
    data.set_index('publishedAt',inplace=True)
    data = data.resample('D').agg({'sentiment':'mean'})
    data['sentiment']=data['sentiment'].fillna(data.mean()[0])
    return data
##
news = get_res(articles)
#
final_agg_df = pd.DataFrame(agg_df['sentiment'].copy(deep=False))
news_df = news.copy(deep=False)
print(news_df)
#
agg_df['extra'] = news['sentiment']
agg_df = agg_df.fillna(0)
final_agg_df['sentiment'] = (final_agg_df.sentiment + agg_df.extra)/2
agg_df = final_agg_df
print(agg_df)
#
st.subheader('Past 3 days Sentiment data')
fig2 = plt.figure(figsize=(12,6))
#
plt.plot((agg_df['sentiment'].resample('D').agg({'sentiment':'mean'})))
#--------------------------------------------------------------------------------------
st.write(fig2)
#
agg_df['sentiment'][agg_df.sentiment>=0.5] = 1
agg_df['sentiment'][agg_df.sentiment<0.5] = -1
agg_df.index.names = ['Date']
print(agg_df)
#
final_df = pd.merge(company_data,agg_df,on='Date')
print(final_df)
#-----------------------------------------------------------------------------------------------
model = load_model('latest_hist_keras_model_3day.h5',compile=False)
final_df = final_df.reset_index()
final_df = final_df.drop(['Date','Adj Close'],axis = 1)
plt.plot(final_df.Close)
df = final_df
data_testing = pd.DataFrame(df[['Close','sentiment']])
#
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
#
input_data = scaler.fit_transform(data_testing)
import numpy as np
x_test = input_data
x_test= np.array(x_test)
print(x_test)
x_test = x_test.reshape(1,3,2)#1 record of 3 rows(3days) and 2 columns(close,sentiment)
y_pred = model.predict(x_test)#sentiment model output prediction
temp = y_pred.copy()
lst = []
for i in temp:
    k = list(i)
    k.append(0)
    lst.append(k)
predn = np.array([np.array(i) for i in lst])
predn = scaler.inverse_transform(predn)
predicted = predn[:,0]
#
#
rounded = "{:.2f}".format(predicted[0])
st.subheader(f'Predicted Price for tomorrow based on sentiment data: {currency} {rounded}')
print("Done")
#
st.subheader(f'{name} Stock data between {str(start100)[:10]} and {str(end100)[:10]} (100 days)')
st.write(df100)
#
df100 = df100.reset_index()
df100 = df100.drop(['Date','Adj Close'],axis = 1)
#
st.subheader('Closing Price Graph')
fig = plt.figure(figsize=(12,6))
plt.plot(df100.Close)
st.pyplot(fig)
#
df_100 = df100.tail(100)
data100 = pd.DataFrame(df_100['Close'])
#
from sklearn.preprocessing import MinMaxScaler
scaler100 = MinMaxScaler(feature_range=(0,1))
input_data100 = scaler100.fit_transform(data100)
#
x_test100 = input_data100
x_test100 = np.array(x_test100)
#
x_test100 = x_test100.reshape(1,100,1)#1 record of 100 rows(100 days) and 1 column(close)
y_pred100 = model100.predict(x_test100)
#
y_pred100 = scaler100.inverse_transform(y_pred100)
final_prediction100 = y_pred100[0][0]
rounded100 = "{:.2f}".format(final_prediction100)
st.subheader(f'Predicted Price for tomorrow based on past 100 days data : {currency} {rounded100}')
fp = (float(rounded) + float(rounded100))/2
fp = "{:.2f}".format(fp)

st.subheader(f'Recommended prediction price : {currency} {fp}')
