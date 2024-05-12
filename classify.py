import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    return tweet.strip()

def read_tweets(filename):
    df = pd.read_csv(filename)
    df['tweet'] = df['tweet'].apply(preprocess_tweet)
    return df

def predict_depressed_tweets(df):
    X = vectorizer.transform(df['tweet'])
    df['depressed'] = model.predict(X)
    return df

tweets_df = read_tweets('random_tweets.csv')

tweets_df = predict_depressed_tweets(tweets_df)

plt.figure(figsize=(6, 4))
sns.countplot(x='depressed', data=tweets_df, palette='pastel', hue='depressed', legend=False)
plt.title('Class Distribution')
plt.xlabel('Depressed')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Depressed', 'Depressed'])
plt.show()

depressed_tweets_df = tweets_df[tweets_df['depressed'] == 1]

depressed_tweets_df.to_csv('depressed_tweets.txt', columns=['tweet'], index=False, header=None)

print("First ten depressed tweets:")
print('-'*30)
for tweet in depressed_tweets_df['tweet'].head(10):
    print(tweet)
print('-'*30)
