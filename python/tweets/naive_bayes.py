#!/usr/bin/env python

# Source: https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
# Author: Shaumik Daityari - https://www.digitalocean.com/community/users/sdaityari
# Dataset: Twitter Streaming API - https://dev.twitter.com/overview/documentation
# Data set originally retrieved using the nltk database.

import random
import re, string
from dataset import load_data
from nltk import classify
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def tokenize(twitter_samples):
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    print("tokenize")
    print(tweet_tokens[0])
    return positive_tweets, negative_tweets, text, tweet_tokens

def normalize(twitter_samples, positive_tweets, negative_tweets, text, tweet_tokens):
    print("normalize")
    print(lemmatize_sentence(tweet_tokens[0]))

def clean(tweet_tokens):
    stop_words = stopwords.words('english')
    print("clean")
    print(remove_noise(tweet_tokens[0], stop_words))

def examples():
    positive_tweets, negative_tweets, text, tweet_tokens = tokenize(twitter_samples)
    normalize(twitter_samples, positive_tweets, negative_tweets, text, tweet_tokens)
    clean(tweet_tokens)

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
        
def main(twitter_samples):
    #Preprocess data
    stop_words = stopwords.words('english')
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    print(positive_tweet_tokens[500])
    print(positive_cleaned_tokens_list[500])

    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    all_neg_words = get_all_words(negative_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    #Prepare data for modelling
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    #Split data set
    positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)
    train_data = dataset[:7000]
    test_data = dataset[7000:]

    #Model 
    classifier = NaiveBayesClassifier.train(train_data)
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))

    def test_custom_tweet(custom_tweet):
        print(custom_tweet)
        custom_tokens = remove_noise(word_tokenize(custom_tweet))
        print(classifier.classify(dict([token, True] for token in custom_tokens)))

    negative_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
    test_custom_tweet(negative_tweet)
    
    positive_tweet = 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'
    test_custom_tweet(positive_tweet)

    sarcastic_tweet = 'Thank you for sending my baggage to CityX and flying me to CityY at the same time. Brilliant service. #thanksGenericAirline'
    test_custom_tweet(sarcastic_tweet)

twitter_samples = load_data()
main(twitter_samples)