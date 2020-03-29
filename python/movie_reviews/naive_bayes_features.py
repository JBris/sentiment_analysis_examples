#!/usr/bin/env python

# Source: https://www.datacamp.com/community/tutorials/simplifying-sentiment-analysis-python
# Author: Sayak Paul - https://www.datacamp.com/profile/spsayakpaul
# Dataset: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from dataset import load_data
import random

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def model(movie_reviews):
    documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    # Define the feature extractor
    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    word_features = list(all_words)[:2000]

    # Train Naive Bayes classifier
    featuresets = [(document_features(d, word_features), c) for (d,c) in documents]
    train_set, test_set = featuresets[100:], featuresets[:100]
    classifier = NaiveBayesClassifier.train(train_set)

    # Test the classifier
    print(nltk.classify.accuracy(classifier, test_set))

    # Show the most important features as interpreted by Naive Bayes
    classifier.show_most_informative_features(10)

movie_reviews = load_data()
model(movie_reviews)