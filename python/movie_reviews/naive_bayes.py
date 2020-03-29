#!/usr/bin/env python

# Source: https://hub.packtpub.com/how-to-perform-sentiment-analysis-using-python-tutorial/
# Author: Sugandha Lahoti - https://hub.packtpub.com/author/sugandhal/
# Extract originally taken from https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-cookbook
# Dataset: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from dataset import load_data

def extract_features(word_list):
    return dict([(word, True) for word in word_list])

def split_dataset(movie_reviews):
    # Load positive and negative reviews  
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')

    features_positive = [(extract_features(movie_reviews.words(fileids=[f])), 
            'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])), 
            'Negative') for f in negative_fileids]

    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))

    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]  
    print("Number of training datapoints:", len(features_train))
    print("Number of test datapoints:", len(features_test))

    return features_train, features_test

def get_input_reviews():
    # Sample input reviews
    return [
        "It is an amazing movie", 
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie", 
        "The direction was terrible and the story was all over the place" 
    ]

def model(features_train, features_test):
    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(features_train)
    print("Accuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))

    print("Top 10 most informative words:")
    for item in classifier.most_informative_features()[:10]:
        print(item[0])
    
    input_reviews = get_input_reviews()

    print ("Predictions:")
    for review in input_reviews:        
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()

        print("Review:", review)
        print ("Predicted sentiment:", pred_sentiment)
        print ("Probability:", round(probdist.prob(pred_sentiment), 2))

movie_reviews = load_data()
features_train, features_test = split_dataset(movie_reviews)
model(features_train, features_test)
