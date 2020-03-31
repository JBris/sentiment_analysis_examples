#!/usr/bin/env python

# Source: https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
# Author: Aaron Kub - https://towardsdatascience.com/@aaronkub
# Dataset: http://ai.stanford.edu/~amaas/data/sentiment/
# Merged data set can be found at https://github.com/aaronkub/machine-learning-examples/tree/master/imdb-sentiment-analysis

import re
from dataset import load_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def clean_data(reviews_train, reviews_test):
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    NO_SPACE = ""
    SPACE = " "

    def preprocess_reviews(reviews):
        
        reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
        
        return reviews

    reviews_train_clean = preprocess_reviews(reviews_train)
    reviews_test_clean = preprocess_reviews(reviews_test)
    return reviews_train_clean, reviews_test_clean

def model(reviews_train_clean, reviews_test_clean ):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75
    )

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        
        svm = LinearSVC(C=c)
        svm.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" 
            % (c, accuracy_score(y_val, svm.predict(X_val))))
        
    final_svm_ngram = LinearSVC(C=0.01)
    final_svm_ngram.fit(X, target)
    print ("Final Accuracy: %s" 
        % accuracy_score(target, final_svm_ngram.predict(X_test)))

    return ngram_vectorizer, final_svm_ngram

def final_model(reviews_train_clean, reviews_test_clean ):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    stop_words = ['in', 'of', 'at', 'a', 'the']
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75
    )

    for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
        
        svm = LinearSVC(C=c)
        svm.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" 
            % (c, accuracy_score(y_val, svm.predict(X_val))))
        
    final = LinearSVC(C=0.01)
    final.fit(X, target)
    print ("Final Accuracy: %s" 
        % accuracy_score(target, final.predict(X_test)))

    return ngram_vectorizer, final

def most_influential_features(ngram_vectorizer, final):
    feature_to_coef = {
        word: coef for word, coef in zip(
            ngram_vectorizer.get_feature_names(), final.coef_[0]
        )
    }

    for best_positive in sorted(
        feature_to_coef.items(), 
        key=lambda x: x[1], 
        reverse=True)[:30]:
        print (best_positive)
        
    print("\n\n")
    for best_negative in sorted(
        feature_to_coef.items(), 
        key=lambda x: x[1])[:30]:
        print (best_negative)
        
reviews_train, reviews_test = load_data()
reviews_train_clean, reviews_test_clean = clean_data(reviews_train, reviews_test)
#ngram_vectorizer, final = model(reviews_train_clean, reviews_test_clean)
ngram_vectorizer, final = final_model(reviews_train_clean, reviews_test_clean)
most_influential_features(ngram_vectorizer, final)