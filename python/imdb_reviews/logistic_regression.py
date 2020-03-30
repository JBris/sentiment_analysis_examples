#!/usr/bin/env python

import re
from dataset import load_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def clean_data(reviews_train, reviews_test):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    def preprocess_reviews(reviews):
        reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
        return reviews

    reviews_train_clean = preprocess_reviews(reviews_train)
    reviews_test_clean = preprocess_reviews(reviews_test)
    return reviews_train_clean, reviews_test_clean

def vectorize(reviews_train_clean, reviews_test_clean):
    cv = CountVectorizer(binary=True)
    cv.fit(reviews_train_clean) 
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)
    return X, X_test, cv

def model(X, X_test, cv):
    # Build classifier
    target = [1 if i < 12500 else 0 for i in range(25000)]

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75
    )

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" 
            % (c, accuracy_score(y_val, lr.predict(X_val))))

    # Train model
    final_model = LogisticRegression(C=0.05)
    final_model.fit(X, target)
    print ("Final Accuracy: %s" 
        % accuracy_score(target, final_model.predict(X_test)))
    # Final Accuracy: 0.88128

    # Sanity check
    feature_to_coef = {
        word: coef for word, coef in zip(
            cv.get_feature_names(), final_model.coef_[0]
        )
    }
    for best_positive in sorted(
        feature_to_coef.items(), 
        key=lambda x: x[1], 
        reverse=True)[:5]:
        print (best_positive)
        
    for best_negative in sorted(
        feature_to_coef.items(), 
        key=lambda x: x[1])[:5]:
        print (best_negative)

reviews_train, reviews_test = load_data()
reviews_train_clean, reviews_test_clean = clean_data(reviews_train, reviews_test)
X, X_test, cv = vectorize(reviews_train_clean, reviews_test_clean)
model(X, X_test, cv)
