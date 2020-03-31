#!/usr/bin/env python

# Source: https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1
# Author: Vasista Reddy - https://medium.com/@vasista
# Dataset: http://www.cs.cornell.edu/people/pabo/movie-review-data/
# CSV files can be found at https://github.com/Vasistareddy/sentiment_analysis/tree/master/data

import time
from dataset import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

def model(trainData, testData):
    print(trainData.sample(frac=1).head(5)) # shuffle the df and pick first 5
    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df = 5,
                                max_df = 0.8,
                                sublinear_tf = True,
                                use_idf = True)
    train_vectors = vectorizer.fit_transform(trainData['Content'])
    test_vectors = vectorizer.transform(testData['Content'])

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, trainData['Label'])
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1# results
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    report = classification_report(testData['Label'], prediction_linear, output_dict=True)
    print('positive: ', report['pos'])
    print('negative: ', report['neg'])

    # f1-score = 2 * ((precision * recall)/(precision + recall))

    print()
    def test(review):
        print(review)
        review_vector = vectorizer.transform([review]) # vectorizing
        print(classifier_linear.predict(review_vector))
        print()

    test("""The movie was rubbish. Don't watch it. Boring.""")
    test("""Very funny and very entertaining!""")
    test("""Hilarious. Gut-busting. Family-friendly. Exciting. I love Adam Sandler.""")
    test("""Hilarious. I love Adam Sandler.""") #Returns a negative result...
    test("""Utterly spooky and terrifying. I was sitting on the edge of my seat.""")
    test("""Fell asleep trying to watch this. There are much better movies available.""")

trainData, testData = load_data()
model(trainData, testData)
