#!/usr/bin/env python

# Source: https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
# Author: Jason Brownlee - https://machinelearningmastery.com/author/jasonb/
# Dataset: http://ai.stanford.edu/~amaas/data/sentiment/

import numpy as np
import matplotlib.pyplot as plt
from dataset import get_imdb

def explore():
    (X_train, y_train), (X_test, y_test) = get_imdb().load_data()
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # summarize size
    print("Training data: ")
    print(X.shape)
    print(y.shape)

    # Summarize number of classes
    print("Classes: ")
    print(np.unique(y))

    # Summarize number of words
    print("Number of words: ")
    print(len(np.unique(np.hstack(X))))

    # Summarize review length
    print("Review length: ")
    result = [len(x) for x in X]
    print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))

    # plot review length
    plt.boxplot(result)
    plt.show()

explore()