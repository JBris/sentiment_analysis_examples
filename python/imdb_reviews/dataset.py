#!/usr/bin/env python

import os
from keras.datasets import imdb

def get_imdb():
    return imdb

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    reviews_train = []
    for line in open(dirname + '/data/movie_data/full_train.txt', 'r', encoding="utf8"):
        reviews_train.append(line.strip())
        
    reviews_test = []
    for line in open(dirname + '/data/movie_data/full_test.txt', 'r', encoding="utf8"):
        reviews_test.append(line.strip())

    return reviews_train, reviews_test