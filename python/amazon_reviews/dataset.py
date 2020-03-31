#!/usr/bin/env python

import os
import pandas as pd

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    trainData = pd.read_csv(dirname +'/data/train.csv')
    testData = pd.read_csv(dirname +'/data/test.csv')
    return trainData, testData