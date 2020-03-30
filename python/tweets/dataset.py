#!/usr/bin/env python

import os
from nltk.corpus import TwitterCorpusReader

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    return TwitterCorpusReader(dirname + '/data/twitter_samples', r'.*\.json')
