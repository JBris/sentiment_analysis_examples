#!/usr/bin/env python

import os
from nltk.corpus import CategorizedPlaintextCorpusReader

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    return CategorizedPlaintextCorpusReader(dirname + '/data/txt_sentoken', r'.*\.txt', cat_pattern=r'(\w+)/*')
