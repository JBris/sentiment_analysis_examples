#!/usr/bin/env python

# Source: https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/
# Author: Jason Brownlee - https://machinelearningmastery.com/author/jasonb/
# Dataset: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

import os
import string
from collections import Counter
from nltk.corpus import stopwords
import sys

def usage():
    print("""
    ###############################################################
    Help:

    * Description:
        - Prepares movie reviews for analysis and modelling.
		- Build a vocabulary from training data.
		- Processes, cleans, and filters training data using vocabulary. Writes to file.

    * Usage:
        - ./data_preparation.py [ vocabulary | v ]
        - ./data_preparation.py [ process | p ]
        
    ###############################################################
    """)

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs_vocabulary(directory, vocab):
	# walk through all files in the folder
	for filename in os.listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# load all docs in a directory
def process_docs_process(directory, vocab):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	lines = list()
	# walk through all files in the folder
	for filename in os.listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

def vocabulary():
    dirname = os.path.dirname(os.path.realpath(__file__))
    # define vocab
    vocab = Counter()
    # add all docs to vocab
    process_docs_vocabulary(dirname + '/data/txt_sentoken/neg', vocab)
    process_docs_vocabulary(dirname + '/data/txt_sentoken/pos', vocab)
    # print the size of the vocab
    print(len(vocab))
    # print the top words in the vocab
    print(vocab.most_common(50))
    # keep tokens with > 5 occurrence
    min_occurane = 5
    tokens = [k for k,c in vocab.items() if c >= min_occurane]
    print(len(tokens))
    # save tokens to a vocabulary file
    save_list(tokens, dirname + '/data/txt_sentoken/vocab.txt')

def process():
    dirname = os.path.dirname(os.path.realpath(__file__))
    # load vocabulary
    vocab_filename = dirname + '/data/txt_sentoken/vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    # prepare negative reviews
    negative_lines = process_docs(dirname + '/data/txt_sentoken/neg', vocab)
    save_list(negative_lines, dirname + '/data/txt_sentoken/negative.txt')
    # prepare positive reviews
    positive_lines = process_docs(dirname + '/data/txt_sentoken/pos', vocab)
    save_list(positive_lines, dirname + '/data/txt_sentoken/positive.txt')

if len(sys.argv) < 2:
    usage()
    exit()

operation = sys.argv[1]

if (operation == "v") or (operation == "vocabulary"):
    vocabulary()
    exit()

if (operation == "p") or (operation == "process"):
    process()
    exit()

usage()