#!/usr/bin/env python

# Source: https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
# Author: Aaron Kub - https://towardsdatascience.com/@aaronkub
# Dataset: http://ai.stanford.edu/~amaas/data/sentiment/
# Merged data set can be found at https://github.com/aaronkub/machine-learning-examples/tree/master/imdb-sentiment-analysis

import re
from dataset import load_data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

def baseline(reviews_train_clean, reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]

    baseline_vectorizer = CountVectorizer(binary=True)
    baseline_vectorizer.fit(reviews_train_clean)
    X_baseline = baseline_vectorizer.transform(reviews_train_clean)
    X_test_baseline = baseline_vectorizer.transform(reviews_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(
        X_baseline, target, train_size = 0.75
    )

    print("Baseline")
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" 
            % (c, accuracy_score(y_val, lr.predict(X_val))))

    final_model = LogisticRegression(C=0.05)
    final_model.fit(X_baseline, target)
    print ("Final Accuracy: %s" 
        % accuracy_score(target, final_model.predict(X_test_baseline)))

def process(reviews_train_clean, reviews_test_clean):
    # Removing stop words
    english_stop_words = stopwords.words('english')
    def remove_stop_words(corpus):
        removed_stop_words = []
        for review in corpus:
            removed_stop_words.append(
                ' '.join([word for word in review.split() 
                        if word not in english_stop_words])
            )
        return removed_stop_words

    no_stop_words = remove_stop_words(reviews_train_clean)
    no_stop_words_test = remove_stop_words(reviews_test_clean)

    # Stemming 
    def get_stemmed_text(corpus):
        stemmer = PorterStemmer()
        return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

    stemmed_reviews_train = get_stemmed_text(reviews_train_clean)
    stemmed_reviews_test = get_stemmed_text(reviews_test_clean)

    # Lemmatization
    def get_lemmatized_text(corpus):
        lemmatizer = WordNetLemmatizer()
        return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

    lemmatized_reviews_train = get_lemmatized_text(reviews_train_clean)
    lemmatized_reviews_test = get_lemmatized_text(reviews_test_clean)

def n_grams(reviews_train_clean, reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75
    )

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" 
            % (c, accuracy_score(y_val, lr.predict(X_val))))
        
    final_ngram = LogisticRegression(C=0.5)
    final_ngram.fit(X, target)
    print ("Final Accuracy: %s" 
        % accuracy_score(target, final_ngram.predict(X_test)))

def word_counts(reviews_train_clean, reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    wc_vectorizer = CountVectorizer(binary=False)
    wc_vectorizer.fit(reviews_train_clean)
    X = wc_vectorizer.transform(reviews_train_clean)
    X_test = wc_vectorizer.transform(reviews_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75, 
    )

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" 
            % (c, accuracy_score(y_val, lr.predict(X_val))))
        
    final_wc = LogisticRegression(C=0.05)
    final_wc.fit(X, target)
    print ("Final Accuracy: %s" 
        % accuracy_score(target, final_wc.predict(X_test)))

def tf_idf(reviews_train_clean, reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(reviews_train_clean)
    X = tfidf_vectorizer.transform(reviews_train_clean)
    X_test = tfidf_vectorizer.transform(reviews_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75
    )

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" 
            % (c, accuracy_score(y_val, lr.predict(X_val))))
        
    final_tfidf = LogisticRegression(C=1)
    final_tfidf.fit(X, target)
    print ("Final Accuracy: %s" 
        % accuracy_score(target, final_tfidf.predict(X_test)))

reviews_train, reviews_test = load_data()
reviews_train_clean, reviews_test_clean = clean_data(reviews_train, reviews_test)
#baseline(reviews_train_clean, reviews_test_clean)
#process(reviews_train_clean, reviews_test_clean )
#n_grams(reviews_train_clean, reviews_test_clean)
#word_counts(reviews_train_clean, reviews_test_clean)
tf_idf(reviews_train_clean, reviews_test_clean)