ARG BASE_IMAGE

FROM $BASE_IMAGE

ENV PYTHON_PACKAGES="\
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    pandas \
    nltk \
    seaborn \
    keras \
    tensorflow \
    tqdm \
" 

RUN pip install --upgrade pip && pip install --no-cache-dir $PYTHON_PACKAGES

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords'); "