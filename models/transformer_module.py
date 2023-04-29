# basic libraries
import sys
import pandas as pd
import numpy as np

# libraries for tokenization
import re # for regular expressions
from nltk import download as dl
from nltk.tokenize import sent_tokenize, word_tokenize

dl("punkt", quiet=True)
from nltk.stem import WordNetLemmatizer
dl("wordnet", quiet=True)
from nltk.corpus import stopwords
dl("stopwords", quiet=True)

# library for word2vec approach
from gensim.models import Word2Vec

# library for sparse data
from scipy.sparse import csr_matrix, vstack

# library for transformer base class
from sklearn.base import TransformerMixin


def run_once(f):
    """decorator function allowing to execute an arbitrary function once if decorated"""

    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


#######################################################


class w2v(TransformerMixin):
    """Transformer class
    Transforms words within their embeddings into vectors, digitizes them and projects each onto a long onehotencoded vector
    """

    def __init__(
        self, resolution_vec=[3, 3, 3, 3, 3, 3], window=3, min_count=1, epochs=10
    ):
        """Initialize w2v object by directly calling set_params"""
        self.set_params(resolution_vec, window, min_count, epochs)
        return None

    def set_params(self, resolution, window, min_count, epochs):
        """Sets parameters for this "word-to-vector" transformer."""

        self.resolution = np.array(
            resolution
        )  # vector covering number of bins per dimension of the "word2vec" vectors
        self.dimension = len(
            self.resolution
        )  # number of dimensions of the "word2vec" vectors
        self.window = window  # length of word window characterizing the range of embedding for each word
        self.min_count = min_count  # number of occurrence for a word in a corpus to take into account
        self.epochs = epochs  # number of times the corpus is iterated for building the word2vec model

        return self

    def fit(self, X, y=None):
        """fits model using message corpus X"""

        messages = [
            tokenize(s) for message in X for s in sent_tokenize(message)
        ]  # transforms corpus into tokenized messages
        self.model = Word2Vec(
            sentences=messages,
            vector_size=self.dimension,
            window=self.window,
            min_count=self.min_count,
            workers=64,
            epochs=self.epochs,
        )

        self.build_word2vec_bin_vector(messages)

        return self

    def transform(self, X):
        """transforms message(s) "X" into histogram vector(s)
        returns sparse csr_matrix
        """
        return vstack(pd.Series(X).apply(self.message2histogram))

    def build_word2vec_bin_vector(self, messages):
        """creates binning vector and saves it in this object
        output: None
        """
        self.search_min_max_values_for_each_word2vec_dimension_in_corpus(messages)
        self.bins = [
            np.linspace(self.xmin[i], self.xmax[i], self.resolution[i])
            for i in range(self.dimension)
        ]  # builds bins depending on the min max values
        return None

    def message2histogram(self, message):
        """For each word in a message, one long onehotencoded vector is created.
        All onehotencoded vectors stemming from one message are summarized into a 1-D histogram.
        """

        histogram0 = np.array(
            [0 for i in range(pd.Series(self.resolution).product())]
        )  # builds a 1-D histogram vector with zeros of the correct length
        data = self.histogram(message, histogram0)

        return csr_matrix(data)

    def search_min_max_values_for_each_word2vec_dimension_in_corpus(self, messages):
        """searches through each "word2vec" vector in the corpus and determines the min and max value in each dimension
        min and max values are saved in object vectors xmax and xmin
        output: None
        """
        # sets min and max value for the first grap-able word
        find = False
        for message in messages:
            for word in message:
                try:
                    self.xmin = self.model.wv[word].copy()
                    self.xmax = self.xmin.copy()
                    find = True
                    break
                except:
                    pass
            if find:
                break

        # search through corpus the min-max-values for each vector component
        for message in messages:
            for word in message:
                try:
                    v = self.model.wv[word]
                    test = np.array([self.xmin, v]).T
                    self.xmin = [min(test[i]) for i in range(self.dimension)]
                    test = np.array([self.xmax, v]).T
                    self.xmax = [max(test[i]) for i in range(self.dimension)]
                except:  # if word is not in the corpus
                    pass
        return None

    def histogram(self, message, histogram0):
        """determines histogram for a message
        input: text message
        output: histogram as vector of integers
        """
        histogram = histogram0

        # calculates "basevec" which can map a "word2vec" vector onto a unique one-dimensional onehotencoded vector using a scalar product
        @run_once
        def init_basevec():
            self.basevec = [1] * self.dimension
            for i in range(1, self.dimension, 1):
                self.basevec[i] = self.resolution[i - 1] * self.basevec[i - 1]
            return None

        init_basevec()

        # calculates a 1-D histogram representing each text message "message"
        for word in tokenize(message):
            try:
                vec = self.model.wv[
                    word
                ]  # calculate "word2vec" vector "vec" for a given word "word"
                index_multi_dim = np.array(
                    [
                        min(
                            max(0, np.digitize(vec[i], self.bins[i], right=True) - 1),
                            self.resolution[i] - 1,
                        )
                        for i in range(self.dimension)
                    ]
                )  # discretizes "word2vec" vector "vec"

                # determines the index position of the "1" in a onehotencoded vector representation of "vec"
                index_1d = np.dot(index_multi_dim, self.basevec)

                # building the 1-D histogram
                histogram[index_1d] = histogram[index_1d] + 1
            except:
                pass

        return histogram


#######################################################


def tokenize(text):
    """Replaces URLs with a placeholder and deletes non-specific words.
    Then tokenizes text.
    output: array-like word tokens of cleaned text
    """

    # replaces urls with "urlplaceholder"
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenizes text in words and lemmatizes each word; excludes stop words and  '"'
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            if tok!='"':
                clean_tok = lemmatizer.lemmatize(tok).lower().strip()
                clean_tokens.append(clean_tok)

    return clean_tokens
