import numpy as np
import pathlib

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
from gensim import downloader
from gensim.models import Word2Vec

from preprocess import Preprocess


class Doc2VecVectorizer:

    def __init__(self, vector_size, window, workers, model_path):
        self.workers = workers
        self.window = window
        self.vector_size = vector_size
        self.model_path = model_path

        self.preprocess = Preprocess()

    def load_model(self):
        return Doc2Vec.load(self.model_path)

    def save_model(self, model):
        if model is not None:
            model.save(self.model_path)
        else:
            raise Exception("Model has not been trained to be saved")

    def model_exists(self):
        path = pathlib.Path(self.model_path)
        if path.exists():
            return True
        return False

    def fit(self, X, y):
        if self.model_exists():
            self.model = self.load_model()
        else:
            vocabulary = []
            for doc in X:
                lemmatized_doc = self.preprocess.lemmatize(doc)
                vocabulary.append(lemmatized_doc)

            documents = []
            for i, doc in enumerate(vocabulary):
                tagged_doc = TaggedDocument(doc, [i])
                documents.append(tagged_doc)

            model = Doc2Vec(documents,
                            vector_size=self.vector_size,
                            window=self.window,
                            workers=self.workers)
            self.save_model(model)
            self.model = model

        return self

    def transform(self, X):
        embeddings = []
        for doc in X:
            lemmatized_doc = self.preprocess.lemmatize(doc)
            emb = self.model.infer_vector(lemmatized_doc)
            embeddings.append(emb)

        return embeddings


class MeanGloveTwitterVectorizer:

    def __init__(self, model_path):
        self.model_path = model_path
        if self.model_exists():
            self.glove_vectors = self.load_model()
        else:
            self.glove_vectors = downloader.load("glove-twitter-25")
            self.glove_vectors.save(self.model_path)

        self.preprocess = Preprocess()

    def load_model(self):
        return KeyedVectors.load(self.model_path)

    def save_model(self, model):
        if model is not None:
            model.save(self.model_path)
        else:
            raise Exception("Model has not been trained to be saved")

    def model_exists(self):
        path = pathlib.Path(self.model_path)
        if path.exists():
            return True
        return False

    def fit(self, X, y):
        return self

    def transform(self, X):
        embeddings = []
        for doc in X:
            doc_embeddings = []

            tokenized_doc = self.preprocess.tokenize(doc, keep_stopwords=True)
            lower_tokenized_doc = self.preprocess.lowercase(tokenized_doc)

            for token in lower_tokenized_doc:
                try:
                    embedding = self.glove_vectors.wv[token]
                    doc_embeddings.append(embedding)
                except KeyError as err:
                    continue

            if doc_embeddings:
                doc_embedding = np.mean(doc_embeddings, axis=0)
            else:
                doc_embedding = np.zeros(25)
            embeddings.append(doc_embedding)

        return embeddings


class CustomWord2VecVectorizer:

    def __init__(self, size, model_path, window=5, workers=4):
        self.workers = workers
        self.window = window
        self.size = size
        self.model_path = model_path
        self.preprocess = Preprocess()

    def load_model(self):
        return Word2Vec.load(self.model_path)

    def save_model(self, model):
        if model is not None:
            model.save(self.model_path)
        else:
            raise Exception("Model has not been trained to be saved")

    def model_exists(self):
        path = pathlib.Path(self.model_path)
        if path.exists():
            return True
        return False

    def fit(self, X, y):
        if self.model_exists():
            self.model = self.load_model()
        else:
            tokenized_docs_train = []
            for doc in X:
                tokenized_doc = self.preprocess.tokenize(doc,
                                                         keep_stopwords=True)
                lower_tokenized_doc = self.preprocess.lowercase(tokenized_doc)
                tokenized_docs_train.append(lower_tokenized_doc)

            model = Word2Vec(tokenized_docs_train,
                             size=self.size,
                             window=self.window,
                             worker=self.workers)

            self.save_model(model)
            self.model = model

        return self

    def transform(self, X):
        embeddings = []
        for doc in X:
            doc_embeddings = []

            tokenized_doc = self.preprocess.tokenize(doc, keep_stopwords=True)
            lower_tokenized_doc = self.preprocess.lowercase(tokenized_doc)

            for token in lower_tokenized_doc:
                try:
                    embedding = self.model.wv[token]
                    doc_embeddings.append(embedding)
                except KeyError as err:
                    continue

            if doc_embeddings:
                doc_embedding = np.mean(doc_embeddings, axis=0)
            else:
                doc_embedding = np.zeros(self.size)
            embeddings.append(doc_embedding)

        return embeddings
