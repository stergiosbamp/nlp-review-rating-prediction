import numpy as np
import pathlib

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec, KeyedVectors
from gensim import downloader


from extract import Extractor
from models import TextClassification
from preprocess import Preprocess


def load_data():
    filename = "data/Software_5.json"
    data_fields = ["reviewText"]
    target_field = "overall"
    X, y = Extractor.extract_examples(filename,
                                    data_fields,
                                    target_field,
                                    drop_duplicates=True,
                                    return_X_y=True)
    return X, y


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


class MeanGloveVectorizer:
    def __init__(self, glove_filename):
        with open(glove_filename, "rb") as lines:
            glove = {
                line.split()[0]: np.array(map(float,
                                              line.split()[1:]))
                for line in lines
            }

        self.glove = glove
        self.length = int(glove_filename.split(".")[-2].split("d")[0])
        self.preprocess = Preprocess()

    def fit(self, X, y):
        print("fitting...")
        return self

    def transform(self, X):
        vocabulary = []
        for doc in X:
            lemmatized_doc = self.preprocess.lemmatize(doc)
            vocabulary.append(lemmatized_doc)

        embeddings = np.array([
            np.mean([
                self.glove[w] if w in self.glove else np.zeros(self.length)
                for w in words
            ],
                    axis=0)
            for words in vocabulary
        ])

        return embeddings


if __name__ == '__main__':

    # Read dataset
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)

    classification_task = TextClassification(Doc2VecVectorizer(vector_size=20,
                                                               window=2,
                                                               workers=4,
                                                               model_path='software.doc2vec'),
                                             LogisticRegression())
    classification_task.fit(X_train, y_train)
    # TODO
    # Also save the classification task apart from the embeddings model
    y_predicted = classification_task.predict(X_test)

    print(classification_task.classification_report(y_test, y_predicted))
    exit()

    vocabulary = []
    preprocess = Preprocess()

    # Download glove twitter embeddings and save it for later use
    # glove_vectors = downloader.load("glove-twitter-25")
    # glove_vectors.save("glove-twitter-25.kv")

    print("loading...")
    glove_embeddings = KeyedVectors.load("glove-twitter-25.kv")
    print("loaded model")

    train_embeddings = []
    for doc in X_train:
        doc_embeddings = []

        tokenized_doc = preprocess.tokenize(doc, keep_stopwords=True)
        lower_tokenized_doc = [token.lower() for token in tokenized_doc]

        for token in lower_tokenized_doc:
            try:
                emb = glove_embeddings.wv[token]
                doc_embeddings.append(emb)
            except KeyError as err:
                continue

        if doc_embeddings:
            doc_embedding = np.mean(doc_embeddings, axis=0)
        else:
            doc_embedding = np.zeros(25)
        train_embeddings.append(doc_embedding)

    print("Ended training embeddings")
    test_embeddings = []
    for doc in X_test:
        doc_embeddings = []

        tokenized_doc = preprocess.tokenize(doc, keep_stopwords=True)
        lower_tokenized_doc = [token.lower() for token in tokenized_doc]

        for token in lower_tokenized_doc:
            try:
                emb = glove_embeddings.wv[token]
                doc_embeddings.append(emb)
            except KeyError as err:
                continue

        if doc_embeddings:
            doc_embedding = np.mean(doc_embeddings, axis=0)
        else:
            doc_embedding = np.zeros(25)
        test_embeddings.append(doc_embedding)

    print("Ended testing embeddings")
    clf = RandomForestClassifier()
    clf.fit(train_embeddings, y_train)
    y_pred = clf.predict(test_embeddings)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"F1-score: {metrics.f1_score(y_test, y_pred, average='macro')}")
