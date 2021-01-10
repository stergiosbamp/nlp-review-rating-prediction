import numpy as np

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from extract import Extractor
from models import TextClassification
from preprocess import Preprocess


def load_data():
    filename = "data/AMAZON_FASHION_5.json"
    data_fields = ["reviewText"]
    target_field = "overall"
    X, y = Extractor.extract_examples(filename,
                                    data_fields,
                                    target_field,
                                    drop_duplicates=True,
                                    return_X_y=True)
    return X, y


class Doc2VecVectorizer:
    def __init__(self):
        self.preprocess = Preprocess()

    def fit(self, X, y):
        vocabulary = []
        for doc in X:
            lemmatized_doc = self.preprocess.lemmatize(doc)
            vocabulary.append(lemmatized_doc)

        documents = []
        for i, doc in enumerate(vocabulary):
            tagged_doc = TaggedDocument(doc, [i])
            documents.append(tagged_doc)

        model = Doc2Vec(documents, vector_size=20, window=2, workers=4)
        self.model = model

        return self

    def transform(self, X):
        embeddings = []
        for doc in X:
            lemmatized_doc = self.preprocess.lemmatize(doc)
            emb = self.model.infer_vector(lemmatized_doc)
            embeddings.append(emb)

        return embeddings


class MeanGloveVectorizer():

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

    # Doc2Vec
    clf = make_pipeline(Doc2VecVectorizer(), LogisticRegression())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"F1-score: {metrics.f1_score(y_test, y_pred, average='macro')}")

    # Glove
    clf_glove = make_pipeline(MeanGloveVectorizer("data/glove.6B.100d.txt"),
                              LogisticRegression())
    clf_glove.fit(X_train, y_train)
    y_pred_glove = clf_glove.predict(X_test)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred_glove)}")
    print(
        f"F1-score: {metrics.f1_score(y_test, y_pred_glove, average='macro')}")

    print(clf)
    print(clf_glove)

    # for doc in X_test:

    # clf_task = TextClassification(TfidfVectorizer(), LogisticRegression())
