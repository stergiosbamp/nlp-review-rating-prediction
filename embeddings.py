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


if __name__ == '__main__':

    # Read dataset
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)

    clf = make_pipeline(Doc2VecVectorizer(), LogisticRegression())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"F1-score: {metrics.f1_score(y_test, y_pred, average='macro')}")

    # for doc in X_test:


    # clf_task = TextClassification(TfidfVectorizer(), LogisticRegression())
