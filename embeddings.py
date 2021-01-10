from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


if __name__ == '__main__':

    # Read dataset
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)

    vocabulary = []
    preprocess = Preprocess()

    """
    This is just once
    """
    for doc in X_train:
        lemmatized_doc = preprocess.lemmatize(doc)
        vocabulary.append(lemmatized_doc)

    # Create TaggedDocuments
    documents = []
    for i, doc in enumerate(vocabulary):
        tagged_doc = TaggedDocument(doc, [i])
        documents.append(tagged_doc)
    model = Doc2Vec(documents, vector_size=20, window=2, workers=4)
    model.save("my-doc2vec-model")

    model = Doc2Vec.load("my-doc2vec-model")

    train_embeddings = [embedding for embedding in model.docvecs.vectors_docs]

    test_embeddings = []
    for doc in X_test:
        lemmatized_doc = preprocess.lemmatize(doc)
        emb = model.infer_vector(lemmatized_doc)
        test_embeddings.append(emb)

    clf = LogisticRegression()
    clf.fit(train_embeddings, y_train)
    y_pred = clf.predict(test_embeddings)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"F1-score: {metrics.f1_score(y_test, y_pred, average='macro')}")

    # for doc in X_test:


    # clf_task = TextClassification(TfidfVectorizer(), LogisticRegression())
