import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from embeddings import Doc2VecVectorizer
from models import TextClassification
from experiment import Experiment
from resample import Oversampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment file for SVM")
    parser.add_argument('url', type=str, help="A dataset URL from ")
    args = parser.parse_args()
    url = args.url

    X, y = Experiment.load_data(url, dest_dir="../data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train_res, y_train_res = Oversampler().resample(X_train, y_train)

    doc_vectorizer = Doc2VecVectorizer(vector_size=100,
                                       model_path="../data/software.doc2vec")
    text_clf = TextClassification(doc_vectorizer, LogisticRegression(solver='saga', C=1.5))
    text_clf.fit(X_train_res, y_train_res)
    y_predicted = text_clf.predict(X_test)
    text_clf.classification_report(y_test,
                                   y_predicted,
                                   filename="../findings/DOC2VEC/doc2vec-100.csv",
                                   save=True)
    print("MEA: {:.5f}".format(text_clf.mean_absolute_error(y_test, y_predicted)))
