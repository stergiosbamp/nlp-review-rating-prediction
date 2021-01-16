import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

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

    vectorizer = TfidfVectorizer()

    classifiers = [
        (LinearSVC(loss='hinge'), 'svm'),
        (MultinomialNB(alpha=0.1), 'naive-bayes'),
        (LogisticRegression(solver='saga', C=1.5), 'logistic')
    ]

    for clf, file_to_save in classifiers:
        print("For classifier: {}".format(clf))
        text_clf = TextClassification(vectorizer, clf)
        text_clf.fit(X_train_res, y_train_res)
        y_predicted = text_clf.predict(X_test)
        text_clf.classification_report(y_test,
                                       y_predicted,
                                       filename="../findings/BALANCED/results-{}.csv".format(file_to_save),
                                       save=True)
        print("MEA: {:.5f}".format(text_clf.mean_absolute_error(y_test, y_predicted)))
