import argparse

from embeddings import MeanGloveTwitterVectorizer
from models import TextClassification
from experiment import Experiment
from resample import Oversampler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def glove_twitter_50():
    mean_50_vectorizer = MeanGloveTwitterVectorizer(model_path='../data/glove-twitter-50.kv',
                                                    model_to_download='glove-twitter-50',
                                                    model_vector_size=50)

    text_clf = TextClassification(mean_50_vectorizer, LogisticRegression(solver='saga', C=1.5))

    return text_clf


def glove_twitter_100():
    mean_100_vectorizer = MeanGloveTwitterVectorizer(model_path='../data/glove-twitter-100.kv',
                                                     model_to_download='glove-twitter-100',
                                                     model_vector_size=100)

    text_clf = TextClassification(mean_100_vectorizer, LogisticRegression(solver='saga', C=1.5))

    return text_clf


def glove_twitter_200():
    mean_200_vectorizer = MeanGloveTwitterVectorizer(model_path='../data/glove-twitter-200.kv',
                                                     model_to_download='glove-twitter-200',
                                                     model_vector_size=200)

    text_clf = TextClassification(mean_200_vectorizer, LogisticRegression(solver='saga', C=1.5))

    return text_clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment file for SVM")
    parser.add_argument('url', type=str, help="A dataset URL from ")
    args = parser.parse_args()
    url = args.url

    X, y = Experiment.load_data(url, dest_dir="../data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train_res, y_train_res = Oversampler().resample(X_train, y_train)

    print("Examining Glove Twitter 50...")
    # Glove Twitter 50
    text_clf_twitter_50 = glove_twitter_50()

    text_clf_twitter_50.fit(X_train_res, y_train_res)
    y_predicted = text_clf_twitter_50.predict(X_test)
    text_clf_twitter_50.classification_report(y_test, y_predicted,
                                              filename="../findings/GLOVE-TWITTER/SIZE-50/glove-twitter-50-results.csv",
                                              save=True)
    print("MEA: {:.5f}".format(text_clf_twitter_50.mean_absolute_error(y_test, y_predicted)))

    print("Examining Glove Twitter 100...")
    # Glove Twitter 100
    text_clf_twitter_100 = glove_twitter_100()

    text_clf_twitter_100.fit(X_train_res, y_train_res)
    y_predicted = text_clf_twitter_100.predict(X_test)
    text_clf_twitter_100.classification_report(y_test, y_predicted,
                                               filename="../findings/GLOVE-TWITTER/SIZE-100/glove-twitter-100-results.csv",
                                               save=True)
    print("MEA: {:.5f}".format(text_clf_twitter_100.mean_absolute_error(y_test, y_predicted)))

    print("Examining Glove Twitter 200...")
    # Glove Twitter 200
    text_clf_twitter_200 = glove_twitter_200()

    text_clf_twitter_200.fit(X_train_res, y_train_res)
    y_predicted = text_clf_twitter_200.predict(X_test)
    text_clf_twitter_200.classification_report(y_test, y_predicted,
                                               filename="../findings/GLOVE-TWITTER/SIZE-200/glove-twitter-200-results.csv",
                                               save=True)
    print("MEA: {:.5f}".format(text_clf_twitter_200.mean_absolute_error(y_test, y_predicted)))
