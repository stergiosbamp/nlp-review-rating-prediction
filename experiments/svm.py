import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from experiment import SVMExperiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment file for SVM")
    parser.add_argument('url', type=str, help="A dataset URL from ")
    args = parser.parse_args()
    url = args.url

    experiment = SVMExperiment()

    X, y = experiment.load_data(url, dest_dir="../data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    experiment.run(X, y)
