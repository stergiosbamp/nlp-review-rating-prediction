import argparse

from experiment import NaiveBayesExperiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment file for SVM")
    parser.add_argument('url', type=str, help="A dataset URL from ")
    args = parser.parse_args()
    url = args.url

    experiment = NaiveBayesExperiment()

    X, y = experiment.load_data(url, dest_dir="../data")

    experiment.run(X, y)
