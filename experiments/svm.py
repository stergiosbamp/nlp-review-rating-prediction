import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from models import TextClassification
from extract import Extractor
from preprocess import Preprocess


def load_data():
    data_fields = ["reviewText"]
    target_field = "overall"

    df = Extractor.extract_examples("../data/Software_5.json", data_fields=data_fields, target_field=target_field)
    df.drop_duplicates(inplace=True)
    X = df.reviewText
    y = df.overall
    return X, y


def find_best_clf_parameters():
    """
    This method returns the best hyper parameters for the classifier.
    By convention we use the default TfIdfVectorizer to search for the best classifier in the pipeline.
    """

    vectorizer = TfidfVectorizer()
    classifier = LinearSVC()

    task = TextClassification(vectorizer, classifier)
    param_grid = {
        'linearsvc__loss': ['hinge', 'squared_hinge']
    }
    best_params = task.best_hyperparameters(X_train, y_train, param_grid)
    loss = best_params['linearsvc__loss']

    return loss


if __name__ == '__main__':
    BASE_DIR = '../findings/svm/'

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print("Finding best hyper parameters for LinearSVC...")
    if not os.path.exists("{}/best-clf.pkl".format(BASE_DIR)):
        best_loss = find_best_clf_parameters()
        best_classifier = LinearSVC(loss=best_loss)
        with open("{}/best-clf.pkl".format(BASE_DIR), 'wb') as f:
            pickle.dump(best_classifier, f)
    else:
        with open("{}/best-clf.pkl".format(BASE_DIR), 'rb') as f:
            best_classifier = pickle.load(f)

    experiments = [TfidfVectorizer(),
                   TfidfVectorizer(tokenizer=Preprocess().tokenize_stem),
                   TfidfVectorizer(tokenizer=Preprocess().lemmatize)]
    experiments_id = ['tf-idf-default', 'tf-idf-stem', 'tf-idf-lemma']

    for obj in zip(experiments, experiments_id):
        experiment_vec, experiment_id = obj

        print("For experiment using: {}".format(experiment_vec.__repr__()))
        if not os.path.exists("{}/{}.pkl".format(BASE_DIR, experiment_id)):
            classification_task = TextClassification(experiment_vec, best_classifier)
            classification_task.fit(X_train, y_train)
            with open("{}/{}.pkl".format(BASE_DIR, experiment_id), 'wb') as f:
                pickle.dump(classification_task, f)
        else:
            with open("{}/{}.pkl".format(BASE_DIR, experiment_id), 'rb') as f:
                classification_task = pickle.load(f)

        y_predicted = classification_task.predict(X_test)

        print("Saving classification results...")

        classification_task.classification_report(y_test,
                                                  y_predicted,
                                                  filename="{}/classification-report-{}".format(BASE_DIR, experiment_id),
                                                  save=True)

