import pickle
import pathlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from dataset import DatasetBuilder
from models import TextClassification
from preprocess import Preprocess


class Experiment:
    BASE_DIR = "findings"
    BEST_CLF_ID = "best-clf"

    def __init__(self, clf, directory, param_grid):
        self.clf = clf
        self.param_grid = param_grid
        self.dir = directory
        self.vectorizers = [TfidfVectorizer(),
                            TfidfVectorizer(tokenizer=Preprocess().tokenize_stem),
                            TfidfVectorizer(tokenizer=Preprocess().lemmatize)]
        self.experiments_id = ['tf-idf-default', 'tf-idf-stem', 'tf-idf-lemma']

    def find_best_clf(self, x_train, y_train):
        text_clf = TextClassification(TfidfVectorizer(), self.clf)
        best_clf = text_clf.best_estimator(x_train, y_train, self.param_grid)
        return best_clf

    def save(self, obj, filename):
        path = pathlib.Path(self.BASE_DIR, self.dir, filename)
        path.mkdir(parents=True, exist_ok=True)

        with open("{}.pkl".format(str(path)), 'wb') as f:
            pickle.dump(obj, f)

    def load(self, filename):
        path = pathlib.Path(self.BASE_DIR, self.dir, filename)

        with open("{}.pkl".format(str(path)), 'rb') as f:
            obj = pickle.load(f)
        return obj

    def pickle_exists(self, filename):
        path = pathlib.Path(self.BASE_DIR, self.dir, filename)

        if path.exists():
            return True
        return False

    @staticmethod
    def data(url):
        data_fields = ["reviewText"]
        target_field = "overall"
        dataset_builder = DatasetBuilder()
        X, y = dataset_builder.get_data(url,
                                        data_fields,
                                        target_field,
                                        drop_duplicates=True,
                                        return_X_y=True)
        return X, y


if __name__ == '__main__':
    url = 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Software_5.json.gz'

    param_grid = {
        'linearsvc__loss': ['hinge', 'squared_hinge']
    }
    experiment = Experiment(LinearSVC(), "SVM", param_grid)

    X, y = experiment.data(url)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    if experiment.pickle_exists(experiment.BEST_CLF_ID):
        best_clf = experiment.load(experiment.BEST_CLF_ID)
    else:

        best_clf = experiment.find_best_clf(X_train, y_train)
        experiment.save(best_clf, experiment.BEST_CLF_ID)

    for obj in zip(experiment.vectorizers, experiment.experiments_id):
        vectorizer, experiment_id = obj

        print("For experiment using: {}".format(vectorizer.__repr__()))
        if experiment.pickle_exists(experiment_id):
            clf_task = experiment.load(experiment_id)
        else:
            clf_task = TextClassification(vectorizer=vectorizer, classifier=best_clf)
            clf_task.fit(X_train, y_train)
            experiment.save(clf_task, experiment_id)

        y_predicted = clf_task.predict(X_test)
        print("MEA:", clf_task.mean_absolute_error(y_test, y_predicted))
        print("Saving classification metrics")
        clf_task.classification_report(y_test,
                                       y_predicted,
                                       filename="{}/clf-metrics-{}".format(experiment.BASE_DIR, experiment_id),
                                       save=True)




