import pickle
import pathlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from dataset import DatasetBuilder
from models import TextClassification
from preprocess import Preprocess


class Experiment:
    BASE_DIR = "../findings"
    BEST_CLF_ID = "best-clf"

    def __init__(self, clf, directory, param_grid=None):
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
        path = pathlib.Path(self.BASE_DIR, self.dir)
        path.mkdir(parents=True, exist_ok=True)
        path = path.joinpath(filename)
        with open("{}.pkl".format(str(path)), 'wb') as f:
            pickle.dump(obj, f)

    def load(self, filename):
        path = pathlib.Path(self.BASE_DIR, self.dir)
        path = path.joinpath(filename)

        with open("{}.pkl".format(str(path)), 'rb') as f:
            obj = pickle.load(f)
        return obj

    def pickle_exists(self, filename):
        path = pathlib.Path(self.BASE_DIR, self.dir, filename)
        path = path.with_suffix(".pkl")
        if path.exists():
            return True
        return False

    @staticmethod
    def load_data(url, dest_dir="data/"):
        data_fields = ["reviewText"]
        target_field = "overall"
        dataset_builder = DatasetBuilder(dest_dir=dest_dir)
        X, y = dataset_builder.get_data(url,
                                        data_fields,
                                        target_field,
                                        drop_duplicates=True,
                                        return_X_y=True)
        return X, y

    def run(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # If param grid is None that means the client of experiment
        # doesn't want to perform GridSearchCV
        if self.param_grid is not None:
            if self.pickle_exists(self.BEST_CLF_ID):
                print("Loading best classifier from pickle file")
                best_clf = self.load(self.BEST_CLF_ID)
            else:
                best_clf = self.find_best_clf(X_train, y_train)
                self.save(best_clf, self.BEST_CLF_ID)
        else:
            best_clf = self.clf

        for obj in zip(self.vectorizers, self.experiments_id):
            vectorizer, experiment_id = obj

            print("For experiment using: {}".format(vectorizer.__repr__()))
            if self.pickle_exists(experiment_id):
                print("Loading classification task from pickle file")
                clf_task = self.load(experiment_id)
            else:
                clf_task = TextClassification(vectorizer=vectorizer, classifier=best_clf)
                clf_task.fit(X_train, y_train)
                self.save(clf_task, experiment_id)

            y_predicted = clf_task.predict(X_test)
            print("MEA:", clf_task.mean_absolute_error(y_test, y_predicted))
            print("Saving classification metrics")
            target_file = pathlib.Path(self.BASE_DIR, self.dir, "clf-metrics-{}".format(experiment_id))
            clf_task.classification_report(y_test,
                                           y_predicted,
                                           filename=target_file,
                                           save=True)


class SVMExperiment(Experiment):
    def __init__(self):
        param_grid = {
            'linearsvc__loss': ['hinge', 'squared_hinge']
        }
        super().__init__(LinearSVC(), "SVM", param_grid)


class NaiveBayesExperiment(Experiment):
    def __init__(self):
        param_grid = {
            'multinomialnb__alpha': [0.0, 0.1, 0.5, 1.0]
        }
        super().__init__(MultinomialNB(), "NAIVE-BAYES", param_grid)


class LogisticRegressionExperiment(Experiment):
    def __init__(self):
        param_grid = {
            'logisticregression__solver': ['saga', 'lbfgs'],
            'logisticregression__C': [0.3, 1.0, 1.5]
        }
        super().__init__(LogisticRegression(n_jobs=-1), 'LOGISTIC-REGRESSION', param_grid)
