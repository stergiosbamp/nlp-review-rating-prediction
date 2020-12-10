import pandas as pd
import pathlib

from sklearn import metrics

from sklearn.pipeline import make_pipeline


class TextClassification:

    def __init__(self, vectorizer, classifier):
        self.pipeline = make_pipeline(vectorizer, classifier)

    def fit(self, x_train, y_train):
        self.pipeline.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        y_predicted = self.pipeline.predict(x_test)
        return y_predicted

    @staticmethod
    def accuracy(y_test, y_pred):
        return metrics.accuracy_score(y_test, y_pred)

    @staticmethod
    def f1_score(y_test, y_pred, average='weighted'):
        return metrics.f1_score(y_test, y_pred, average=average)

    @staticmethod
    def mean_absolute_error(y_test, y_pred):
        return metrics.mean_absolute_error(y_test, y_pred)

    @staticmethod
    def classification_report(y_test, y_pred, filename=None, save=False):
        if save:
            report = metrics.classification_report(y_test, y_pred, output_dict=True)

            if filename is None:
                raise Exception("Cannot perform save with empty filename")

            file = pathlib.Path(filename)
            if file.suffix == '':
                file = str(file) + ".csv"

            df = pd.DataFrame.from_dict(report)
            df.to_csv(file)
        else:
            return metrics.classification_report(y_test, y_pred)
