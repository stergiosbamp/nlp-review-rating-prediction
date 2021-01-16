import numpy as np

from imblearn.over_sampling import RandomOverSampler


class Resampler:

    def __init__(self, resampler):
        self.resampler = resampler

    def resample(self, X, y):
        X_res, y_res = self.resampler.fit_resample(
            np.reshape(X.to_numpy(), (-1, 1)), y)

        docs = []
        for i in range(X_res.shape[0]):
            docs.append(X_res[i, 0])

        return docs, y_res


class Oversampler(Resampler):

    def __init__(self):
        super(Oversampler, self).__init__(RandomOverSampler(random_state=14))
