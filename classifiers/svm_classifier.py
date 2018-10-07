from classifiers.standard_classifier import StandardClassifer
from sklearn.svm import SVC
from scipy.stats import expon
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

import scipy
import numpy as np

from global_constants import DatasetType


class SvmClassifier(StandardClassifer):
    def __init__(self, corpus):
        super().__init__(corpus=corpus)
        self.bestModel = None

    def train(self):
        # Determine optimal regularizer
        exponential_distribution = scipy.stats.expon(scale=100)
        all_regularizer_values = exponential_distribution.rvs(100).tolist()
        lesser_than_one = np.linspace(0.00001, 1.0, 11)
        all_regularizer_values.extend(lesser_than_one)
        all_regularizer_values.extend([10, 100, 1000, 10000])
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                             'C': all_regularizer_values},
                            {'kernel': ['linear'], 'C': all_regularizer_values}]
        # SVM
        svm = SVC(C=1.0)
        grid_search = GridSearchCV(estimator=svm, param_grid=tuned_parameters, cv=10,
                                   n_jobs=8, scoring=None, refit=True)
        grid_search.fit(X=self.corpus.tfIdfFeatures[DatasetType.Training],
                        y=self.corpus.labelsDict[DatasetType.Training])
        self.bestModel = grid_search.best_estimator_
        print("Best Parameters:")
        print(grid_search.best_params_)

    def test(self, dataset_type):
        if dataset_type == DatasetType.Training or dataset_type == DatasetType.Validation:
            mean_accuracy = self.bestModel.score(X=self.corpus.tfIdfFeatures[dataset_type],
                                                 y=self.corpus.labelsDict[dataset_type])
            print("Mean Accuracy on {0}={1}".format(dataset_type, mean_accuracy))




