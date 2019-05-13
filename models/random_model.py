import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class RandomModel(Classifier):

    """Random model returning a random list of labels from the training set
       for a test occurence.
       This stupid model is just used as a baseline.
    """
    def __init__(self, ranking_size=30):
        """
        """
        self.ranking_size = ranking_size

    def fit(self, X, y):

        super().fit(X, y)

        y_unique = np.unique(y)
        self.y_predicted_probas_ = (np.ones(len(self.classes_))/len(self.classes_))[:self.ranking_size]

        return self

    def predict(self, X, return_proba=False):

        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)

        y_predicted = [np.random.choice(self.classes_, size=self.ranking_size, replace=False) for i in range(len(X))]
        if return_proba:
            return np.array(y_predicted), np.tile(self.y_predicted_probas_,(len(X),1))

        return np.array(y_predicted)

if __name__ == '__main__':

    from evaluate import model_selection_pipeline, generate_challenge_run
    # from sklearn.utils.estimator_checks import check_estimator
    # check_estimator(RandomForest)
    from scipy.stats import randint as sp_randint, uniform as sp_uniform

    dataset = '../data/pl_trusted_size1_noclc_scaled_pca.csv'

    init_param = dict()
    param_grid = {}

    results_file = 'experiments/random_model.txt'
    model_file = 'experiments/random_model.pkl'

    model_selection_pipeline(dataset, RandomModel, init_param,
                             param_grid, results_file=results_file)

    # Top30 score:0.046
    # MRR score:0.006038416041714787
    # Params: {'ranking_size': 30}


