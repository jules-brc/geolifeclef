import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class FrequenceModel(Classifier):

    """Frequence model returning always the most frequent labels in the
       training set.
       This is a baseline to compare with more elaborate models. If a model
       performs less that this one, we can conclude that it did not learn
       anything from the data.
    """
    def __init__(self, ranking_size=30):
        """
        """
        self.ranking_size = ranking_size

    def fit(self, X, y):

        super().fit(X, y)

        y_unique, counts = np.unique(y, return_counts=True)

        # probabilities of the labels
        probas = counts/len(y)
        # get the indexes of the sorted probabilities, in decreasing order
        top_predictions = np.flip(np.argsort(probas)[-self.ranking_size:],axis=0)
        # get most frequent labels
        self.y_predicted_ = y_unique[top_predictions]
        # get as well the probabilities of the predictions
        self.y_predicted_probas_ = probas[top_predictions]

        return self

    def predict(self, X, return_proba=False):

        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        y_predicted = np.tile(self.y_predicted_, (len(X),1))
        if return_proba:
            y_predicted_probas = np.tile(self.y_predicted_probas_, (len(X),1))
            return y_predicted, y_predicted_probas

        return y_predicted

if __name__ == '__main__':

    from evaluate import model_selection_pipeline, generate_challenge_run
    # from sklearn.utils.estimator_checks import check_estimator
    # check_estimator(RandomForest)
    from scipy.stats import randint as sp_randint, uniform as sp_uniform

    dataset = '../data/pl_trusted_size1_noclc_scaled_pca.csv'

    init_param = dict()
    param_grid = {}

    results_file = 'experiments/frequence_model.txt'
    model_file = 'experiments/frequence_model.pkl'

    model_selection_pipeline(dataset, FrequenceModel, init_param,
                             param_grid, results_file=results_file)

    # Top30 score:0.297
    # MRR score:0.06470175515004985
    # Params: {'ranking_size': 30}

    # Maybe the data contains a lot of common species?
    # Maybe it follows Zipf law? Try to plot number of species considered/percent
    # of species in the dataset
