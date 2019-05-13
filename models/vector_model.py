
import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import pairwise_distances

class VectorModel(Classifier):

    """Simple nearest neighbors classifier which predict labels from the
       neighbors ordered by increasing distance from a query example. Returns
       the nearest-neighbor label, then the next neighbor label, and so on..
       with the constraint that a neighbor's label is chosen only if it hasn't
       been added so far to the ranking.

       This model may be prone to overfitting because it predict the next neighbor
       label in a greedy fashion, not taking into account a neighborhood of
       several examples. This is why a k-nearest neighbor classifier may give
       better results (in practice, it does).

       The vector model does not support probabilities predictions but only
       labels predictions.
    """
    def __init__(self, metric='euclidean', ranking_size=30):
        """
        :param metric: the distance metric used, should be something like
            - 'euclidean' for regular euclidean distance
            - 'manhattan'
            - 'haversine' for distances between (latitude,longitude) points only
            - 'cosine'
        """
        self.metric = metric
        self.ranking_size = ranking_size

    def predict(self, X):

        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        # compute all distances, in parallel if possible
        all_distances = pairwise_distances(X, self.X_,
                                           metric=self.metric, n_jobs=-1)
        # get index of the sorted points' distances
        all_argsorts = np.argsort(all_distances, axis=1)
        y_predicted = list()
        # selecting closests points distinct labels
        for argsort in all_argsorts:

            # get closests labels
            y_closest = self.y_[argsort]
            # predict distinct labels
            y_found = set()
            y_pred = list()
            for y in y_closest:
                if len(y_pred)>= self.ranking_size:
                    break
                if y not in y_found:
                  y_pred.append(y)
                  y_found.add(y)

            y_predicted.append(y_pred)

        return np.array(y_predicted)

if __name__ == '__main__':

    from evaluate import model_selection_pipeline, generate_challenge_run
    # from sklearn.utils.estimator_checks import check_estimator
    # check_estimator(RandomForest)
    from scipy.stats import randint as sp_randint, uniform as sp_uniform

    dataset = '../data/pl_trusted_size1_noclc_scaled_pca.csv'

    init_param = dict(metric='euclidean')

    param_grid = {'metric':['euclidean', 'cosine']}

    results_file = 'experiments/vector_model.txt'
    model_file = 'experiments/vector_model.pkl'

    model_selection_pipeline(dataset, VectorModel, init_param, param_grid,
                             n_iter_search=10,
                             results_file=results_file, model_file=model_file)

    # FAIL! PROGRAMS CRASHES WHEN IT IS TRAINED ON THE ALL DATA (230K OCC)

    # Test vector model
    # Top30 score:0.246
    # MRR score:0.05718168788586186
    # Params: {'metric': 'euclidean', 'ranking_size': 30}

    # Test vector model
    # Top30 score:0.23800000000000002
    # MRR score:0.0586088829636054
    # Params: {'metric': 'cosine', 'ranking_size': 30}
