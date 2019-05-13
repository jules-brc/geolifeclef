import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances
# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class NearestCentroidModel(Classifier):

    """Simple classifier that represents each class by the centroid of its
       members. Query example are classified to the class with the nearest
       centroid.

       It has no parameters to choose, making it a good baseline classifier.
       It does, however, suffer on non-convex classes, as well as when classes
       have drastically different variances, as equal variance in all dimensions
       is assumed.

       A parameter allows to implement the nearest shrunken centroid classifier :
       in effect, the value of each feature for each centroid is divided by the
       within-class variance of that feature. The feature values are then reduced
       a threshold. Most notably, if a particular feature value crosses zero,
       it is set to zero.
       In effect, this removes the feature from affecting the classification.
       This is useful, for example, for removing noisy features.
       A small shrink threshold (for example 0.2) may increase the accuracy.

       When applied to text classification using tf-idf vectors to represent
       documents, the nearest centroid classifier is known as the Rocchio
       classifier.
    """
    def __init__(self, metric='euclidean', shrink_threshold=None, ranking_size=30):
        """
          :param metric:
           The metric to use when calculating distance between instances.
           The default metric is Euclidean. Choices are:
            - 'euclidean' for standard Euclidean distance
            - 'manhattan': for the Manhattan distance
            - 'haversine' for distances between (latitude,longitude) points only
            - 'cosine': for cosinus similarity
           :param shrink_thresold:
            The threshold for shrinking centroids to remove features
        """
        self.metric = metric
        self.shrink_threshold = shrink_threshold
        self.ranking_size = ranking_size
        self.clf = NearestCentroid(metric=metric,
                                   shrink_threshold=shrink_threshold)

    def predict(self, X, return_proba=False, clf_predict=False, *args, **kwargs):
        """Predict the list of labels most likely to be observed
           for the data points given
        """
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        # compute all distances, in parallel if possible
        all_distances = pairwise_distances(X, self.clf.centroids_,
                                           metric=self.metric, n_jobs=-1)

        # get index of the sorted centroids' distances
        all_argsorts = np.argsort(all_distances, axis=1)
        # selecting closests classes centroids

        y_predicted = [self.classes_[argsort][:self.ranking_size] for argsort in all_argsorts]
        if return_proba:
            y_predicted_probas = list()
            for distance,argsort in zip(all_distances,all_argsorts):
                # predicting probabilities: inverse of the distance, normalized
                inverse_distances = (distance[argsort])**(-1)
                y_predicted_probas.append((inverse_distances / np.sum(inverse_distances))[:self.ranking_size])

            return np.array(y_predicted), np.array(y_predicted_probas)
        return np.array(y_predicted)

if __name__ == '__main__':

    from evaluate import model_selection_pipeline, generate_challenge_run
    # from sklearn.utils.estimator_checks import check_estimator
    # check_estimator(RandomForest)
    from scipy.stats import randint as sp_randint, uniform as sp_uniform

    dataset = '../data/pl_trusted_size1_noclc_scaled_pca.csv'

    init_param = dict()
    param_grid = {}

    results_file = 'experiments/nearest_centroid_model.txt'
    model_file = 'experiments/nearest_centroid_model.pkl'

    model_selection_pipeline(dataset, NearestCentroidModel, init_param,
                             param_grid, results_file=results_file)

    # Top30 score:0.154
    # MRR score:0.022959270415017052
    # Params: {'metric': 'euclidean', 'ranking_size': 30, 'shrink_threshold': 0.9}
