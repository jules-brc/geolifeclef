import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighborsModel(Classifier):

    """Classifier predicting the labels by counting label occurrences
       among the k-nearest neighbors of a query example.

       Despite their simplicity, nearest neighbors approache are successful in
       a large number of classification problems.
       The optimal choice of the value  is highly data-dependent: in general
       a larger  suppresses the effects of noise, but makes the classification
       boundaries less distinct.

       The basic nearest neighbors classification uses uniform weights. But
       under some circumstances, it is better to weight the neighbors
       such that nearer neighbors contribute more to the fit.

       VARIANTS TO THIS MODEL THAT MAY GIVE INTERESTING RESULTS:

       - weights to the neighbors are the frequency of their label in the whole
         training set

       - filtering neighbors by environment, then distances in the environmental space

       - use of co-occurences? Co-occurences matrix of species in a s-size location window?
         => revealed groups/patterns of species aggregates over space/environment
         => k-means over groups of species (event. filtered in space or environment), both hard and soft (GMM?)
         => logistic regression, or svm in this new representation space (hybrid clustering/classif)
    """
    def __init__(self, n_neighbors=5, weights='uniform', p=2, metric='minkowski', ranking_size=30):
        """
           :param n_neighbors:
           Number of neighbors to use by default

           :param weights:
           The weight function used in prediction.
           Possible values:
            - 'uniform' : uniform weights. All points in each neighborhood are
               weighted equally.
            - 'distance' : weight points by the inverse of their distance.
               In this case, closer neighbors of a query point will have a greater
               influence than neighbors which are further away.
            - [callable] : a user-defined function which accepts an array of
               distances, and returns an array of the same shape containing the
               weights.

           :param p:
           Power parameter for the Minkowski metric
           :param metric:
           The distance metric to use for the tree. The default metric is
           Minkowski, and with p=2 is equivalent to the standard Euclidean
           metric. Choices are:
            - 'euclidean' for standard Euclidean distance
            - 'manhattan': for the Manhattan distance
            - 'haversine' for distances between (latitude,longitude) points only
            - 'cosine': for cosinus similarity
            - 'minkowski': the Minkowski distance (euclidean if p=2)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        # the Scikit-learn K neighbors classifier
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                                            metric=metric,
                                            weights=weights,
                                            n_jobs=-1)
        self.ranking_size = ranking_size



if __name__ == '__main__':

    from evaluate import model_selection_pipeline, generate_challenge_run
    # from sklearn.utils.estimator_checks import check_estimator
    # check_estimator(RandomForest)
    from scipy.stats import randint as sp_randint, uniform as sp_uniform

    dataset = '../data/pl_trusted_size1_noclc_scaled_pca.csv'

    init_param = dict(metric='euclidean', n_neighbors=400, weights='distance')

    param_grid = {'n_neighbors': sp_randint(50, 401),
                  'weights':['uniform', 'distance'],
                  'p': sp_randint(1,4),
                  'metric':['minkowski', 'euclidean', 'cosine']
                  }

    results_file = 'experiments/knn_model.txt'
    model_file = 'experiments/knn_model.pkl'

    model_selection_pipeline(dataset, KNearestNeighborsModel, init_param, param_grid)

    # Test KNN model
    # Top30 score:0.3939759036144579
    # MRR score:0.08739564147437189
    # Params: {'lmnn': None, 'metric': 'euclidean', 'n_neighbors': 400, 'p': None, 'ranking_size': 30, 'weights': 'distance'}





    # Improvements/experiments

    # Small number of neighbors give bad results. Needs a lot.
    # The euclidean metric and cosine metric seems to give almost identical results,
    # remains to be confirmed.


    # Using Large-margin nearest neighbor metric learning
    # keep k-nearest neighbors in the same class, while keeping examples from
    # different classes separated by a large margin. This algorithm makes no
    # assumptions about the distribution of the data.

    # USELESS: it does not significatively improve results and it is
    # absolutely not scalable (on 5k samples took more than a minute to
    # optimize)
