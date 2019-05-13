import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

from sklearn.ensemble import ExtraTreesClassifier

class ExtraTreesModel(Classifier):

    """Extra trees classifier.
       This classifier is very similar to a random forest; in extremely
       randomized trees, randomness goes one step further in the way splits are
       computed. As in random forests, a random subset of candidate features is
       used, but instead of looking for the most discriminative thresholds,
       thresholds are drawn at random for each candidate feature and the best of
       these randomly-generated thresholds is picked as the splitting rule.

       This usually allows to reduce the variance of the model a bit more, at
       the expense of a slightly greater increase in bias:

       A VARIANT TO THIS MODEL THAT MAY BE INTERESTING:

       Our data has a lot of overlapping data points of different labels: we
       could try to fit a random forest where each tree is fitted on a random
       subsample with the constraint that is does not contain any overlapping
       points. Maybe it will be more discriminative ?
    """
    def __init__(self, n_estimators=100, criterion='gini',max_depth=None,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 max_features='auto', min_impurity_decrease=0.0,
                 bootstrap=True, verbose=0,
                 warm_start=False, ranking_size=30):
        """
           :param n_estimators:
           The number of trees in the forest.

           :param criterion:
           The function to measure the quality of a split. Supported criteria
           are 'gini' for the Gini impurity and 'entropy' for the information
           gain. Note: this parameter is tree-specific.

           :param max_depth:
           The maximum depth of the tree. If None, then nodes are expanded until
           all leaves are pure or until all leaves contain less than
           min_samples_split samples.

           :param min_samples_split:
           The minimum number of samples required to split an internal node:
           - if int, then consider min_samples_split as the minimum number.
           - if float, then min_samples_split is a fraction and
           ceil(min_samples_split * n_samples) are the minimum number of samples
           for each split.

           :param min_weight_fraction_leaf:
           The minimum weighted fraction of the sum total of weights
           (of all the input samples) required to be at a leaf node. Samples
           have equal weight when sample_weight is not provided.

           :param max_features:
           The number of features to consider when looking for the best split:
           - if int, then consider max_features features at each split.
           - if float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
           - if 'auto', then max_features=sqrt(n_features).
           - if None, then max_features=n_features.

           :param min_impurity_decrease:
           A node will be split if this split induces a decrease of the
           impurity greater than or equal to this value.

           :param bootstrap:
           Whether bootstrap samples are used when building trees. If False,
           the whole datset is used to build each tree.

           :param warm_start:
           When set to True, reuse the solution of the previous call to fit and
           add more estimators to the ensemble, otherwise, just fit a whole
           new forest.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.warm_start = warm_start
        # the Scikit-learn random forest classifier
        self.clf = ExtraTreesClassifier(n_estimators=n_estimators,
                                          criterion=criterion,
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          min_weight_fraction_leaf=min_weight_fraction_leaf,
                                          max_features=max_features,
                                          min_impurity_decrease=min_impurity_decrease,
                                          bootstrap=bootstrap,
                                          verbose=verbose,
                                          warm_start=warm_start,
                                          n_jobs=-1)
        self.ranking_size = ranking_size

    def fit(self, X, y, sample_weight=None):

        super().fit(X, y, sample_weight=sample_weight)

        # importance of features with respect to the predictability of the
        # class
        self.feature_importances_ = self.clf.feature_importances_
        return self

if __name__ == '__main__':

    from evaluate import model_selection_pipeline, generate_challenge_run
    # from sklearn.utils.estimator_checks import check_estimator
    # check_estimator(RandomForest)
    from scipy.stats import randint as sp_randint, uniform as sp_uniform
    dataset = '../data/pl_trusted_size1_noclc_scaled_pca.csv'

    init_param = dict(n_estimators=250, max_depth=3,bootstrap=False)

    param_grid = {'n_estimators': sp_randint(50,500),
                  'criterion': ['gini', 'entropy'],
                  'max_depth': sp_randint(2, 15),
                  'min_samples_split': sp_randint(2,20),
                  'min_samples_leaf': sp_randint(1,20),
                  'max_features': sp_uniform(0.2, 0.8), # range [0.2, 1.]
                  'bootstrap': [False, True]
                  }
    results_file = 'experiments/extra_trees_model.txt'
    model_file = 'experiments/extra_trees_model.pkl'

    model_selection_pipeline(dataset, ExtraTreesModel, param_grid, init_param,
                             results_file=results_file, model_file=model_file)

    # Top30 score:0.365
    # MRR score:0.08798763950372647
    # Params: {'bootstrap': False, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto', 'min_impurity_decrease': 0.0, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 150, 'ranking_size': 30, 'verbose': None, 'warm_start': False}

    # Feature importances sorted with names:
    # clc               0.053894
    # chbio_1           0.045723
    # chbio_11          0.045049
    # chbio_10          0.044953
    # chbio_6           0.043206
    # alti              0.041379
    # Longitude         0.040775
    # chbio_9           0.038167
    # chbio_15          0.037659
    # Latitude          0.035600
    # chbio_8           0.034558
    # etp               0.033587
    # chbio_18          0.032696
    # chbio_5           0.031593
    # chbio_2           0.031310
    # chbio_3           0.031235
    # chbio_14          0.030357
    # chbio_17          0.029172
    # chbio_7           0.028729
    # bs_top            0.025511
    # chbio_4           0.024844
    # proxi_eau_fast    0.024576
    # awc_top           0.020831
    # chbio_12          0.020798
    # erodi             0.020437
    # dimp              0.018889
    # chbio_16          0.018190
    # chbio_13          0.018049
    # crusting          0.017318
    # text              0.017012
    # chbio_19          0.015668
    # dgh               0.015653
    # pd_top            0.012679
    # oc_top            0.010112
    # cec_top           0.009792
