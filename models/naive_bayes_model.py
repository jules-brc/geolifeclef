
import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

from sklearn.naive_bayes import GaussianNB

class NaiveBayesModel(Classifier):

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
    def __init__(self, var_smoothing=1e-9, ranking_size=30):
        """
        """
        # the Scikit-learn naive Bayes classifier
        self.var_smoothing = var_smoothing

        self.clf = GaussianNB(var_smoothing=var_smoothing)
        self.ranking_size = ranking_size

if __name__ == '__main__':

    from evaluate import model_selection_pipeline, generate_challenge_run
    # from sklearn.utils.estimator_checks import check_estimator
    # check_estimator(RandomForest)
    from scipy.stats import randint as sp_randint, uniform as sp_uniform

    dataset = '../data/pl_trusted_size1_noclc_scaled_pca.csv'

    init_param = dict()
    param_grid = {}

    results_file = 'experiments/naive_bayes_model.txt'
    model_file = 'experiments/naive_bayes_model.pkl'

    model_selection_pipeline(dataset, NaiveBayesModel, init_param, param_grid,
                             results_file=results_file)


    # Evaluation on complete data:

    # Top30 score: 0.27030104919069964
    # MRR score: 0.040149805129214226
    # Accuracy: 0.01028101439342015

    # Params: {'ranking_size': 30, 'var_smoothing': 1e-09}
