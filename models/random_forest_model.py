import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(Classifier):

    """Random forest classifier.
       This is a ensemble methods that fits a number of decision trees on
       various sub-samples of the dataset (both on features and samples) and
       averages the predictions of the trees to improve the accuracy and
       control overfitting.

       In random forests, each tree in the ensemble is built from a sample
       drawn with replacement (i.e., a bootstrap sample) from the training set.
       In addition, when splitting a node during the construction of the tree,
       the split that is chosen is no longer the best split among all features.
       Instead, the split that is picked is the best split among a random subset
       of the features. As a result of this randomness, the bias of the forest
       usually slightly increases (with respect to the bias of a single
       non-random tree) but, due to averaging, its variance also decreases,
       usually more than compensating for the increase in bias, hence yielding
       an overall better model.

       The sub-sample size is always the same as the original input sample size
       but the samples are drawn with replacement if bootstrap is True (default).
       ––––––
       The relative rank (i.e. depth) of a feature used as a decision node in a
       tree can be used to assess the relative importance of that feature with
       respect to the predictability of the target variable. Features used at the
       top of the tree contribute to the final prediction decision of a larger
       fraction of the input samples. The expected fraction of the samples they
       contribute to can thus be used as an estimate of the relative importance
       of the features.

       A VARIANT TO THIS MODEL THAT MAY BE INTERESTING:

       Our data has a lot of overlapping data points of different labels: we
       could try to fit a random forest where each tree is fitted on a random
       subsample with the constraint that is does not contain any overlapping
       points. Maybe it will be more discriminative ?
    """
    def __init__(self, n_estimators=100, criterion='gini',max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features='auto',
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
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.warm_start = warm_start
        # the Scikit-learn random forest classifier
        self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                          criterion=criterion,
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          max_features=max_features,
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

if __name__=='__main__':

    # from evaluate import evaluate_model, model_selection_pipeline
    # from sklearn.model_selection import train_test_split, RandomizedSearchCV
    # from sklearn.utils.estimator_checks import check_estimator
    # from scipy.stats import randint as sp_randint, uniform as sp_uniform
    # import pickle

    # # # for reproducibility
    # # np.random.seed(42)

    # # Working on PlantNet Trusted: 237,086 occurrences
    # # loading the environmental data
    # env_df = pd.read_csv('../data/pl_trusted_size1_noclc_scaled_pca.csv',
    #                  sep=';', header='infer', quotechar='"')

    # # target pandas series of the species identifiers (there are 505 labels)
    # target_df = env_df['glc19SpId']

    # # Next were going to use a random subsample of the data for model evaluation
    # # and random search cross validation. This way we are able to train the
    # # models in a decent time, and random search cross validation will be a too
    # # long process if we trained on all the data (220k+ occurrences)

    # # Then we find the best parameters using this subsample, then train a model
    # # on the complete data with these parameters, and get predictions on the
    # # test set using this model.

    # subsample_size = 5000
    # env_df_s = env_df.sample(n=subsample_size, replace=False, axis=0)
    # target_df_s = env_df_s['glc19SpId']
    # X_s = env_df_s.values
    # y_s = target_df_s.values

    # # First, evaluate the model with arbitrary parameters, to have a first
    # # impression of the score. This allows to exclude models that are not
    # # interesting.
    # n_splits = 2
    # print(f'Test on subsample of size {subsample_size}, on {n_splits} train-test split(s)\n')
    # mean_top30, mean_mrr, mean_mrank, clf =\
    #             evaluate_model(X_s, y_s, RandomForestModel, n_splits=n_splits,
    #                            params=dict(n_estimators=250, max_depth=3,
    #                                        bootstrap=False))
    # print(f'Top30 score: {mean_top30}')
    # print(f'MRR score: {mean_mrr}')
    # print(f'Mean rank: {mean_mrank}')
    # print('Params:',clf.get_params(),'\n')

    # # Use random search cross validation to optimize models, then save scores
    # # and parameters in a text file. Save also model in pickle format.

    # # hyperparameters:

    # param_grid = {'n_estimators': sp_randint(50,500),
    #               'criterion': ['gini', 'entropy'],
    #               'max_depth': sp_randint(2, 15),
    #               'min_samples_split': sp_randint(2,20),
    #               'min_samples_leaf': sp_randint(1,20),
    #               'max_features': sp_uniform(0.2, 1.),
    #               'bootstrap': [False, True]
    #               }
    # n_iter_search = 30
    # clf_search = RandomizedSearchCV(RandomForestModel(),
    #                                 param_distributions=param_grid,
    #                                 n_iter=n_iter_search, cv=10,
    #                                 iid=False,
    #                                 verbose=5,
    #                                 n_jobs=-1)

    # print("Random search cross validation begins\n")
    # clf_search.fit(X_s, y_s)
    # print("Done!\n")
    # print('Random search results on subsample:\n')
    # print(f"Best Top30 score: {clf_search.best_score_}\n")
    # print(f"Best parameters set found:\n{clf_search.best_params_}\n")

    # with open('experiments/random_forest_model.txt', 'w+') as f:
    #     f.write('---------------------------------------------------------------------\n')
    #     f.write('Random search results on subsample:\n\n')
    #     f.write(f"Best Top30 score: {clf_search.best_score_}\n\n")
    #     f.write(f"Best parameters set found:\n{clf_search.best_params_}\n\n")
    #     f.write(f"Cross validation results:\n{clf_search.cv_results_}")

    # # Use the best parameters found to re-train a model using the complete data
    # # First, evaluate the model on a train/test split
    # X = env_df.values
    # y = target_df.values
    # n_splits = 1
    # print(f'Evaluation on complete data, on {n_splits} train-test split(s)\n')
    # mean_top30, mean_mrr, mean_mrank, clf =\
    #             evaluate_model(X, y, RandomForestModel, n_splits=n_splits,
    #                            params=dict(**clf_search.best_params_)
    #                            )
    # print(f'Top30 score: {mean_top30}')
    # print(f'MRR score: {mean_mrr}')
    # print(f'Mean rank: {mean_mrank}\n')

    # with open('experiments/random_forest_model.txt', 'a') as f:

    #     f.write('Evaluation on complete data, on {n_splits} train-test split(s):\n\n')
    #     f.write(f'Top30 score: {mean_top30}\n')
    #     f.write(f'MRR score: {mean_mrr}\n')
    #     f.write(f'Mean rank: {mean_mrank}\n\n')
    #     f.write('---------------------------------------------------------------------\n\n')

    # # Retrain the model using the complete data
    # print(f'Retrain model on complete data')
    # clf_best = RandomForestModel(**clf_search.best_params_)
    # clf_best.fit(X, y)
    # with open('experiments/random_forest_model.pkl', 'wb') as pf:
    #     pickle.dump(clf_best, pf)

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
    results_file = 'experiments/random_forest_model.txt'
    model_file = 'experiments/random_forest_model.pkl'

    model_selection_pipeline(dataset, RandomForestModel, param_grid,
                             results_file=results_file, model_file=model_file)

    # -----------------------------------------------------------------------

    # Random search results on subsample: (random search of 20)

    # Best Top30 score: 0.726968508354838

    # Best parameters set found:
    # {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 12, 'max_features': 0.6464370845408791, 'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 100}
    # Scorer used:

    # -----------------------------------------------------------------------

    # Random search results on subsample: (random search of 20)

    # Best Top30 score: 0.7231140559044663

    # Best parameters set found:
    # {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 3, 'max_features': 0.8760572512085278, 'min_samples_leaf': 6, 'min_samples_split': 6, 'n_estimators': 212}

    # -----------------------------------------------------------------------

    # Random search results on subsample: (random search of 5)

    # Best Top30 score: 0.7301557864056187

    # Best parameters set found:
    # {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 14, 'max_features': 0.7179782287602747, 'min_samples_leaf': 9, 'min_samples_split': 11, 'n_estimators': 90}


    # Evaluation on complete data, on 1 train-test split(s)

    # Top30 score: 0.9993462398245392
    # MRR score: 0.9862601795749031
    # Mean rank: 1.05708316626923

    # MAYBE A HUGE OVERFITTING (THOUGH NOT SURE, NEED TO SEE ON THE TEST SET




    # Top30 score:0.375
    # MRR score:0.0919576936253201
    # Params: {'bootstrap': False, 'class_weight': None, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto', 'min_impurity_decrease': 0.0, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 150, 'ranking_size': 30, 'verbose': None, 'warm_start': False}

    # Class weights should be uniform !
    # class_weight='balanced' gives terrible results (MRR score:0.048)

    # Feature importances sorted with names:
    # chbio_1           0.066431
    # clc               0.056392
    # chbio_11          0.054491
    # Longitude         0.053134
    # Latitude          0.052193
    # chbio_9           0.050266
    # chbio_10          0.046436
    # alti              0.044607
    # chbio_4           0.042577
    # chbio_6           0.042331
    # chbio_3           0.041898
    # chbio_7           0.038869
    # etp               0.038241
    # chbio_8           0.037054
    # chbio_14          0.035261
    # chbio_17          0.034765
    # chbio_18          0.034759
    # chbio_15          0.033355
    # chbio_5           0.030331
    # chbio_2           0.028532
    # chbio_12          0.023309
    # chbio_13          0.019767
    # chbio_19          0.018941
    # chbio_16          0.013218
    # crusting          0.012025
    # text              0.010289
    # awc_top           0.010094
    # bs_top            0.006162
    # proxi_eau_fast    0.005806
    # dgh               0.004879
    # pd_top            0.003892
    # dimp              0.003675
    # erodi             0.002975
    # oc_top            0.002353
    # cec_top           0.000690
