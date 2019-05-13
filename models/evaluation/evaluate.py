
from sklearn.model_selection import train_test_split
import numpy as np

def evaluate_on_train_test_splits(X, y, Model, n_splits=1, params=None):

    """Evaluate a model on some provided data, using a certain number of
       train/test splits, computing the Top30 score, the MRR score, and the
       mean rank, averaged over all train/test splits.

       :param n_splits:
       Number of train/test splits
       :param params: dict of params to pass to the model
       :return:
       The last fitted model
    """
    scores = {'mrr':[],'top30':[],'mean_rank':[],'accuracy':[]}

    for n in range(n_splits):

        # Evaluate as the average accuracy on one train/split random sample:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
        clf = Model(**params)
        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_test)

        for key in scores.keys():

            score = getattr(clf, key+'_score')(y_test, y_pred)
            scores[key].append(score)

    for key in scores.keys():

        # computes the mean and the (biased) standard deviation of the scores
        mean = np.mean(scores[key])
        std = np.sqrt(np.mean((scores[key] - mean)**2))

        scores[key] = {'mean':mean, 'std':std}

    return scores, clf

def evaluate_on_test_data(X, y, X_test,y_test, Model, params):

    """Trains a model on a training set and evaluates it on a separate test set
    """
    clf = Model(**params)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)

    scores = {'top30':0., 'mrr':0., 'mean_rank':0., 'accuracy':0.}
    for key in scores.keys():

            score = getattr(clf, key+'_score')(y_test, y_pred)
            scores[key] = score

    return scores, clf


import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,\
                                    RandomizedSearchCV
import pickle

def model_selection_pipeline(dataset, Model, init_param=None, param_grid=None,
                             subsample_size=5000, n_iter_search=30,
                             cv_splits=10,
                             results_file=None, model_file=None,):

    """The all pipeline for selecting a model from an already transformed dataset,
       The dataset must contain a column of the species label.

       :param init_param: evaluate the model on random subsample with those
       parameters first
       :param param_grid: the distributions of parameters to pass to the
       randomized search CV
       :param n_iter_search: number of parameters tries in the randomized
       search CV
       :param cv_splits: number of folds for per CV in the random search
       :param results_file: where to print the results of the evaluation of
       models (first evaluation, CV results, final evaluation)
       :param model_file: where to serialize the model trained on all the data

        EXAMPLE:
        dataset = '../data/pl_trusted_size1_noclc_scaled_pca.csv'
        param_grid = {'n_estimators': sp_randint(50,500),
                      'criterion': ['gini', 'entropy'],
                      'max_depth': sp_randint(2, 15),
                      'min_samples_split': sp_randint(2,20),
                      'min_samples_leaf': sp_randint(1,20),
                      'max_features': sp_uniform(0.2, 1.),
                      'bootstrap': [False, True]
                      }
        results_file = 'experiments/random_forest_model.py'
        model_file = 'experiments/random_forest_model.pkl'
        model_selection_pipeline(dataset, RandomForestModel, param_grid,
                             results_files, model_file)
    """
    # for reproducibility
    # np.random.seed(42)

    # Working on PlantNet Trusted: 237,086 occurrences
    # loading the environmental data
    df = pd.read_csv(dataset, sep=';', header='infer', quotechar='"')

    # target pandas series of the species identifiers: there a 1,364 labels
    target_df = df['glc19SpId']
    df = df.drop('glc19SpId', axis=1)

    # First, divide the data into a training set and a test set

    train, test, target_train, target_test = train_test_split(df, target_df, test_size=0.4)
    assert('glc19SpId' not in train.columns)
    assert('glc19SpId' not in test.columns)
    assert(target_train.name =='glc19SpId')
    assert(target_test.name =='glc19SpId')

    # We're going to use a random subsample of the data for model evaluation
    # and hyperparameter tuning. This way we are able to train the
    # models in a decent time, as random search cross validation will be a too
    # long process if we trained on all the data (220k+ occurrences)

    # subsample_size = 5000

    #TODO : Use stratified shuffle split of Sklearn instead to reduce bias to
    # the minimum

    train_s = train.sample(n=subsample_size, replace=False, axis=0)
    target_train_s = target_train.loc[train_s.index]
    assert(len(train_s) == len(target_train_s))
    X_s = train_s.values
    y_s = target_train_s.values

    # First, evaluate the model with arbitrary parameters, to have a first
    # impression of the score. This allows to exclude models that are not
    # interesting.
    n_splits = 5
    print(f'Test on subsample of size {subsample_size}, on {n_splits} train-test split(s)\n')
    scores, clf = evaluate_on_train_test_splits(X_s, y_s, Model,
                                                n_splits=n_splits,
                                                params=init_param)

    print(f"Top30 score: {scores['top30']['mean']}")
    print(f"MRR score: {scores['mrr']['mean']}")
    print(f"Accuracy: {scores['accuracy']['mean']}")
    print(f"Mean rank: {scores['mean_rank']['mean']}")
    print('Params:',clf.get_params(),'\n')

    # To select the best hyperparameters, we use a random search CV on the
    # subsample of the data. The scores and parameters found are saved in a
    # text file.

    # hyperparameters grid
    print('Tuning hyperparameters:\n', list(param_grid.keys()))

    # To remove bias introduced by particular CV folds, we use a repeated
    # stratified k-fold CV stategy for the random search

    # cv_splits = 10
    n_repeats_k_fold = 3 # number of times CV is repeated
    repeated_k_fold = RepeatedStratifiedKFold(n_repeats=n_repeats_k_fold, n_splits=cv_splits)
    # k_fold = StratifiedKFold(n_splits=cv_splits, shuffle=True)

    # n_iter_search = 30
    clf_search = RandomizedSearchCV(Model(),
                                    param_distributions=param_grid,
                                    n_iter=n_iter_search,
                                    cv = repeated_k_fold,
                                    iid=False, refit=False,
                                    return_train_score=True,
                                    verbose=4, n_jobs=-1)

    # this computes also train scores to gain insights on how different parameter
    # settings impact the overfitting/underfitting trade-off (variance/bias).

    print("Random search cross validation begins\n")
    clf_search.fit(X_s, y_s)
    print("\nDone!\n")
    print('Random search results on subsample:\n')
    print(f"Best Top30 score: {clf_search.best_score_}\n")
    print(f"Best parameters set found:\n{clf_search.best_params_}\n")

    if results_file is not None:
        with open(results_file, 'a+') as f:
            f.write('---------------------------------------------------------------------\n\n')
            f.write('Random search results on subsample:\n\n')
            f.write(f"Best Top30 score: {clf_search.best_score_}\n\n")
            f.write(f"Best parameters set found:\n{clf_search.best_params_}\n\n")
            f.write(f"Cross validation results:\n{clf_search.cv_results_}\n\n")

    # Using the best parameters found, we train a model on the complete training
    # data and evaluate it on the test data

    X, y = train.values, target_train.values
    X_test, y_test = test.values, target_test.values

    print(f'Evaluation on complete data\n')

    params = clf_search.best_params_
    scores, clf = evaluate_on_test_data(X, y, X_test, y_test, Model, params)

    print(f"Top30 score: {scores['top30']}")
    print(f"MRR score: {scores['mrr']}")
    print(f"Accuracy: {scores['accuracy']}")
    print(f"Params:\n{clf_search.best_params_}\n")

    if results_file is not None:
        with open(results_file, 'a+') as f:

            f.write(f'Evaluation on complete data:\n\n')
            f.write(f"Top30 score: {scores['top30']}\n")
            f.write(f"MRR score: {scores['mrr']}\n")
            f.write(f"Accuracy: {scores['accuracy']}\n")
            f.write('---------------------------------------------------------------------\n\n')

    if model_file is not None:

        X, y = df.values, target_df.values
        # Retrain the model using the complete data
        print(f'Retrain model on complete data')
        clf_best = Model(**clf_search.best_params_)
        clf_best.fit(X, y)
        print(f'Save model')
        with open(model_file, 'wb') as pf:
            pickle.dump(clf_best, pf)


def generate_challenge_run(model, test_data, run_file=None):

    """Generate the test predictions in the format of the runs for the challenge.
       Optionally write the run in a csv file

        EXAMPLE:
        with open('experiments/random_forest_model.pkl', 'wb') as f:
            model = pickle.load(f)

        test_data = '../data/test_size1_noclc_scaled_pca.csv'
        run_file = 'runs/random_forest_run.csv'

        run_df = generate_challenge_run(model, test_data, run_file)
        display(run_df)
    """
    # loading the test environmental data
    test_df = pd.read_csv(test_data, sep=';', header='infer', quotechar='"')

    # the resulting run dataframe
    run_df = pd.DataFrame(columns=['glc19TestOccId', 'glc19SpId', 'Rank',
                          'Probability'])

    # TO DO: FINISH IT
    # ...

    # save the run in a csv file
    pd.to_csv(run_file, sep=';', index=False, quotechar='"')
