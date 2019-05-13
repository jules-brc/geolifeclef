import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier
from sklearn.linear_model import SGDClassifier

class StochasticGradientModel(Classifier):

    """
    """
    def __init__(self, loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 max_iter=None, shuffle=True,
                 verbose=0, epsilon=0.1,
                 learning_rate='optimal', eta0=0.0, power_t=0.5,
                 early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, class_weight=None, warm_start=False,
                 average=False, ranking_size=30):
        """
        """
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average

        self.clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha,
                                 l1_ratio = l1_ratio, max_iter=max_iter,
                                 tol=1e-3, shuffle=shuffle, verbose=verbose,
                                 epsilon=epsilon, learning_rate=learning_rate,
                                 eta0=eta0, power_t=power_t,
                                 early_stopping=early_stopping,
                                 validation_fraction=validation_fraction,
                                 n_iter_no_change=n_iter_no_change,
                                 class_weight=class_weight, warm_start=warm_start,
                                 average=average,
                                 n_jobs=-1)

        self.ranking_size = ranking_size

if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from glcdataset import build_environmental_data
    from sklearn.preprocessing import StandardScaler

    # for reproducibility
    np.random.seed(42)

    # working on a subset of Pl@ntNet Trusted: 2500 occurrences
    df = pd.read_csv('example_occurrences.csv',
                 sep=';', header='infer', quotechar='"', low_memory=True)

    df = df[['Longitude','Latitude','glc19SpId','scName']]\
           .dropna(axis=0, how='all')\
           .astype({'glc19SpId': 'int64'})

    # target pandas series of the species identifiers (there are 505 labels)
    target_df = df['glc19SpId']

    # building the environmental data
    env_df = build_environmental_data(df[['Latitude','Longitude']],patches_dir='example_envtensors')
    X = env_df.values
    y = target_df.values
    # Standardize the features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Evaluate as the average accuracy on one train/split random sample:
    print("Test SGD model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = StochasticGradientModel(loss='log', max_iter=200)

    classifier.fit(X_train,y_train)
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')
    print('Params:',classifier.get_params())
    print("Example of predict proba:")
    print(f"occurrence:\n{X_test[12]}")
    y_pred, y_probas = classifier.predict(X_test[12].reshape(1,-1), return_proba=True)
    print(f'predicted labels:\n{y_pred}')
    print(f'predicted probas:\n{y_probas}\n')

    print(f'Mean rank:{classifier.mean_rank(y_predicted, y_test)}')

    # Log loss: logistic regression

    # Top30 score:0.34
    # MRR score:0.05011857619288295
    # Params: {'alpha': 0.0001, 'average': False, 'class_weight': None, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.0, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': 200, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'ranking_size': 30, 'shuffle': True, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

    # Modified huber loss gives shit results
