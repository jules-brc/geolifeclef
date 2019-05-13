import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

from sklearn.neural_network import MLPClassifier

class NeuralNetworkModel(Classifier):

    """This is a multi-layer Perceptron (MLP) classifier that trains using
       Backpropagation.
       This model optimizes the log-loss function using LBFGS, SGD or Adam
       optimizer.
    """
    def __init__(self, hidden_layer_sizes=(100, ), activation='relu',
                 solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate='adaptive', learning_rate_init=0.001,
                 max_iter=200, shuffle=True, tol=1e-4, verbose=False,
                 early_stopping=False, validation_fraction=0.1,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10,
                 ranking_size=30):
        """
           :param hidden_layer_sizes:
           The ith element represents the number of neurons in the ith hidden
           layer.

           :param alpha:
           L2 penalty (regularization term) parameter.

           :param batch_size:
           Size of minibatches for stochastic optimizers. If the solver is
           'lbfgs', the classifier will not use minibatch. When set to “auto”,
           batch_size=min(200, n_samples)

           :param learning_rate:
           In {‘constant’, ‘invscaling’, ‘adaptive’}; learning rate schedule
           for weight updates. Only used when solver='sgd'.

           :param learning_rate_init:
           The initial learning rate used. Only used when solver=’sgd’ or ‘adam’.

           :param max_iter:
           Maximum number of iterations. The solver iterates until convergence
           (determined by ‘tol’) or this number of iterations.
           For stochastic solvers (‘sgd’, ‘adam’), note that this determines
           the number of epochs (how many times each data point will be used),
           not the number of gradient steps.

           :param shuffle:
           Whether to shuffle samples in each iteration. Only used when
           solver=’sgd’ or ‘adam’.

           :param verbose:
           Whether to print progress messages to stdout.

           :param tol:
           Tolerance for the optimization. When the loss or score is not
           improving by at least tol for n_iter_no_change consecutive
           iterations, unless learning_rate is set to ‘adaptive’, convergence
           is considered to be reached and training stops.

           :param early_stopping:
           Whether to use early stopping to terminate training when validation
           score is not improving. If set to true, it will automatically set
           aside 10% of training data as validation and terminate training when
           validation score is not improving by at least tol for n_iter_no_change
           consecutive epochs. Only effective when solver=’sgd’ or ‘adam’
           :param validation_fraction:
           The proportion of training data to set aside as validation set for
           early stopping. Must be between 0 and 1. Only used if early_stopping
           is True
           :param n_iter_no_change:
           Maximum number of epochs to not meet tol improvement. Only effective
           when solver=’sgd’ or ‘adam’
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size,
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.tol = tol
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change

        self.clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                 activation=activation,
                                 solver=solver,
                                 alpha=alpha,
                                 batch_size=batch_size,
                                 learning_rate=learning_rate,
                                 learning_rate_init=learning_rate_init,
                                 max_iter=max_iter,
                                 shuffle=shuffle,
                                 tol=tol,
                                 verbose=verbose,
                                 early_stopping=early_stopping,
                                 validation_fraction=validation_fraction,
                                 beta_1=beta_1,
                                 beta_2=beta_2,
                                 epsilon=epsilon,
                                 n_iter_no_change=n_iter_no_change)

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
    print("Test neural network model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = NeuralNetworkModel(hidden_layer_sizes=(50,), solver='adam',
                                    alpha=1e-3,activation='relu',
                                    max_iter=200, early_stopping=True)
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

    # Test neural network model
    # Top30 score:0.32
    # MRR score:0.0661882521391968
    # Params: {'activation': 'relu', 'alpha': 0.001, 'batch_size': ('auto',), 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': True, 'epsilon': 1e-08, 'hidden_layer_sizes': (50,), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 200, 'n_iter_no_change': 10, 'ranking_size': 30, 'shuffle': True, 'solver': 'adam', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False}
