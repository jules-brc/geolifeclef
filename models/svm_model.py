import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

from sklearn.svm import SVC

class SupportVectorModel(Classifier):

    """
    """
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 shrinking=True, ranking_size=30):
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
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        # the Scikit-learn K neighbors classifier
        self.clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                       coef0=coef0, shrinking=shrinking, probability=True)

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
    print("Test SVM model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = SupportVectorModel(C=0.5, kernel='rbf', gamma='scale')

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

    # Top30 score:0.34900000000000003
    # MRR score:0.07481428223366225
    # Params: {'C': 1.0, 'coef0': 0.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear', 'ranking_size': 30, 'shrinking': True}

    # Problem: The model takes a while to fit on this small prototyping dataset
    # of only 5000 samples compared to +250k in total
    # This is not scalable
