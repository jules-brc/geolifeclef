import numpy as np
import pandas as pd
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
        self.ranking_size = ranking_size
        # the Scikit-learn K neighbors classifier
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                                            metric=metric,
                                            weights=weights,
                                            n_jobs=-1)

if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from glcdataset import build_environmental_data
    from sklearn.preprocessing import StandardScaler

    # for reproducibility
    np.random.seed(None)

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
    print("Test KNN model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = KNearestNeighborsModel(n_neighbors=150, weights='distance',metric='euclidean')
    classifier.fit(X_train,y_train)
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')
    print('Params:',classifier.get_params())

    print("Example of predict proba:")
    print(f"occurrence:\n{X_test[12]}")
    y_pred, y_probas = classifier.predict(X_test[12].reshape(1,-1), return_proba=True)
    print(f'predicted labels:\n{y_pred}')
    print(f'predicted probas:\n{y_probas}')

    # Test KNN model
    # Top30 score:0.3
    # MRR score:0.06804483513744343
    # Params: {'metric': 'euclidean', 'n_neighbors': 150, 'p': None, 'ranking_size': 30, 'weights': 'distance'}

    # Test KNN model
    # Top30 score:0.29
    # MRR score:0.06622010516111612
    # Params: {'metric': 'cosine', 'n_neighbors': 150, 'p': None, 'ranking_size': 30, 'weights': 'distance'}

    # Small number of neighbors give bad results. At minimum 30
    # The euclidean metric and cosine metric seems to give almost identical results,
    # remains to be confirmed.
