import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

from sklearn.neighbors import RadiusNeighborsClassifier
# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class RadiusNeighborsModel(Classifier):

    """Classifier implementing a vote among neighbors within a given radius
       The radius
       Classifier predicting the labels by counting occurrences among the
       neighbors within a given radius r from a query example.

       In cases where the data is not uniformly sampled,
       radius-based neighbors classifier can be a better choice compared to
       k-nearest neighbors classifier. Points in sparser neighborhoods use fewer
       nearest neighbors for the classification
       For high-dimensional parameter spaces, this method becomes less effective
       due to the so-called “curse of dimensionality”.

       The choice of the radius is highly data-dependent, similarly to k in the
       k-nearest neighbors classifier.
    """
    def __init__(self, radius=1.0, weights='uniform', p=2, metric='minkowski', ranking_size=30):
        """
           :param radius:
           Range of parameter space to use by default for query example
           :param weights:
           The weight function used in prediction.
           Possible values:
            - 'uniform' : uniform weights. All points in each neighborhood are
               weighted equally.
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

           :param how_outliers:
           The way outlier samples (samples with no neighbors on given radius)
           are predicted. Possible values:
           - 'most_common' : return the most common labels in the training set
           - 'random' : return a random label ranking from the training set
           - [callable] : a user-defined function which accepts an example and
              returns a label ranking.
        """
        self.radius = radius
        if weights != 'uniform':
            raise Exception("Only 'uniform' for the weights parameter is supported")
        self.weights = weights
        self.p = p
        self.metric = metric
        self.ranking_size = ranking_size
        # Scikit-learn Radius neighbors classifier
        self.clf = RadiusNeighborsClassifier(radius=radius,
                                        weights=weights,
                                        p=p,
                                        metric=metric,
                                        n_jobs=-1
                                        )
    def fit(self, X, y):

        super().fit(X, y)

        # The way outlier samples (samples with no neighbors on given radius)
        # are predicted is the following: predict only one label, the most
        # common one in the training set
        y_unique,counts = np.unique(y, return_counts=True)
        outlier_label = y_unique[np.argmax(counts)]
        self.outlier_label_ = outlier_label
        self.outlier_proba_ = np.max(counts)/len(y)

    def predict(self, X, return_proba=False):

        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)

        # Compute neighbors indexes and distances for every test example
        # The result points are not necessarily sorted by distance to their
        # query point.

        neigh_distances, neigh_indexes = self.clf.radius_neighbors(X, return_distance=True)
        # neigh_argsorts = [np.argsort(ngh_dist) for ngh_dist in distances]

        y_predicted = list()
        y_predicted_probas = list()
        for indexes,distances in zip(neigh_indexes, neigh_distances):

            try:
                y_neigh = self.y_[indexes]
            except IndexError:

                y_predicted.append([self.outlier_label_])
                y_predicted_probas.append([self.outlier_proba_]+[0. for k in range(self.ranking_size)])
                continue
            y_unique, counts = np.unique(y_neigh, return_counts=True)

            # Get the most frequent labels from the neighbors
            # probability estimate
            probas = counts/len(y_neigh)
            # get the indexes of the sorted probabilities, in decreasing order
            top_predictions = np.flip(np.argsort(probas)[-self.ranking_size:],axis=0)
            y_pred = y_neigh[top_predictions]
            y_pred_probas = probas[top_predictions]
            if len(y_unique) < self.ranking_size:
                rank_probas = np.zeros(self.ranking_size)
                rank_probas[:len(y_unique)] = y_pred_probas
                y_pred_probas = rank_probas

            y_predicted.append(y_pred)
            y_predicted_probas.append(y_pred_probas)

        if return_proba:
            return np.array(y_predicted),np.array(y_predicted_probas)

        return np.array(y_predicted)

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
    print("Test radius neighbors model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = RadiusNeighborsModel(radius=5., metric='euclidean')
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

    print(classifier.clf.get_params())

    # Top30 score: 0.194
    # MRR score: 0.02807270977637864
    # Params: {'metric': 'euclidean', 'p': 2, 'radius': 5.0, 'ranking_size': 30, 'weights': 'uniform'}

