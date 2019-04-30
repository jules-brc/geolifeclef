import numpy as np
import pandas as pd
from classifier import Classifier

from sklearn.neighbors import RadiusNeighborsClassifier
# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

## NEEDS DEBUGGING : REWRITE FIT METHOD TO PREDICT PROBABILITIES
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

       The choice of the radius is highly data-dependent, just like k for the
       k-nearest neighbors classifier.
    """
    def __init__(self, radius=1.0, weights='uniform', p=2, metric='minkowski', how_outliers='most_common', ranking_size=30):
        """
           :param radius:
           Range of parameter space to use by default for query example
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
           :param how_outliers:
           The way outlier samples (samples with no neighbors on given radius)
           are predicted. Possible values:
           - 'most_common' : return the most common labels in the training set
           - 'random' : return a random label ranking from the training set
           - [callable] : a user-defined function which accepts an example and
              returns a label ranking.
        """
        self.radius = radius
        self.weights = weights
        self.p = p
        self.metric = metric
        if not how_outliers in {'most_common','random',None}:
            raise RuntimeError('How outliers: unknown argument')

        self.ranking_size = ranking_size
        # Scikit-learn Radius neighbors classifier
        self.clf = RadiusNeighborsClassifier(radius=radius,
                                        weights=weights,
                                        p=p,
                                        metric=metric,
                                        outlier_label=None,
                                        n_jobs=-1
                                        )
        def fit(self, X, y):

            # set the returned labels for outliers examples
            if self.how_outliers == 'most_common':
                y_unique,counts = np.unique(y, return_counts=True)
                outlier_label = y_unique[np.argmax(counts)]
                self.set_params(outlier_label=outlier_label)
                self.clf.set_params(outlier_label=outlier_label)

            elif self.how_outliers == 'random':
                outlier_lable = np.random.choice(np.unique(y))
                self.set_params(outlier_label=None)
                self.clf.set_params(outlier_label=None)
            else:
                self.set_params(outlier_label=None)
                self.clf.set_params(outlier_label=None)
            super().fit(X, y)

        def predict_proba(self, X, y):

            # check is fit had been called
            check_is_fitted(self, ['X_', 'y_'])
            # input validation
            X = check_array(X)
            # predict probabilities for each label
            y_predicted = list()
            # compute neighbors distances and indexes for every test example
            distances, indexes = self.clf.radius_neighbors(X,
                                                                      return_distance=True)
            neigh_argsorts = [np.argsort(ngh_dist) for ngh_dist in distances]
            y_predicted = list()
            for argsort,index in zip(neigh_argsort, indexes):
                # indexes of closests neighbors
                try:
                    argsort_index = index[argsort]
                except IndexError: # outlier example: assign an outlier value
                    pass

            for ngh_idx,ngh_argsort in zip(ngh_indexes, ngh_distances_argsort):

            ngh_indexes_argsort = [ngh_idx[ngh_argsort]for ngh_idx,ngh_argsort in zip(ngh_distances_ar)

        probas = self.clf.predict_proba(X)
        # get the indexes of the sorted probabilities, in decreasing order
        top_predictions = np.argsort(probas, axis=1)[:,-self.ranking_size:]
        # get the names of the classes from the indexes
        y_predicted = self.classes_[top_predictions]
        return np.array(y_predicted)

        y_predicted = list()
        # selecting closests points distinct labels
        for argsort in all_argsorts:

            # get closests labels
            y_closest = self.y_[argsort]
            # predict distinct labels
            y_found = set()
            y_pred = list()
            for y in y_closest:
                if len(y_pred)>= self.ranking_size:
                    break
                if y not in y_found:
                  y_pred.append(y)
                  y_found.add(y)

            y_predicted.append(y_pred)
            # THIS DOESN'T WORK: don't know why it return duplicates
            # y_pred = self.y_[np.sort(y_indexes)][:self.ranking_size]
        return y_predicted

        ## TODO : define a predict_proba method since it's not available with
        # the radius neighbors classifier from Scikit-learn


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
    classifier = RadiusNeighborsModel(radius=1., weights='distance',metric='euclidean')
    classifier.fit(X_train,y_train)
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')

    print(classifier.clf.get_params())
