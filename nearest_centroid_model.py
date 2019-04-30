import numpy as np
import pandas as pd
from classifier import Classifier

from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances
# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class NearestCentroidModel(Classifier):

    """Simple classifier that represents each class by the centroid of its
       members. Query example are classified to the class with the nearest
       centroid.

       It has no parameters to choose, making it a good baseline classifier.
       It does, however, suffer on non-convex classes, as well as when classes
       have drastically different variances, as equal variance in all dimensions
       is assumed.

       A parameter allows to implement the nearest shrunken centroid classifier :
       in effect, the value of each feature for each centroid is divided by the
       within-class variance of that feature. The feature values are then reduced
       a threshold. Most notably, if a particular feature value crosses zero,
       it is set to zero.
       In effect, this removes the feature from affecting the classification.
       This is useful, for example, for removing noisy features.
       A small shrink threshold (for example 0.2) may increase the accuracy.

       When applied to text classification using tf-idf vectors to represent
       documents, the nearest centroid classifier is known as the Rocchio
       classifier.
    """
    def __init__(self, metric='euclidean', shrink_threshold=None, ranking_size=30):
        """
          :param metric:
           The metric to use when calculating distance between instances.
           The default metric is Euclidean. Choices are:
            - 'euclidean' for standard Euclidean distance
            - 'manhattan': for the Manhattan distance
            - 'haversine' for distances between (latitude,longitude) points only
            - 'cosine': for cosinus similarity
           :param shrink_thresold:
            The threshold for shrinking centroids to remove features
        """
        self.metric = metric
        self.shrink_threshold = shrink_threshold
        self.ranking_size = ranking_size
        self.clf = NearestCentroid(metric=metric,
                                   shrink_threshold=shrink_threshold)

    def predict(self, X, return_proba=False, clf_predict=False, *args, **kwargs):
        """Predict the list of labels most likely to be observed
           for the data points given
        """
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        # compute all distances, in parallel if possible
        all_distances = pairwise_distances(X, self.clf.centroids_,
                                           metric=self.metric, n_jobs=-1)

        # get index of the sorted centroids' distances
        all_argsorts = np.argsort(all_distances, axis=1)
        # selecting closests classes centroids

        y_predicted = [self.classes_[argsort][:self.ranking_size] for argsort in all_argsorts]
        if return_proba:
            y_predicted_probas = list()
            for distance,argsort in zip(all_distances,all_argsorts):
                # predicting probabilities: inverse of the distance, normalized
                inverse_distances = (distance[argsort])**(-1)
                y_predicted_probas.append((inverse_distances / np.sum(inverse_distances))[:self.ranking_size])

            return np.array(y_predicted), np.array(y_predicted_probas)
        return np.array(y_predicted)

if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from glcdataset import build_environmental_data, get_taxref_names
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

    # correspondence table between ids and the species taxonomic names
    # (Taxref names with year of discoverie)
    taxonomic_names = pd.read_csv('../data/occurrences/taxaName_glc19SpId.csv',
                                  sep=';',header='infer', quotechar='"',low_memory=True)

    # building the environmental data
    env_df = build_environmental_data(df[['Latitude','Longitude']],patches_dir='example_envtensors')
    X = env_df.values
    y = target_df.values
    # Standardize the features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Evaluate as the average accuracy on one train/split random sample:
    print("Test nearest centroid model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = NearestCentroidModel(metric='euclidean', shrink_threshold=0.9)
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
    print(f'sum of probas: {np.sum(y_probas)}')
    print(f'predicted probas:\n{y_probas}')

    # Test nearest centroid model
    # Top30 score:0.17
    # MRR score:0.025115179111806874
    # Params: {'metric': 'euclidean', 'ranking_size': 30, 'shrink_threshold': 0.9}
