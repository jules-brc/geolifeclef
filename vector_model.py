
import numpy as np
import pandas as pd
from classifier import Classifier

# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import pairwise_distances

# TODO : RETURN PROBABILITIES IN PREDICT
class VectorModel(Classifier):

    """Simple nearest neighbors classifier which predict labels from the
       neighbors ordered by increasing distance from a query example. Returns
       the nearest-neighbor label, then the next neighbor label, and so on..
       with the constraint that a neighbor's label is chosen only if it hasn't
       been added so far to the ranking.

       This model may be prone to overfitting because it predict the next neighbor
       label in a greedy fashion, not taking into account a neighborhood of
       several examples. This is why a k-nearest neighbor classifier may give
       better results (in practice, it does).

       The vector model does not support probabilities predictions but only
       labels predictions.
    """
    def __init__(self, metric='euclidean', ranking_size=30):
        """
        :param metric: the distance metric used, should be something like
            - 'euclidean' for regular euclidean distance
            - 'manhattan'
            - 'haversine' for distances between (latitude,longitude) points only
            - 'cosine'
        """
        self.metric = metric
        self.ranking_size = ranking_size

    def predict(self, X):

        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        # compute all distances, in parallel if possible
        all_distances = pairwise_distances(X, self.X_,
                                           metric=self.metric, n_jobs=-1)
        # get index of the sorted points' distances
        all_argsorts = np.argsort(all_distances, axis=1)
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
    print("Test vector model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = VectorModel(metric='cosine')
    classifier.fit(X_train,y_train)
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')
    print('Params:',classifier.get_params())

    # Test vector model
    # Top30 score:0.246
    # MRR score:0.05718168788586186
    # Params: {'metric': 'euclidean', 'ranking_size': 30}

    # Test vector model
    # Top30 score:0.23800000000000002
    # MRR score:0.0586088829636054
    # Params: {'metric': 'cosine', 'ranking_size': 30}
