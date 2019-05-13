import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Using clusters as features

np.random.seed(42)

class ClusteringModel():

    def _load_data(self, sklearn_load_ds):

        data = sklearn_load_ds
        X = pd.DataFrame(data.data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, data.target, test_size=0.3, random_state=42)

    def __init__(self, sklearn_load_ds):
        self._load_data(sklearn_load_ds)

    def classify(self, model=SVC(gamma='scale', kernel='rbf')):
        # model=LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100)):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))
        return self


    def clusterize(self, output='add'):
        n_clusters = len(np.unique(self.y_train))
        clf = KMeans(n_clusters = n_clusters)
        clf.fit(self.X_train)
        y_labels_train = clf.labels_
        y_labels_test = clf.predict(self.X_test)
        if output == 'add':
            self.X_train['km_clust'] = y_labels_train
            self.X_test['km_clust'] = y_labels_test
        elif output == 'replace':
            self.X_train = y_labels_train[:, np.newaxis]
            self.X_test = y_labels_test[:, np.newaxis]
        else:
            raise ValueError('output should be either add or replace')
        return self


# Prediction using only K-means' clusters: find a number of clusters equal to
# the number of classes, replace the dataset by their assignation to cluster.
# This is not so bad (accuracy 78%) but way less powerful than SVMs for example.
# This is an approximation of the nearest-centroid classifier


cl = ClusteringModel(load_digits()).clusterize(output='add').classify()

print('X data:\n',cl.X_train)
print('y data:\n', cl.y_train)

# Prediction wthout using K-means' clusters: just use a SVMs on the digits
# dataset. This gives 99.07% accuracy.

ClusteringModel(load_digits()).classify()

# Prediction using K-means clusters and another classifier: add to the dataset their
# cluster of assignation. It does not seem to improve the model: this gives 98.8%
# accuracy. Causes may be:

# - the classifier already captured all the information and adding a new feature
# confuses him

# - the clusters assignments, integers between 0 and n_classes, aren't categorical
# and non-ordinal values: therefore they should be set as n_classes one-hot encoding
# variables.

# - the clustering is bad: either the data is not 'clusterable', or the k-means
#   clustering of spheric groups is not appropriate. Maybe a spectral clustering
#   may work better in this case.

ClusteringModel(load_digits()).clusterize(output='add').classify()
