import numpy as np
import pandas as pd
from classifier import Classifier

from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(Classifier):

    """Random forest classifier.
       This is a ensemble methods that fits a number of decision trees on
       various sub-samples of the dataset (both on features and examples) and
       averages the predictions of the trees to improve the accuracy and
       control overfitting.
    """
    def __init__(self, n_estimators=100, criterion='gini',max_depth=None,
                 min_samples_split=2,min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,max_features='auto',
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False,
                 warm_start=False, class_weight=None, ranking_size=30):
        """
           :param n_neighbors: the number of neighbors for predicting class probabilities
           :param metric: the distance metric used, should be something like
            - 'euclidean' for regular euclidean distance
            - 'manhattan'
            - 'haversine' for distances between (latitude,longitude) points only
            - 'cosine': the cosinus similarity
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ranking_size = ranking_size

        # the Scikit-learn random forest classifier
        self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                          criterion=criterion,
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          min_weight_fraction_leaf=min_weight_fraction_leaf,
                                          max_features=max_features,
                                          min_impurity_split=min_impurity_split,
                                          bootstrap=bootstrap,
                                          oob_score=oob_score,
                                          warm_start=warm_start,
                                          class_weight=class_weight,
                                          n_jobs=-1)

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
    print("Test random forest model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = RandomForestModel(n_estimators=150,max_depth=10,bootstrap=True)
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

