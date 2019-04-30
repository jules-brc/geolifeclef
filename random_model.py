import numpy as np
import pandas as pd
from classifier import Classifier

# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class RandomModel(Classifier):

    """Random model returning a random list of labels from the training set
       for a test occurence.
       This stupid model is just used as a baseline.
    """
    def __init__(self, ranking_size=30):
        """
        """
        self.ranking_size = ranking_size

    def fit(self, X, y):

        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        y_unique = np.unique(y)
        self.classes_ = y_unique
        self.y_predicted_probas_ = (np.ones(len(y_unique))/len(y_unique))[:self.ranking_size]
        return self

    def predict(self, X, return_proba=False):
        """No return probabilities for the random model
        """
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        y_predicted = [np.random.choice(self.classes_, size=self.ranking_size, replace=False) for i in range(len(X))]
        if return_proba:
            return np.array(y_predicted), np.tile(self.y_predicted_probas_,(len(X),1))

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
    print("Test random model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = RandomModel()
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
