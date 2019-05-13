import numpy as np
import pandas as pd
import sys
sys.path.extend(['..','./base','./evaluation'])
from classifier import Classifier

from sklearn.linear_model import LogisticRegression

class LogisticModel(Classifier):
    """
    """
    def __init__(self, penalty='l2', C=1.0, intercept_scaling=1,
                 class_weight=None, solver='liblinear', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False,
                 ranking_size=30):
        """
        """
        self.clf = LogisticRegression(penalty=penalty, C=1.0,
                                      intercept_scaling=intercept_scaling,
                                      solver=solver, max_iter=max_iter,
                                      multi_class=multi_class,
                                      verbose=verbose,
                                      warm_start=warm_start,
                                      n_jobs=-1)

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
    print("Test logistic model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    classifier = LogisticModel(solver='lbfgs', multi_class='auto', max_iter=100)

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

    # Top30 score:0.311
    # MRR score:0.06645575570357123
    # Params: {'C': None, 'class_weight': None, 'intercept_scaling': None, 'max_iter': None, 'multi_class': None, 'penalty': None, 'ranking_size': 30, 'solver': None, 'verbose': None, 'warm_start': None}
