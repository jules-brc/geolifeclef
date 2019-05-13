
def evaluate_model(X, y, Model, pca_n_components=None,
                   n_splits=1, params=None):

    """Evaluate a model on some provided data, using a certain number of
       train/test splits, computing the Top30 score, the MRR score, and the
       mean rank, averaged over all train/test splits.

       :param scale:
       If True, use a standard scaler on the data before
       :param pca:
       If True, use principal component analysis to transform the data before
       :param pca_n_components:
       Parameters passed to PCA. Can be number of components, ratio of explained
       variance to retain, or 'mle' to find the best number of components.

       :param pca_whiten:
       If True, whiten components vectors of the PCA to ensure uncorrelated
       outputs with unit component-wise variances.
       Whitening will remove some information from the transformed signal (the
       relative variance scales of the components) but can sometime improve the
       predictive accuracy of the downstream estimators by making their data
       respect some hard-wired assumptions.

       :param n_splits:
       Number of train/test splits

       :return:
       The last fitted model
    """
    mean_mrr = 0.
    mean_top30 = 0.
    mean_mrank = 0.
    print(f'Test of {str(Model().__class__.__name__)} on {n_splits} train-test split(s)')
    for n in range(n_splits):

        # Evaluate as the average accuracy on one train/split random sample:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

        classifier = Model(**params)
        classifier.fit(X_train,y_train)

        y_predicted = classifier.predict(X_test)

        mean_top30 += classifier.top30_score(y_predicted, y_test)
        mean_mrr += classifier.mrr_score(y_predicted, y_test)
        mean_mrank += classifier.mean_rank(y_predicted, y_test)

    mean_top30 /= n_splits
    mean_mrr /= n_splits
    mean_mrank /= n_splits

    print('Params:',classifier.get_params())
    print(f'Top30 score: {mean_top30}')
    print(f'MRR score: {mean_mrr}')
    print(f'Mean rank: {mean_mrank}')

    # print("Example of predict proba:")
    # print(f"occurrence:\n{X_test[12]}")
    # y_pred, y_probas = classifier.predict(X_test[12].reshape(1,-1), return_proba=True)
    # print(f'predicted labels:\n{y_pred}')
    # print(f'predicted probas:\n{y_probas}')


if __name__=='__main__':

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from glcdataset import build_environmental_data
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    from knn_model import KNearestNeighborsModel

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

    env_df = build_environmental_data(df[['Latitude','Longitude']],patches_dir='example_envtensors')
    X = env_df.values
    y = target_df.values

    params = dict(n_neighbors=150, weights='distance',metric='euclidean')
    evaluate_model(X, y, KNearestNeighborsModel, scale=True, pca=False, n_splits=5, params=params)
