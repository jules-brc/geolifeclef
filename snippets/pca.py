
if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from glcdataset import build_environmental_data
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

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

    # Reduce the data using principal component analysis, finding the best number
    # of components to retain a good amount of variance
    pca = PCA(n_components='mle', svd_solver='full')
    X_pca = pca.fit_transform(X)
    print("Principal component analysis test")
    print(f'Number of components: {pca.n_components_}')
    print(f'Explained variance ratio of each components:\n{pca.explained_variance_ratio_}\n')
    print(f'Transformed dataset:\n{X_pca}')

