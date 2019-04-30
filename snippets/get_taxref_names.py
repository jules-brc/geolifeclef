
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
    classifier = NearestCentroidModel(metric='euclidean')
    classifier.fit(X_train,y_train)
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')
    print('Params:',classifier.get_params())

    print("Example of predict proba:")
    print(f"occurrence:\n{X_test[12]} -> {get_taxref_names(y_test[12],taxonomic_names)}")

    y_pred, y_probas = classifier.predict(X_test[12].reshape(1,-1), return_proba=True)
    print(f'predicted labels:\n{y_pred}')
    print(f'-> {get_taxref_names(y_pred, taxonomic_names)}')
    print(f'predicted probas:\n{y_probas}')
    print(f'sum of probas: {np.sum(y_probas)}')
    print(f'predicted probas:\n{y_probas}')
