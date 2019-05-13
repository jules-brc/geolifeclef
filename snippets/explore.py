# Load environmental data, preview the dataset, describe the dataset, and plot
# a correlation matrix

if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display

    from glcdataset import build_environmental_data
    # for reproducibility
    np.random.seed(42)

    # working on a subset of Pl@ntNet Trusted: 5000 occurrences
    df = pd.read_csv('example_occurrences.csv',
                 sep=';', header='infer', quotechar='"', low_memory=True)

    df = df[['Longitude','Latitude','glc19SpId','scName']]\
           .dropna(axis=0, how='all')\
           .astype({'glc19SpId': 'int64'})

    # target pandas series of the species identifiers
    target_df = df['glc19SpId']

    # correspondence table between ids and the species taxonomic names
    # (Taxref names with year of discoverie)
    taxonomic_names = pd.read_csv('~/dev/geolifeclef/data/occurrences/taxaName_glc19SpId.csv',
                                  sep=';',header='infer', quotechar='"',low_memory=True)

    # building the environmental data
    env_df = build_environmental_data(df[['Latitude','Longitude']],patches_dir='example_envtensors')

    # preview data
    display(env_df.head(5))

    # describe data
    # TODO : look at which variables are numbers, which are categorical
    display(env_df.describe())

    # show the correlations of the individual features

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10,10))
    corr = env_df.corr()
    corr.index = env_df.columns
    sns.heatmap(corr, annot=True, cmap='RdYlGn')
    plt.title("Correlation Heatmap", fontsize=16)
    plt.show()

    # Use TSNE to visualize data in two-dimensional space
    from sklearn import datasets
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    digits = datasets.load_digits()
    data = digits.data
    projection = TSNE().fit_transform(data)
    plt.scatter(*projection.T, **plot_kwds)
