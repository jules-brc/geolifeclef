# -*- coding: latin-1 -*-
## @author : Jules Bourcier (YaJu SU Team)

## THIS SCRIPT IS A DEBUGGED VERSION OF THE ORIGINAL SCRIPT FROM MAXIMILIENSE
## there was a bug in the way the rasters where stored on disk: some raster
## where re-written on rasters from former points due to the batch_size and
## modulo_disk parameters used to build the path to the files
## it cause erroneous rasters recuperation to build our machine learning
## datasets. Now all the rasters are saved in the same directorie, their name
## is the original index of the example in the dataframe.

import os
import sys
sys.path.extend(['..', '../maximiliense'])
import pandas as pd
import numpy as np

if __name__ == '__main__':

    from IPython.display import display

    from glcdataset import filter_by_class_frequency
    from glcdataset import encode_categorical_feature_fit
    from glcdataset import encode_categorical_feature_transform
    from glcdataset import scale_features_fit
    from glcdataset import scale_features_transform
    from glcdataset import pca_fit
    from glcdataset import pca_transform

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # for reproducibility
    np.random.seed(42)

    # read environmental data
    env_df = pd.read_csv('../data/pl_trusted_size1.csv',
                     sep=';', header='infer', quotechar='"')

    # read the challenge test data
    test_df = pd.read_csv('../data/test_size1.csv',
                          sep=';', header='infer', quotechar="'")

    # print('filtering by class frequency')
    # env_df = filter_by_class_frequency(env_df, y_column='glc19SpId',
    #                                    threshold=1, reset_index=True)

    # get species labels aside : target pandas series of the species identifiers
    target_env_df = env_df['glc19SpId'].reset_index(drop=True)
    env_df = env_df.drop('glc19SpId', axis=1)

    target_test_df = test_df['glc19TestOccId'].reset_index(drop=True)
    test_df = test_df.drop('glc19TestOccId', axis=1)

    # One-hot encode categorical feature "clc" or remove the feature

    # print('one-hot encoding categorical features')
    # encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')

    # env_df = encode_categorical_feature_fit(env_df, column='clc',
    #                                             encoder=encoder)
    # test_df = encode_categorical_feature_transform(test_df, column='clc',
    #                                                   encoder=encoder)
    # or remove the feature
    # if 'clc' in env_df.columns:
    env_df = env_df.drop('clc', axis=1)
    test_df = test_df.drop('clc', axis=1)

    # Standardize numerical/ordinal features, not the categorical ones

    print('scaling other features')
    scaler = StandardScaler()
    env_df = scale_features_fit(env_df, columns=[col for col in list(env_df.columns)\
                                                if not col.startswith('clc')],
                                                scaler=scaler)

    test_df = scale_features_transform(test_df, columns=[col for col in list(test_df.columns)\
                                                if not col.startswith('clc')],
                                                scaler=scaler)

    # X_test = test_df.values
    # y_test = target_test_df.values

    # Do a PCA for dimensionality reduction, keep 99% of explained variance

    print('PCA dimensionality reduction')
    pca = PCA(n_components=0.99, whiten=False)

    env_df = pca_fit(env_df, pca=pca)
    test_df = pca_transform(test_df, pca=pca)

    # concatenate column for the specie's label
    env_df = pd.concat((env_df, target_env_df), axis=1)
    test_df = pd.concat((test_df, target_test_df), axis=1)

    display(env_df.head(5))
    display(test_df.head(5))

    # save the preprocessed data to csv files
    env_df.to_csv('../data/pl_trusted_size1_noclc_scaled_pca.csv', sep=';', index=False, quotechar='"')
    test_df.to_csv('../data/test_size1_noclc_scaled_pca.csv', sep=';', index=False, quotechar='"')
