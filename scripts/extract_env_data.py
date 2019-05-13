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

from environmental_raster_glc import PatchExtractor, raster_metadata

# This script exports the environmental vectors on disk
# See the readme for how to use

def extract_environmental_data(dataset, rasters, destination=None, mean_window_size=None, patch_size=1, row_number_limit=None):

    """This function builds a dataset containing all the latitude,
       longitude, and vectors build from the environmental tensors associated
       saved in a directory, and save in optionnally in a csv file.
       Used to fit to Scikit-Learn models.
       If the environmental tensors are just row vectors (i.e the env. variables
       values at the location) then it loads them in a new dataframe.
       Otherwise, we either take the mean of the tensor values under a window
       parameter, or the tensors are flattened as long row vectors. This last
       option is very expensive in memory and will not work on dataset containing
       250k+ occurrences.

       :param df: the locations dataframe, containing Latitude,Longitude
           columns, and glc19SpId, the labels column.
       :param rasters: the directory where the rasters are located
       :param destination: an optional csv file where to save the data. The script
       takes quite some time so its useful to save the result in a file.

       :param mean_window_size: if not None, takes the mean value of each channel
       on under this window size
       :param patch_size: size of channels in the patches. 1 means channels are
       scalar values at this position, >1 means they are arrays around this
       position.
       :param row_number_limit: a max number of rows to extract. Default extract
       all the data.

       :return: a new dataframe containing the locations concatenated with
       their env. vectors
    """
    n_env_features = len(raster_metadata.keys())
    rasters_names = sorted(raster_metadata.keys())

    if patch_size==1 and mean_window_size:
        raise Exception('Patches are already vectors of scalars (size 1), cannot provide a window size')

    if patch_size==1 or mean_window_size:
        shape_env = (n_env_features)
    else:
        shape_env = n_env_features*patch_size*patch_size

    print('Will build row vectors of size',shape_env)


    # Reads the csv file containing the occurences
    df = pd.read_csv(dataset, sep=';', header='infer', quotechar='"')\
           .dropna(axis=0, how='all')

    #test data file: different label column name

    if 'glc19SpId' in df.columns:
        target_column = 'glc19SpId'
    elif 'glc19TestOccId' in df.columns:
        target_column = 'glc19TestOccId'
    else:
        raise Exception('Unknown target column in the data')

    df = df.astype({target_column:'int64'})

    # keep only columns required, to free up memory
    df = df[['Latitude','Longitude',target_column]]
    ext = PatchExtractor(rasters, size=patch_size, verbose=True)

    positions = []
    # exception = ('proxi_eau_fast','alti', 'clc') # add rasters that don't fit into memory
    exception = tuple()
    env_vectors = list()
    # number of values per channel, 1 if patches are vector
    n_features_per_channel = 1

    if not row_number_limit:
        row_number_limit = len(df)
    print('Starting')
    try:
        positions = list(zip(df.Latitude,df.Longitude))[:row_number_limit]
        print('Loading rasters and extract..')
        variables = []
        for raster in rasters_names:
            if raster in exception:
                continue
            ext.clean()
            ext.append(raster)
            variable = np.stack([ext[p] for p in positions])
            variables.append(variable)
        ext.clean()

        variables = np.concatenate(variables, axis=1)
        # the shape of variables is (batch_size, nb_rasters, size, size)
        print('Build env vectors..')
        # build env vector for each occurrence in the batch
        for p_idx, patch in enumerate(variables):

            if mean_window_size:
                patch = np.array([ch[ch.shape[0]//2 - mean_window_size//2:
                                  ch.shape[0]//2 + mean_window_size//2,
                                  ch.shape[1]//2 - mean_window_size//2:
                                  ch.shape[1]//2 + mean_window_size//2
                                 ].mean() for ch in patch
                                ])
            else:
                if len(patch.shape) > 1:
                    n_features_per_channel = patch[0].shape[0]*patch[0].shape[1]

            # flatten to build row vector
            lat,lng = positions[p_idx]
            env_vectors.append(np.concatenate(([lat,lng],patch),axis=None))

        print('Done! building dataframe')
    except MemoryError as e:
        raise e(f'Reached out of memory, was able to extract {len(env_vectors)} rows')

    if n_features_per_channel == 1:
        header_env = rasters_names
    else:
        header_env = []
        for name in rasters_names:
            header_env.extend([name+f'__{i}' for i in range(n_features_per_channel)])
    header = ['Latitude','Longitude'] + header_env

    env_df = pd.DataFrame(env_vectors, columns=header, dtype='float64')
    print('Saving on disk')

    # concatenate column for the specie's label
    target_df = df[target_column].reset_index(drop=True).loc[:row_number_limit]

    env_df = pd.concat((env_df, target_df), axis=1)
    if destination:
        env_df.to_csv(destination, sep=';', index=False, quotechar='"')

    return env_df

if __name__ == '__main__':

    from IPython.display import display

    df_size8_mean = extract_environmental_data('/Users/Jules/dev/geolifeclef/data/occurrences/Test.csv',
                                               '/Users/Jules/dev/geolifeclef/data/rasters_GLC19',
                                               '../data/test_size8_mean.csv',
                                               mean_window_size=8, patch_size=10)

    df_size1 = pd.read_csv('../data/test_size1.csv',sep=';', header='infer', quotechar='"')

    assert(len(df_size1)==len(df_size8_mean))
    assert(df_size1['glc19TestOccId'].equals(df_size8_mean['glc19TestOccId']))

    df_size8_mean['clc'] = df_size1['clc']
    df_size8_mean.to_csv('../data/test_size8_mean.csv', sep=';', index=False, quotechar='"')
