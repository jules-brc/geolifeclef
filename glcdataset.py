import os
import pandas as pd
import numpy as np

# Scikit-learn validation tools
from sklearn.utils.validation import check_is_fitted

# from torch.utils.data import Dataset

# THE CLASS GLCDataset IS NOT USED ANYMORE
# Maybe we'll use it when working with convolutional neural networks with PyTorch
# class GLCDataset(Dataset):

#     """Represents a dataset from the data of GLC 2019
#     """
#     def __init__(self, data, labels, patches_dir, scnames=None):
#         """
#         Creates a dataset object for the species occurrences, the associated
#         environmental patches, and their labels.
#         OFFLINE USE : The env. patches must already be saved in the patches_dir directory

#         :param dataframe: pandas dataframe read from a csv file containing occurrences data
#         :param patches_dir: the root directory containing the saved env. tensors
#         """
#         self.data = data # occurences (lat,lng)
#         self.labels = labels # labels (species id)
#         # correspondence between species ids and scientific names, if provided
#         if scnames is not None:
#             self.scnames = scnames
#         self.patches_dir = patches_dir

#     def __len__(self):
#         return len(self.labels)

#     # Leaves the reading of env. patches to __getitem__. This is memory
#     # efficient because all the patches are not stored in memory all at once
#     # but loaded when it is required
#     def __getitem__(self, idx):

#         """:param idx: the integer index
#            :return: a dictionary containing lat, lng, and patch
#         """
#         # get original index, before possible dropped rows
#         true_idx = self.data.index[idx]
#         # get the name of the file
#         patch_name = self.patches_dir + '/' + str(true_idx)+'.npy'
#         # reads the file
#         patch = np.load(patch_name)
#         lat,lng = self.data.loc[true_idx,'Longitude'],self.data.loc[true_idx,'Latitude']

#         return {'lat': lat,'lng': lng,'patch': patch}

#     def get_label(self, idx):
#         return int(self.labels.iloc[idx])

#     def tensors_to_vectors(self, window_size=4, repr_space='env'):

#         """Builds a vector out of a env. tensor for each datapoint
#            :param window_size: the size of the pixel window to calculate the mean value
#                for each layer of a tensor
#            :param repr_repr_space: the variables to use to build the vector: default is 'env' so the vectors
#                are build from the environmental patches. Others are 'loc' and 'loc+env', to use
#                location and location+environment

#            :return: the list of vectors for each datapoint
#         """
#         vectors = []
#         for i in range(self.__len__()):

#             item = self[i]
#             lat,lng,patch = item['lat'],item['lng'],item['patch']
#             if repr_space in ('env','loc+env'):

#                 vect = np.empty(patch.shape[0], dtype=np.float64)
#                 for k,layer in enumerate(self[i]['patch']):
#                     # get the mean value in the pixel window around the center
#                     x_c, y_c = layer.shape[0]//2, layer.shape[1]//2
#                     vect[k] = layer[x_c - window_size//2: x_c + window_size//2,
#                                     y_c - window_size//2: y_c + window_size//2].mean()
#                 if repr_space == 'env':
#                     vectors.append(vect)
#                 else: # add lat,lng
#                     vectors.append(np.hstack(([lat,lng],vect)))
#             elif repr_space == 'loc':
#                 vectors.append(np.array([lat,lng], dtype=np.float64))
#             else:
#                 raise Exception("Unknown parameter repr_space to build the vectors")
#         return vectors


def filter_by_identification_confidence(df, threshold, reset_index=False):

  """Filter rows from a dataset where the specie identification confidence score is
     above the thresold.
     :param df:
     The dataframe of Plantnet Queries with species identification scores to
     filter from.
     :param threshold:
     Filter above this value, between 0. and 1.
     :param reset_index:
     Wheter or not to reset the index

     :return:
     The filtered dataframe
  """
  df_above = df[df['FirstResPLv2Score'] > threshold]
  if reset_index:
    df_above = df_above.reset_index(drop=True)

  return df_above

def filter_by_class_frequency(df, y_column='glc19SpId', threshold=1, reset_index=False):

  """Filter rows from the dataset where the label frequency is above the
     threshold.
  """
  res_df = df.groupby(y_column).filter(lambda x: len(x) > threshold)
  return res_df


# create an encoder object and fit it the the column data
# Different values are found at fit step
# We decide to ignore unknown data, by setting all different values to zero

def encode_categorical_feature_fit(df, column='clc', encoder=None):

    """Encode the categorical feature with one-hot encoding

    :param column: the feature column name to encode
    :param value_range: range of values for the feature
    :param encoder: the encoder to fit on the data
    """
    res_df = df.copy()
    X_col = df[column].values.reshape(-1,1)
    X_onehot = encoder.fit_transform(X_col)
    print('New features with one-hot encoding:', X_onehot.shape[1])

    # build new dataframe with one hot features instead of the column
    res_df = res_df.drop(column, axis=1)

    for i,onehot_feat in enumerate(encoder.get_feature_names(input_features=[column])):
        res_df[onehot_feat] = X_onehot[:,i]

    return res_df

def encode_categorical_feature_transform(df, column='clc', encoder=None):

    """Encode the categorical feature with one-hot encoding

    :param column: the feature column name to encode
    :param value_range: range of values for the feature
    :param encoder: the encoder to fit on the data
    """
    res_df = df.copy()
    X_col = df[column].values.reshape(-1,1)
    X_onehot = encoder.transform(X_col)

    # build new dataframe with one hot features instead of the column
    res_df = res_df.drop(column, axis=1)
    for i,onehot_feat in enumerate(encoder.get_feature_names(input_features=[column])):
        res_df[onehot_feat] = X_onehot[:,i]

    return res_df

def scale_features_fit(df, columns=['alti'], scaler=None):
    res_df = df.copy()
    X_cols = df[columns].values
    X_scaled = scaler.fit_transform(X_cols)

    # build new dataframe
    for i,col in enumerate(columns):
        res_df[col] = X_scaled[:,i]

    return res_df

def scale_features_transform(df, columns=['alti'], scaler=None):

    res_df = df.copy()
    X_cols = df[columns].values
    X_scaled = scaler.transform(X_cols)

    # build new dataframe
    for i,col in enumerate(columns):
        res_df[col] = X_scaled[:,i]

    return res_df

def pca_fit(df, pca=None):

    X = df.values
    X_pca = pca.fit_transform(X)

    # build new dataframe
    n_components = pca.n_components_
    features = [f'pca_{i}' for i in range(n_components)]

    res_df = pd.DataFrame(columns=features, dtype='float64')
    for i,feat in enumerate(features):
        res_df[feat] = X_pca[:,i]

    return res_df

def pca_transform(df, pca=None):

    X = df.values
    X_pca = pca.transform(X)

    # build new dataframe
    n_components = pca.n_components_
    features = [f'pca_{i}' for i in range(n_components)]

    res_df = pd.DataFrame(columns=features, dtype='float64')
    for i,feat in enumerate(features):
        res_df[feat] = X_pca[:,i]

    return res_df

def get_taxref_names(y, taxonomic_names):

    """Returns the taxonomic names which corresponds to species ids in y
       :param y:
       a label, a list or array of labels, or a matrix of labels
       (2-dimensional numpy array)
       :taxonomic_names: a dataframe containing two columns, 'glc19SpId' and
       'taxaName' giving the correspondences between species ids and names

       :return: a list of corresponding names, or a list of list
    """
    # dictionary correspondence between species ids and names
    dict_names = dict(zip(taxonomic_names['glc19SpId'], taxonomic_names['taxaName']))

    y = np.asarray(y)
    if len(y.shape)< 2:
        y = y.reshape(1,-1)
    try:
        return [[dict_names[spid] for spid in array_spids] for array_spids in y]
    except KeyError as e:
      raise Exception(f"Specie id has no taxonomic names: {e.args[0]}")

if __name__ == '__main__':

    from IPython.display import display

    # random seed for reproducibility
    np.random.seed(42)

    # working on a subset of Pl@ntNet Trusted: 2500 occurrences
    df = pd.read_csv('example_occurrences.csv',
                     sep=';', header='infer', quotechar='"', low_memory=True)

    # print(len(df), 'occurences in the dataset')
    # df_above_99 = filter_by_dentification_confidence(df, 0.99)
    # print(len(df_above_99), 'occurences above 99%% identification certainty')


    df_freq = filter_by_class_frequency(df,'glc19SpId',5)

    assert(len(df)>len(df_freq))
    assert (len(df_freq.groupby('glc19SpId').filter(lambda x: len(x) <= 5))==0)
    df = df[['Longitude','Latitude','glc19SpId','scName']]\
          .dropna(axis=0, how='all')\
          .astype({'glc19SpId': 'int64'})

    # target pandas series of the species identifiers (there are 505 labels)
    target_df = df['glc19SpId']
    # correspondence table between ids and the species taxonomic names
    # (Taxref names with year of discoverie)
    taxonomic_names = pd.read_csv('~/dev/geolifeclef/data/occurrences/taxaName_glc19SpId.csv',
                                  sep=';',header='infer', quotechar='"',low_memory=True)

    # print(len(target_df.unique()), 'number of species\n')
    # display(df.head(5))
    # duplicated_df = df[df.duplicated(subset=['Latitude','Longitude'],keep=False)]
    # print(f'There are {len(duplicated_df)} entries ({len(duplicated_df)/len(df)*100}%%) observed at overlapping locations:')
    # display(duplicated_df.head(5))

    # env_df = build_environmental_data(df[['Latitude','Longitude']],patches_dir='example_envtensors')

    # X = env_df.values
    # y = target_df.values

    # from sklearn.preprocessing import OneHotEncoder

    # # create an encoder object and fit it the the column data
    # encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # env_df = encode_categorical_feature_fit(env_df, encoder=encoder)

    # display(env_df)


def build_environmental_data(df, patches_dir, mean_window_size=None):

    """This function builds a dataset containing all the latitude,
       longitude, and vectors build from the environmental tensors associated
       saved in a directory.
       Used to fit to Scikit-Learn models.
       If the environmental tensors are just row vectors (i.e the env. variables
       values at the location) then it loads them in a new dataframe.
       Otherwise, the tensors are flattened as long row vectors;
       that's when the tensors are the env. variables values around the location.

       :param df: the locations dataframe, containing (Latitude,Longitude)
           columns
       :param patches_dir: the directory where the env. patches are saved
       :param mean_window_size: if not None, takes the mean value of each
       raster on the provided window size

       :return: a new dataframe containing the locations concatenated with
           their env. vectors
    """
    # import the names of the environmental variables
    from environmental_raster_glc import raster_metadata

    env_array = list()
    # number of values per channel, 1 if patches are vector
    n_features_per_channel = 1
    for idx in range(len(df)):

        # get the original index used to write the patches on disk
        true_idx = df.index[idx]
        # find the name of the file
        patch_name = patches_dir + '/' + str(true_idx)+'.npy'
        # reads the file
        patch = np.load(patch_name)
        # build the row vector
        lat, lng = df.loc[true_idx,'Longitude'], df.loc[true_idx,'Latitude']

        if mean_window_size:
            try:
                patch = np.array([ ch[ch.shape[0]//2 - mean_window_size//2:
                                      ch.shape[0]//2 + mean_window_size//2,
                                      ch.shape[1]//2 - mean_window_size//2:
                                      ch.shape[1]//2 + mean_window_size//2
                                     ].mean() for ch in patch
                                 ])
            except IndexError:
                raise Exception("Channels don't have two dimensions!")
        else:
            if len(patch.shape) > 1:
                n_features_per_channel = patch[0].shape[0]*patch[0].shape[1]
            elif len(patch.shape) ==2 :
                raise Exception("Channel of dimension one: should only be a scalar\
                                 or a two dimensional array")
        # flatten to build row vector
        env_array.append(np.concatenate(([lat,lng],patch),axis=None))

    rasters_names = sorted(raster_metadata.keys())
    if n_features_per_channel == 1:
        header_env = rasters_names
    else:
        header_env = []
        for name in rasters_names:
            header_env.extend([name+f'__{i}' for i in range(n_features_per_channel)])
    header = ['Latitude','Longitude'] + header_env
    env_df = pd.DataFrame(env_array, columns=header, dtype='float64')
    return env_df
