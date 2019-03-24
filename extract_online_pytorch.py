import torch
from torch.utils.data import Dataset

from environmental_raster_glc import PatchExtractor


class GeoLifeClefDataset(Dataset):
    def __init__(self, extractor, dataset, labels):
        self.extractor = extractor
        self.labels = labels
        self.dataset = dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tensor = self.extractor[self.dataset[idx]]
        return torch.from_numpy(tensor).float(), self.labels[idx]


if __name__ == '__main__':
    patch_extractor = PatchExtractor('../data/rasters_GLC19', size=64, verbose=True)

    patch_extractor.append('chbio_1')
    patch_extractor.append('text')
    # patch_extractor.add_all()

    # example of dataset
    dataset_list = [(43.61, 3.88), (42.61, 4.88), (46.15, -1.1), (49.54, -1.7)]
    labels_list = [0, 1, 0, 1]

    dataset_pytorch = GeoLifeClefDataset(patch_extractor, dataset_list, labels_list)

    print(len(dataset_pytorch), 'elements in the dataset')

    # dataset_pytorch can now be used in a data_loader
    data_loader = torch.utils.data.DataLoader(dataset_pytorch, shuffle=True, batch_size=2)

    for batch in data_loader:
        data, label = batch
        print('[batch, channels, width, height]:', data.size())
        print('[batch]:', label)
        print('*' * 5)


import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from environmental_raster_glc import PatchExtractor


class GLCDataset(Dataset):

    """GLCDataset represents a dataset from the data of GLC 2019
    """
    def __init__(self, dataset, patches_dir):

        """
        Creates a dataset object for the species occurences, the associated
        environmental patches, and their labels.
        The env. patches must already exist in the patches_dir directory

        :param dataset: the csv file of the occurences data
        :param patches_dir: the directory containing the .npy files of the
            environmental patches for each occurence in the dataset
        """
        # Reads the csv file containing the occurences,
        # separates occurences (Lat,Lng) and labels (glc19SpId)
        df = pd.read_csv(dataset, sep=';', quotechar='"', low_memory=True)
        df = df.dropna(axis=0, how='all')
        df.reset_index()
        self.labels = df.loc(:,['Longitude','Latitude','glc19SpId'])
        self.occurences = df.loc(:,['glc19SpId'])

        for idx, occurence in enumerate(df.iterrows())

    def __len__(self):
        return len(self.labels)


    # Leaves the reading of env. patches to __getitem__. This is memory
    # efficient because all the patches are not stored in memory at once but
    # read as required. (WORK IN PROGRESS)
    def __getitem__(self, idx):

        # reads the file with name the index in the complete dataframe, with
        # extension .npy

        patch_name = os.path.join(self.patches_dir,'0',self.occurences.loc[idx,'index']+'.npy')
        patch = numpy.load(patch_name)
        lat,lng = self.occurences.loc(idx,'Longitude'),self.occurences.loc(idx,'Latitude')

        return lat,lng,patch


if __name__ == '__main__':

    from environmental_raster_glc import PatchExtractor

    rasters_dir = '../data/rasters'

    ext = PatchExtractor(rasters, size=64, verbose=True)
    glc = GLCDataset()

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

#Csv for correspondence between species ids and Taxref names
TAXANAME_CSV = '../data/occurrences/taxaName_glc19SpId.csv'

class GLCDataset(Dataset):

    """Represents a dataset from the data of GLC 2019
    """
    def __init__(self, data_file, patches_dir):

        """
        Creates a dataset object for the species occurrences, the associated
        environmental patches, and their labels.
        OFFLINE USE : The env. patches must already be saved in the patches_dir directory

        :param data_file: the csv file of the occurrences data
        :param patches_dir: the directory containing the .npy files of the
            environmental patches for each occurence in the dataset
        """

        # Reads the csv file containing the occurrences,
        # separates occurrences (Lat,Lng) and labels (glc19SpId)
        df = pd.read_csv(dataset, sep=';', quotechar='"', low_memory=True)
        df = df.dropna(axis=0, how='all')
        self.data = df[['Longitude','Latitude']] #occurences (lat,lng)
        self.labels = df['glc19SpId'] #labels (species id)

        # correspondence between species ids and scientific names
        self.scnames = df[['glc19SpId','scName']]
        self.patches_dir = patches_dir

    def __len__(self):
        return len(self.labels)

    # Leaves the reading of env. patches to __getitem__. This is memory
    # efficient because all the patches are not stored in memory at once but
    # read as required. (WORK IN PROGRESS)
    def __getitem__(self, idx):

        """:param idx: the integer index
           :return: a dictionary containing lat, lng, and patch
        """
        true_idx = self.data.index[idx] #The real index
        # Reads the file with the name as the index plus extension .npy
        patch_name = os.path.join(self.patches_dir, str(true_idx)+'.npy')
        patch = np.load(patch_name)
        lat,lng = self.data.loc[true_idx,'Longitude'],self.data.loc[true_idx,'Latitude']

        return {'lat': lat,'lng': lng,'patch': patch}

    def get_label(self, idx):
        return int(self.labels.iloc[idx])

    def tensors_to_vectors(self, window_size=4):

        """Builds a vector out of a env. tensor for each datapoint
           :param window_size: the size of the pixel window to calculate the
            mean value for each layer of a tensor
           :return: the list of vectors for each datapoint
        """
        vectors = []
        for i in range(self.__len__()):

            patch = self[i]['patch']
            vectors.append(np.empty(len(patch), dtype=np.float64))

            for k,layer in enumerate(self[i]['patch']):
                x_c, y_c = layer.shape[0]//2, layer.shape[1]//2
                vectors[i][k] = patch[x_c - window_size //2: x_c + window_size //2,
                                      y_c - window_size //2, y_c + window_size //2].mean()
        return vectors

    def scname_correspondence(self, list_spids):
        """Returns the taxonomic names which corresponds to the list of
           species ids
           :param list_spids: the list of species ids
           :return: the list of taxonomic names
        """
        df = pd.read_csv(TAXANAME_CSV, sep=';', quotechar='"')
        return [df[df.glc19SpId == spid]['taxaName'].iloc[0] for spid in list_spids]

if __name__ == '__main__':

    # Test
    glc_dataset = GLCDataset('example_occurrences.csv', 'example_envtensors/0')

    print(len(glc_dataset), 'occurrences in the dataset')
    print(len(glc_dataset.labels.unique()), 'number of species')
    print("Head:")
    for i in range(len(glc_dataset)):

        sample = glc_dataset[i]
        print(i, ": lat lng:", sample['lat'],sample['lng'], "patch_size:",sample['patch'].shape)
        if i == 10:
            break

    some_ids = list(glc_dataset.labels.iloc[:10])
    scnames = glc_dataset.scname_correspondence(some_ids)
    print("Some ids:", some_ids)
    print("Taxref names:", scnames)
    # The dataset can now be used in a DataLoader
    # data_loader = torch.utils.data.DataLoader(glc_dataset, shuffle=True, batch_size=2)

