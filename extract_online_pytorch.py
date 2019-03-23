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
    patch_extractor = PatchExtractor('/data/rasters_GLC19', size=64, verbose=True)

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

