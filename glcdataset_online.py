import pandas as pd
import numpy as np

from glcdataset import GLCDataset,TAXANAME_CSV
#### BUGGY #####

class GLCDatasetOnline(GLCDataset):

    """Represents a dataset from the data of GLC 2019
    """
    def __init__(self, data, labels, scnames, patch_extractor):
        """
        Creates a dataset object for the species occurrences, the associated
        environmental patches, and their labels.
        OFFLINE USE : The env. patches must already be saved in the patches_dir directory

        :param dataframe: pandas dataframe read from a csv file containing occurrences data
        :param patch_extractor: the patch extractor to extract the environmental patches
           on the go from occurrences
        """
        self.data = data #occurences (lat,lng)
        self.labels = labels #labels (species id)
        # correspondence between species ids and scientific names, if provided
        if scnames is not None:
            self.scnames = scnames
        self.patch_extractor = patch_extractor

    # Leaves the reading of env. patches to __getitem__. This is memory
    # efficient because all the patches are not stored in memory at once but
    # read as required.

    def __getitem__(self, idx):
        """:param idx: the integer index
           :return: a dictionary containing lat, lng, and patch
        """
        true_idx = self.data.index[idx]
        # Reads the env tensor from the patch extractor
        lat,lng = self.data.loc[true_idx,'Longitude'],self.data.loc[true_idx,'Latitude']
        patch = self.patch_extractor[lat,lng]

        return {'lat': lat,'lng': lng,'patch': patch}

if __name__ == '__main__':

    # Test
    from environmental_raster_glc import PatchExtractor
    from environmental_raster_glc import raster_metadata

    # building the patch extractor
    # some channels are set to be avoided in the 'exception' list
    ext = PatchExtractor('../data/rasters_GLC19', size=64, verbose=False)
    exception = ('alti','proxi_eau_fast')
    for channel in raster_metadata:
        if channel not in exception:
            ext.append(channel)

    df = pd.read_csv('example_occurrences.csv', sep=';', header='infer', quotechar='"', low_memory=True)
    df = df[['Longitude','Latitude','glc19SpId','scName']]
    if not (len(df.dropna(axis=0, how='all')) == len(df)):
        raise Exception("nan lines in dataframe, cannot build the dataset!")

    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset = GLCDataset(df[['Longitude','Latitude']], df['glc19SpId'],
                             scnames=df[['glc19SpId','scName']],patch_extractor=ext)

    print(len(glc_dataset), 'occurrences in the dataset')
    print(len(glc_dataset.labels.unique()), 'number of species\n')

    # for i in range(len(glc_dataset)):

    #     sample = glc_dataset[i]
    #     print(i, ": lat %.2f, lng %.2f,"%(sample['lat'],sample['lng']),"patch_size:",sample['patch'].shape)
    #     if i == 10:
    #         break
    print(pd.concat([glc_dataset.data, glc_dataset.scnames], axis=1).head(15))

    # The dataset can now be used in a DataLoader
    # data_loader = torch.utils.data.DataLoader(glc_dataset, shuffle=True, batch_size=2
