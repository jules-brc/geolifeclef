
import numpy as np
import pandas as pd
from classifier import Classifier

import scipy.spatial.distance

class VectorModel(Classifier):

    """Simple vector model based on nearest-neighbors in the environmental
       space
    """
    def __init__(self, window_size=4):
        """
           :param window_size: the size of the pixel window to calculate the
            mean value for each layer of a tensor
        """
        self.train_vectors = None # vectors for the training dataset
        self.window_size = window_size

    def fit(self, dataset):

        """Builds for each point in the training set a K-dimensional vector,
           K being the number of layers in the env. tensor
           :param dataset: the GLCDataset training set
        """
        self.train_set = dataset
        self.train_vectors = dataset.tensors_to_vectors(self.window_size)

    def predict(self, dataset, ranking_size=30):

        """For each point in the dataset, returns the labels of the 30 closest points in the training set.
           It only keeps the closests training points of different species.
        """
        predictions = []
        test_vectors = dataset.tensors_to_vectors(self.window_size)

        for j in range(len(dataset)):

            vector_j = test_vectors[j]
            # euclidean distances from the test point j to all training points i
            distances = np.array([scipy.spatial.distance.euclidean(vector_j,vector_i)
                                  for vector_i in self.train_vectors
                                 ])
            #print(distances)
            # sorts by ascending distance and gives the predicted labels
            argsort = np.argsort(distances)
            y_predicted = []
            labels_found = set() #labels already returned
            for i in argsort:
                if len(y_predicted) >= ranking_size:
                    break
                label = self.train_set.get_label(i)
                if not label in labels_found:
                    y_predicted.append(label)
                    labels_found.add(label)
            predictions.append(y_predicted)
        return predictions

if __name__ == '__main__':

    from glcdataset import GLCDataset

    print("Vector model tested on train set\n")
    df = pd.read_csv('example_occurrences.csv', sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')
    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset = GLCDataset(df[['Longitude','Latitude']], df['glc19SpId'],
                             scnames=df[['glc19SpId','scName']],patches_dir='example_envtensors/0')

    vectormodel = VectorModel(window_size=65)

    vectormodel.fit(glc_dataset)
    predictions = vectormodel.predict(glc_dataset)
    scnames = vectormodel.train_set.scnames
    for idx in range(4):

        y_predicted = predictions[idx]
        print("Occurrence:", vectormodel.train_set.data.iloc[idx].values)
        print("Observed specie:", scnames.iloc[idx]['scName'])
        print("Predicted species, ranked:")

        print([scnames[scnames.glc19SpId == y]['scName'].iloc[0] for y in y_predicted[:10]])
        print('\n')

    print("Top30 score:",vectormodel.top30_score(glc_dataset))
    print("MRR score:", vectormodel.mrr_score(glc_dataset))
    print("Cross validation score:", vectormodel.cross_validation(glc_dataset, 4, shuffle=False, evaluation_metric='mrr'))
