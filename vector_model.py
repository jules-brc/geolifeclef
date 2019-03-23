
import numpy as np
import pandas as pd

class Classifier(object):

    """Generic class for a classifier
    """
    def __init__(self, trainset):
        """
           :param trainset: the GLCDataset training set for the model
        """
        self.trainset = None

    def fit(self, dataset):
        """Trains the model on the dataset
           :param dataset: the GLCDataset training set
        """
        raise NotImplementedError("fit not implemented")

    def predict(self, dataset, rank_size=100):
        """Predict the list of labels most likely to be observed
           for the data points given
        """
        raise NotImplementedError("predict not implemented")

    def mrr_score(self, dataset):
        """Computes the mean reciprocal rank from a test set provided,
           which means we find the inverse of the rank of the actual class along
           the predicted labels for every row in the test set, and
           calculate the mean.

           :param dataset: the test set
           :return: the mean reciprocal rank, from 0 to 1 (perfect prediction)
        """
        predictions = self.predict(dataset)
        mrr = 0.
        for i,prediction in enumerate(predictions):
            try:
                rank = prediction.index(dataset.get_label(i))
                mrr += 1./rank
            except ValueError: # the actual specie is not returned
                mrr += 0.
        return 1./len(dataset)* mrr

    def cross_validation(self, dataset, n_folds):
        pass

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

        self.trainset = dataset
        self.train_vectors = dataset.tensors_to_vectors(self.window_size)

    def predict(self, dataset, ranking_size=100):

        predictions = []
        vectors_test = dataset.tensors_to_vectors(self.window_size)

        for j in range(len(dataset)):

            vector_j = vectors_test[j]

            # euclidean distances from the test point j to all training points i
            distances = np.array([np.sqrt(np.sum((vector_j-vector_i)**2))
                                   for vector_i in self.train_vectors
                                   ])

            # sorts by ascending distance and gives the predicted labels
            argsort = np.argsort(distances)

            toplabels = np.empty(rank_size, dtype=np.int32)
            labels_found = set() #labels already returned

            for i in argsort:

                label = self.trainset.get_label(i)
                if not label in labels_found:
                    toplabels[n_labels] = label
                    n_labels += 1
                    labels_found.add(label)

                if n_labels >= ranking_size:
                    break

            predictions.append(toplabels)
        return predictions

if __name__ == '__main__':

    from glcdataset import GLCDataset

    print("Vector model tested on train set")
    glc_dataset = GLCDataset('example_occurrences.csv', 'example_envtensors/0')
    vectormodel = VectorModel()

    vectormodel.fit(glc_dataset)
    predictions = vectormodel.predict(glc_dataset)

    scnames = vectormodel.trainset.scnames #scientific names of species
    for i in [0,1,2]:

        predict = predictions[i]

        print("Point in space:", vectormodel.trainset[i]['lat'],vectormodel.trainset[i]['lng'])
        print("Observed specie:", scnames.loc[i,'scName'])
        print("Predicted species (ranked):", [scnames[scnames.glc19SpId in predict]['scName']])
    #print(vectormodel.vectors)
