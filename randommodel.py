import numpy as np
import pandas as pd

from classifier import Classifier

class RandomModel(Classifier):

    """Simple vector model based on nearest-neighbors in the environmental
       space
    """
    def __init__(self):
        """Does nothing
        """
        pass

    def fit(self, dataset):
        """Does nothing
        """
        self.trainset = dataset
    def predict(self, dataset, ranking_size=100):

        predictions = []
        all_labels = self.trainset.labels

        for j in range(len(dataset)):
            toplabels = np.random.choice(all_labels, size=ranking_size)
            predictions.append(toplabels)

        return predictions

if __name__ == '__main__':
    pass
