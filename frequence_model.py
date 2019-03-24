import numpy as np
import pandas as pd
from classifier import Classifier

class FrequenceModel(Classifier):

    """Simple vector model based on nearest-neighbors in the environmental
       space
    """
    def __init__(self, window_size=4):
        """
           :param window_size: the size of the pixel window to calculate the
            mean value for each layer of a tensor
        """
         # species ranked from most to least common in the training set
        self.all_labels_by_frequency = None

    def fit(self, dataset):
        # NEEDS DEBUGGING maybe
        self.train_set = dataset
        all_labels, counts = np.unique(dataset.labels.values, return_counts=True)
        print(np.argsort(counts))
        self.all_labels_by_frequency = pd.Series(all_labels[np.argsort(counts)])

    def predict(self, dataset, ranking_size=30):

        predictions = []
        for j in range(len(dataset)):

            y_predicted = list (self.all_labels_by_frequency[:ranking_size] )
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

    frequencemodel = FrequenceModel()

    frequencemodel.fit(glc_dataset)
    predictions = frequencemodel.predict(glc_dataset)
    scnames = frequencemodel.train_set.scnames
    for idx in range(4):

        y_predicted = predictions[idx]
        print("Occurrence:", frequencemodel.train_set.data.iloc[idx].values)
        print("Observed specie:", scnames.iloc[idx]['scName'])
        print("Predicted species, ranked:")

        print([scnames[scnames.glc19SpId == y]['scName'].iloc[0] for y in y_predicted[:10]])
        print('\n')

    print("Top30 score:",frequencemodel.top30_score(glc_dataset))
    print("MRR score:", frequencemodel.mrr_score(glc_dataset))
    print("Cross validation score:", frequencemodel.cross_validation(glc_dataset, 4, shuffle=False, evaluation_metric='mrr'))
