import numpy as np
import pandas as pd

from classifier import Classifier

class RandomModel(Classifier):

    """Simple vector model based on nearest-neighbors in the environmental
       space
    """
    def __init__(self):

        pass

    def fit(self, dataset):

        self.train_set = dataset
        self.all_labels = pd.Series(dataset.labels.unique())

    def predict(self, dataset, ranking_size=30):

        predictions = []
        for j in range(len(dataset)):
            y_predicted = list (np.random.choice(self.all_labels.values, size=ranking_size))
            predictions.append(y_predicted)

        return predictions

if __name__ == '__main__':

    from glcdataset import GLCDataset

    print("Random model tested on train set\n")
    df = pd.read_csv('example_occurrences.csv', sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')
    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset = GLCDataset(df[['Longitude','Latitude']], df['glc19SpId'],
                             scnames=df[['glc19SpId','scName']],patches_dir='example_envtensors/0')

    randommodel = RandomModel()

    randommodel.fit(glc_dataset)
    predictions = randommodel.predict(glc_dataset)
    scnames = randommodel.train_set.scnames
    for idx in range(4):

        y_predicted = predictions[idx]
        print("Occurrence:", randommodel.train_set.data.iloc[idx].values)
        print("Observed specie:", scnames.iloc[idx]['scName'])
        print("Predicted species, ranked:")

        print([scnames[scnames.glc19SpId == y]['scName'].iloc[0] for y in y_predicted[:10]])
        print('\n')

    print("Top30 score:",randommodel.top30_score(glc_dataset))
    print("MRR score:", randommodel.mrr_score(glc_dataset))
    print("Cross validation score:", randommodel.cross_validation(glc_dataset, 4, shuffle=False, evaluation_metric='mrr'))

