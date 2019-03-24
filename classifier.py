from glcdataset import GLCDataset
import numpy as np

class Classifier(object):

    """Generic class for a classifier
    """
    def __init__(self):

        self.train_set = None
        pass

    def fit(self, dataset):
        """Trains the model on the dataset
           :param dataset: the GLCDataset training set
        """
        raise NotImplementedError("fit not implemented!")

    def predict(self, dataset, ranking_size=30):
        """Predict the list of labels most likely to be observed
           for the data points given
        """
        raise NotImplementedError("predict not implemented!")

    def mrr_score(self, dataset):
        """Computes the mean reciprocal rank from a test set provided:
           It finds the inverse of the rank of the actual class along
           the predicted labels for every row in the test set, and
           calculate the mean.
           :param dataset: the test set
           :return: the mean reciprocal rank, from 0 to 1 (perfect prediction)
        """
        predictions = self.predict(dataset)
        mrr = 0.
        for idx,y_predicted in enumerate(predictions):
            try:
                rank = y_predicted.index(dataset.get_label(idx))
                mrr += 1./(rank+1)
            except ValueError: # the actual specie is not predicted
                mrr += 0.
        return 1./len(dataset)* mrr

    def top30_score(self, dataset):
        """It is the accuracy based on the first 30 answers:
           The mean of the function scoring 1 when the good species is in the 30
           first answers, and 0 otherwise, over all test test occurences.
        """
        predictions = self.predict(dataset)
        predictions = [y_predicted[:30] for y_predicted in predictions] # keep 30 first results
        top30score = 0.
        for idx,y_predicted in enumerate(predictions):
            top30score += (dataset.get_label(idx) in y_predicted)

        return 1./len(dataset)* top30score

    def cross_validation(self, dataset, n_folds, shuffle=True, evaluation_metric='top30'):
        # NEEDS DEBUGGING !!!
        """Cross validation prodedure to evaluate the classifier
           :param n_folds: the number of folds for the cross validation
           :idx_permutation: the permutation over indexes to use before cross validation
           :evaluation_metric: the evaluation metric function
           :return: the mean of the metric over the set of folds
        """
        if evaluation_metric == 'top30':
            metric = self.top30_score
        elif evaluation_metric == 'mrr':
            metric = self.mrr_score
        else:
            raise Exception("Evaluation metric is not known")
        if shuffle:
            idx_random = np.random.permutation(len(dataset))
        else:
            idx_random = np.arange(len(dataset))
        fold_size = len(dataset)//n_folds
        idx_folds = []

        # split the training data in n folds
        for k in range(0,n_folds-1):
            idx_folds.append(idx_random[fold_size*k : fold_size*(k+1)])
        idx_folds.append(idx_random[fold_size*(n_folds-1): ])

        # for each fold:
        # train the classifier on all other fold
        # validation score on the current fold
        scores = []
        for k in range(0,n_folds):

            idx_train = [i for idx_fold in idx_folds[:k]+idx_folds[k+1:] for i in idx_fold]
            idx_test = [i for i in idx_folds[k]]

            train_set = GLCDataset(dataset.data.iloc[idx_train], dataset.labels.iloc[idx_train], None, dataset.patches_dir)
            test_set = GLCDataset(dataset.data.iloc[idx_test], dataset.labels.iloc[idx_test], None, dataset.patches_dir)
            self.fit(train_set)
            scores.append(metric(test_set))
        print(scores)
        return np.mean(scores)
