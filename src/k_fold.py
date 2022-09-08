from math import floor
import numpy as np

class KFold:
    def __init__(self, data, number_folds, randomize=True):
        data = np.copy(data)
        if randomize:
            np.random.shuffle(data)

        label_to_data = self._split_data_by_label(data)
        self._create_folds(label_to_data, number_folds)
    
    def test_train_split(self, test_fold):
        test = self.folds[test_fold]
        train = np.concatenate(list(self.folds[i] for i in range(self.number_folds) if i != test_fold))
        return test, train

    def _split_data_by_label(self, data):
        label_to_data = {}
        for instance in data:
            if label_to_data.get(instance[-1]) is None:
                label_to_data[instance[-1]] = []    
            label_to_data[instance[-1]].append(instance)
        return label_to_data
    
    def _create_folds(self,label_to_data, number_folds):
        self.folds = [[] for _ in range(number_folds)]
        self.number_folds = number_folds
        for _, labeled_data in label_to_data.items():
            i = 0
            for data in labeled_data:
                self.folds[i].append(data)
                i = (i + 1) % number_folds

        self.folds = [np.array(fold) for fold in self.folds]
