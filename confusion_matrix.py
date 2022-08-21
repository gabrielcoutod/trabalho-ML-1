import numpy as np

class ConfusionMatrix:

    def __init__(self, true_values, predicted_values):
        zip_values = zip(true_values, predicted_values)
        self.size = len(np.unique(true_values))
        self.matrix = np.zeros((self.size, self.size), dtype=int)
        self.binary = (self.size == 2)
        for true, predicted in zip_values:
            self.matrix[true, predicted] += 1

    def accuracy(self):
        total = self.matrix.sum()
        true = self.matrix.diagonal().sum()
        return true / total

    def recall(self, type="macro"):
        if self.binary:
            return self._binary_recall()
        if type=="macro":
            return self._macro_recall()
        if type=="micro":
            return self._micro_recall()
    
    def precision(self, type="macro"):
        if self.binary:
            return self._binary_precision()
        if type=="macro":
            return self._macro_precision()
        if type=="micro":
            return self._micro_precision()

    def f1_measure(self, type="macro"):
        if self.binary:
            return self._binary_f1_measure()
        if type=="macro":
            return self._macro_f1_measure()
        if type=="micro":
            return self._micro_f1_measure()

    def _binary_recall(self):
        return self.matrix[0,0] / (self.matrix[0,0] + self.matrix[0,1])

    def _micro_recall(self):
        return self.accuracy()

    def _macro_recall(self):
        recall = 0
        for i in range(self.size):
            recall += self.matrix[i,i] / self.matrix[i,:].sum()

        return recall / self.size

    def _binary_precision(self):
        return self.matrix[0,0] / (self.matrix[0,0] + self.matrix[1,0])

    def _micro_precision(self):
        return self.accuracy()
    
    def _macro_precision(self):
        precision = 0
        for i in range(self.size):
            precision += self.matrix[i,i] / self.matrix[:,i].sum()

        return precision / self.size

    def _binary_f1_measure(self):
        precision = self._binary_precision()
        recall = self._binary_recall()
        f1 = (2*precision*recall)/(precision+recall)
        return f1

    def _micro_f1_measure(self):
        precision = self._micro_precision()
        recall = self._micro_recall()
        f1 = (2*precision*recall)/(precision+recall)
        return f1

    def _macro_f1_measure(self):
        precision = self._macro_precision()
        recall = self._macro_recall()
        f1 = (2*precision*recall)/(precision+recall)
        return f1
