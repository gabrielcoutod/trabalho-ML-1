# Aqui importamos os módulos necessários para os testes
from pytest import fixture
import numpy as np
from confusion_matrix import ConfusionMatrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# 4 3
# 2 2
true_bin = np.array([0,0,1,1,0,0,1,1,0,0,0])
pred_bin = np.array([0,1,0,1,1,1,0,1,0,0,0])

# 1 1 1
# 2 2 2
# 3 3 3
true_multi = np.array([0,1,1,2,2,2,0,0,1,1,1,1,2,2,2,2,2,2])
pred_multi = np.array([0,1,1,2,2,2,1,2,0,0,2,2,0,0,0,1,1,1])

@fixture(scope="module")
def binary_matrix():
    return ConfusionMatrix(true_bin, pred_bin, 2)
    
@fixture(scope="module")
def multiclass_matrix():
    return ConfusionMatrix(true_multi, pred_multi, 3)

def test_binary_matrix(binary_matrix):
    matrix = confusion_matrix(true_bin, pred_bin)
    assert np.array_equal(binary_matrix.matrix, matrix)

def test_binary_accuracy(binary_matrix):
    accuracy = accuracy_score(true_bin, pred_bin)
    assert binary_matrix.accuracy() == accuracy

def test_binary_recall(binary_matrix):
    recall = recall_score(true_bin, pred_bin)
    assert binary_matrix.recall() == recall

def test_binary_precision(binary_matrix):
    precision = precision_score(true_bin, pred_bin)
    assert binary_matrix.precision() == precision

def test_binary_f1_measure(binary_matrix):    
    f1_measure = f1_score(true_bin, pred_bin)
    assert binary_matrix.f1_measure() == f1_measure

def test_multiclass_matrix(multiclass_matrix):
    matrix = confusion_matrix(true_multi, pred_multi)
    assert np.array_equal(multiclass_matrix.matrix, matrix)

def test_multiclass_accuracy(multiclass_matrix):
    acc = accuracy_score(true_multi, pred_multi)
    assert multiclass_matrix.accuracy() == acc

def test_multiclass_recall(multiclass_matrix):
    macro_recall = recall_score(true_multi, pred_multi, average="macro")
    micro_recall = recall_score(true_multi, pred_multi, average="micro")
    assert multiclass_matrix.recall("macro") == macro_recall
    assert multiclass_matrix.recall("micro") == micro_recall

def test_multiclass_precision(multiclass_matrix):
    macro_precision = precision_score(true_multi, pred_multi, average="macro")
    micro_precision = precision_score(true_multi, pred_multi, average="micro")
    assert multiclass_matrix.precision("macro") == macro_precision
    assert multiclass_matrix.precision("micro") == micro_precision

def test_multiclass_f1_measure(multiclass_matrix):
    macro_f1_measure = f1_score(true_multi, pred_multi, average="macro")
    micro_f1_measure = f1_score(true_multi, pred_multi, average="micro")
    assert multiclass_matrix.f1_measure("macro") == macro_f1_measure
    assert multiclass_matrix.f1_measure("micro") == micro_f1_measure
