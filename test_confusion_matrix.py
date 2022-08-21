# Aqui importamos os módulos necessários para os testes
from pytest import fixture
import numpy as np
from confusion_matrix import ConfusionMatrix


@fixture(scope="module")
def binary_matrix():
    # 4 3
    # 2 2
    true = np.array([0,0,1,1,0,0,1,1,0,0,0])
    pred = np.array([0,1,0,1,1,1,0,1,0,0,0])
    return ConfusionMatrix(true, pred)
    
@fixture(scope="module")
def multiclass_matrix():
    # 1 1 1
    # 2 2 2
    # 3 3 3
    true = np.array([0,1,1,2,2,2,0,0,1,1,1,1,2,2,2,2,2,2])
    pred = np.array([0,1,1,2,2,2,1,2,0,0,2,2,0,0,0,1,1,1])
    return ConfusionMatrix(true, pred)

def test_binary_matrix(binary_matrix):
    assert np.array_equal(binary_matrix.matrix, np.array([[4,3],[2,2]], dtype=int))

def test_binary_accuracy(binary_matrix):
    accuracy = 6/11
    assert binary_matrix.accuracy() == accuracy

def test_binary_recall(binary_matrix):
    recall = 4/7
    assert binary_matrix.recall() == recall

def test_binary_precision(binary_matrix):
    precision = 4/6
    assert binary_matrix.precision() == precision

def test_binary_f1_measure(binary_matrix):
    recall = 4/7
    precision = 4/6
    f1_measure = 2*(precision*recall)/(precision+recall)
    assert binary_matrix.f1_measure() == f1_measure

def test_multiclass_matrix(multiclass_matrix):
    assert np.array_equal(multiclass_matrix.matrix, np.array([[1,1,1],[2,2,2],[3,3,3]], dtype=int))

def test_multiclass_accuracy(multiclass_matrix):
    acc = 6/18
    assert multiclass_matrix.accuracy() == acc

def test_multiclass_recall(multiclass_matrix):
    macro_recall = (1/3 + 2/6 + 3/9)/3
    micro_recall = 6/18
    assert multiclass_matrix.recall("macro") == macro_recall
    assert multiclass_matrix.recall("micro") == micro_recall

def test_multiclass_precision(multiclass_matrix):
    macro_precision = (1/6 + 2/6 + 3/6)/3
    micro_precision = 6/18
    assert multiclass_matrix.precision("macro") == macro_precision
    assert multiclass_matrix.precision("micro") == micro_precision

def test_multiclass_f1_measure(multiclass_matrix):
    macro_recall = (1/3 + 2/6 + 3/9)/3
    macro_precision = (1/6 + 2/6 + 3/6)/3
    macro_f1_measure = 2*(macro_precision*macro_recall)/(macro_precision+macro_recall)
    micro_f1_measure = 6/18
    assert multiclass_matrix.f1_measure("macro") == macro_f1_measure
    assert multiclass_matrix.f1_measure("micro") == micro_f1_measure
