# Aqui importamos os módulos necessários para os testes
from pytest import fixture
import numpy as np
from unittest import TestCase
from k_fold import KFold

@fixture(scope="module")
def array():
    arr = np.array([[1, 2, 2], 
                    [3, 4, 2], 
                    [5, 6, 2], 
                    [7, 8, 2], 
                    [9, 10, 1], 
                    [11, 12, 1], 
                    [13, 14, 1], 
                    [15, 16, 1], 
                    [17, 18, 1], 
                    [19, 20, 1],
                    [21, 22, 1], 
                    [23, 24, 2], 
                    [25, 26, 2], 
                    [27, 28, 2], 
                    [29, 30, 2], 
                    [31, 32, 2], 
                    [33, 34, 0], 
                    [35, 36, 0], 
                    [37, 38, 0], 
                    [39, 40, 0]])
    return arr

@fixture(scope="module")
def randomized_k_fold_3(array):
    return KFold(array, 3, randomize=True)
    
@fixture(scope="module")
def not_randomized_k_fold_3(array):
    return KFold(array, 3, randomize=False)

@fixture(scope="module")
def not_randomized_k_fold_4(array):
    return KFold(array, 4, randomize=False)

@fixture(scope="module")
def expected_not_randomized_k_fold_4():
    expected = [np.array([[ 1,  2,  2],
                       [23, 24,  2],
                       [31, 32,  2],
                       [ 9, 10,  1],
                       [17, 18,  1],
                       [33, 34,  0]]), 
                np.array([[ 3,  4,  2],
                       [25, 26,  2],
                       [11, 12,  1],
                       [19, 20,  1],
                       [35, 36,  0]]), 
                np.array([[ 5,  6,  2],
                       [27, 28,  2],
                       [13, 14,  1],
                       [21, 22,  1],
                       [37, 38,  0]]), 
                np.array([[ 7,  8,  2],
                       [29, 30,  2],
                       [15, 16,  1],
                       [39, 40,  0]])]
    return expected

@fixture(scope="module")
def expected_not_randomized_k_fold_3():
    expected = [np.array([[ 1,  2,  2],
                       [ 7,  8,  2],
                       [27, 28,  2],
                       [ 9, 10,  1],
                       [15, 16,  1],
                       [21, 22,  1],
                       [33, 34,  0],
                       [39, 40,  0]]), 
                np.array([[ 3,  4,  2],
                       [23, 24,  2],
                       [29, 30,  2],
                       [11, 12,  1],
                       [17, 18,  1],
                       [35, 36,  0]]), 
                np.array([[ 5,  6,  2],
                       [25, 26,  2],
                       [31, 32,  2],
                       [13, 14,  1],
                       [19, 20,  1],
                       [37, 38,  0]])]
    return expected


def test_randomize(randomized_k_fold_3, not_randomized_k_fold_3):
    assert len(randomized_k_fold_3.folds) == len(not_randomized_k_fold_3.folds)
    assert all([not np.array_equal(a, b) for a,b in zip(randomized_k_fold_3.folds, not_randomized_k_fold_3.folds)])

def test_not_randomized_k_fold_3(not_randomized_k_fold_3, expected_not_randomized_k_fold_3):
    assert len(not_randomized_k_fold_3.folds) == len(expected_not_randomized_k_fold_3)
    assert all([np.array_equal(a, b) for a,b in zip(expected_not_randomized_k_fold_3, not_randomized_k_fold_3.folds)])

def test_not_randomized_k_fold_4(not_randomized_k_fold_4, expected_not_randomized_k_fold_4):
    assert len(not_randomized_k_fold_4.folds) == len(expected_not_randomized_k_fold_4)
    assert all([np.array_equal(a, b) for a,b in zip(expected_not_randomized_k_fold_4, not_randomized_k_fold_4.folds)])

def test_train_test_split_k_fold_4(not_randomized_k_fold_4, expected_not_randomized_k_fold_4):
    test_train_splits = [not_randomized_k_fold_4.test_train_split(i) for i in range(4)]
    expected_train = [np.concatenate(list(expected_not_randomized_k_fold_4[i] for i in range(4) if i != j)) for j in range(4)]
    expected_test = [expected_not_randomized_k_fold_4[i] for i in range(4)]
    expected_train_test_splits = zip(expected_test, expected_train)

    assert all([np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a,b in zip(expected_train_test_splits, test_train_splits)])
