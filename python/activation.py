import numpy as np
from assertpy import assert_that


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def none(x):
    return x
