import numpy as np
from assertpy import assert_that


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def active(type, x):
    if (type is None):
        return x
    assert_that(type).is_in(*tuple(activation_functions))
    return activation_functions[type](x)


activation_functions = {
    'sigmoid' : sigmoid,
}
