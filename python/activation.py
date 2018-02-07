import numpy as np
from assertpy import assert_that


class Sigmoid(object):
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        s = Sigmoid.active(x)
        return s * (1 - s)


class Linear(object):
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def backward(x):
        return 1
