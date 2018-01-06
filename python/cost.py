import numpy as np
from .common import AxisIndex


def mean_squared_error(predicts, labels, sample_axis=AxisIndex.FIRST):
    '''Calculates the mean squared error between `predicts` and `labels`.'''
    return (1 / 2) * np.mean(np.square(predicts - labels), axis=sample_axis, keepdims=True)
