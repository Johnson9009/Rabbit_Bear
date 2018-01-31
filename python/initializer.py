import numpy as np
from .common import AxisIndex
from assertpy import assert_that


class RandomNormal(object):
    def __init__(self, mean=0.0, variance=1.0):
        self.mean = mean
        self.variance = variance

    def __call__(self, shape):
        return np.random.normal(self.mean, self.variance, shape)


class Constant(object):
    def __init__(self, values):
        self.values = values

    def __call__(self, shape):
        assert_that(shape).is_equal_to(np.shape(self.values))
        return self.values


class Zero(object):
    def __call__(self, shape):
        return np.zeros(shape)


class Xavier(object):
    def __init__(random_type='uniform', factor_type='avg', magnitude=3.0, sample_axis=AxisIndex.FIRST):
        assert_that(random_type).is_in('uniform', 'gaussian')
        assert_that(factor_type).is_in('avg', 'in', 'out')
        self.random_type = random_type
        self.factor_type = factor_type
        self.magnitude = magnitude
        self.sample_axis = sample_axis

    def __call__(self, shape):
        assert_that(shape).is_instance_of(tuple)
        assert_that(shape).is_length(2)

        if (self.sample_axis == AxisIndex.FIRST):
            fan_in, fan_out = shape
        else:
            fan_out, fan_in = shape

        if (self.factor_type == 'avg'):
            factor = (fan_in + fan_out) / 2.0
        elif (self.factor_type == 'in'):
            factor = fan_in
        else:
            factor = fan_out

        scale = np.sqrt(self.magnitude / factor)
        if (self.random_type == 'uniform'):
            return np.random.uniform(-scale, scale, shape)
        else:
            return np.random.normal(0.0, scale, shape)


class Initializer(object):
    def __init__(self, weight_initer=RandomNormal(), bias_initer=Zero()):
        self.weight_initer = weight_initer
        self.bias_initer = bias_initer

    def init_weight(self, shape=None):
        return self.weight_initer(shape)

    def init_bias(self, shape=None):
        return self.bias_initer(shape)
