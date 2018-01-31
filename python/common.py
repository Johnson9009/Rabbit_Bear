from enum import IntEnum
from assertpy import assert_that


class AxisIndex(IntEnum):
    FIRST =  0
    LAST  = -1


def get_unit_shape(samples, sample_axis):
    '''Return the shape with the sample axis item set to 1.'''
    shape = list(samples.shape)
    shape[sample_axis] = 1
    return tuple(shape)


class RecurrenceMean(object):
    '''Calculate mean of data using recurrence formula.'''
    def __init__(self):
        self._count = 0
        self._mean = None

    def __call__(self, count, mean):
        self._count += count
        if (self._mean is None):
            self._mean = mean
        else:
            self._mean += (count / self._count) * (mean - self._mean)

    @property
    def mean(self):
        assert_that(self._mean).is_not_none()
        return self._mean
