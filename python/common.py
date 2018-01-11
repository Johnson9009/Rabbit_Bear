from enum import IntEnum


class AxisIndex(IntEnum):
    FIRST =  0
    LAST  = -1


def get_unit_shape(samples, sample_axis):
    '''Return the shape with the sample axis item set to 1.'''
    shape = list(samples.shape)
    shape[sample_axis] = 1
    return tuple(shape)
