import numpy as np
from assertpy import assert_that
from rabbitbear.common import AxisIndex, RecurrenceMean
from rabbitbear.dataset import minibatch_iterator

samples_count = 1000
features_count = 3
sample_axis = AxisIndex.LAST
tolerance = 3.0e-14


def generate_features():
    if (sample_axis == AxisIndex.FIRST):
        return np.arange(samples_count * features_count).reshape((samples_count, features_count))
    else:
        return np.arange(samples_count * features_count).reshape((features_count, samples_count))


def generate_labels():
    if (sample_axis == AxisIndex.FIRST):
        return np.zeros((samples_count, 1))
    else:
        return np.zeros((1, samples_count))


def data_loader(shuffle=True):
    '''For this test we need generate 0 ~ (end_number - 1).'''
    yield (generate_features(), generate_labels(), samples_count)


def main():
    recurrence_mean = RecurrenceMean()
    for mini_features, _, mini_count in minibatch_iterator(data_loader, sample_axis, minibatch_size=32):
        recurrence_mean(mini_count, np.mean(mini_features, axis=sample_axis, keepdims=True))

    golden_shape = (1, features_count) if (sample_axis == AxisIndex.FIRST) else (features_count, 1)
    assert_that(recurrence_mean.mean.shape).is_equal_to(golden_shape)
    golden_mean = np.mean(generate_features(), axis=sample_axis, keepdims=True)
    difference_rate = abs(recurrence_mean.mean - golden_mean) / recurrence_mean.mean
    assert_that((difference_rate < tolerance).all()).is_true()

    assert_that(recurrence_mean._count).is_equal_to(samples_count)


if (__name__ == '__main__'):
    main()
