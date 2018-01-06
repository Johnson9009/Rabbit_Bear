import numpy as np
from rabbitbear.common import AxisIndex
from rabbitbear.dataset import StandardScaler


samples_count = 1000
features_count = 3
sample_axis = AxisIndex.LAST


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


def data_loader_iterator(shuffle=True):
    '''For this test we need generate 0 ~ (end_number - 1).'''
    yield (generate_features(), generate_labels())


def main():
    standard_scaler = StandardScaler(data_loader_iterator, sample_axis, minibatch_size=32)

    golden_mean_shape = (1, features_count) if (sample_axis == AxisIndex.FIRST) else (features_count, 1)
    assert(standard_scaler._mean.shape == golden_mean_shape)
    golden_mean = np.mean(generate_features(), axis=sample_axis, keepdims=True)
    assert((standard_scaler._mean == golden_mean).all())

    assert(standard_scaler._samples_count == samples_count)


if (__name__ == '__main__'):
    main()
