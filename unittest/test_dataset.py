import numpy as np
from rabbitbear.dataset import StandardScaler


samples_count = 1000
features_count = 3


def data_loader_iterator(shuffle=True):
    '''For this test we need generate 0 ~ (end_number - 1).'''
    yield (np.arange(samples_count * features_count).reshape((features_count, samples_count)), np.zeros((1, samples_count)))


def main():
    standard_scaler = StandardScaler(data_loader_iterator, sample_axis_at_end=True, minibatch_size=32)
    print(standard_scaler._mean)
    print(standard_scaler._samples_count)


if (__name__ == '__main__'):
    main()
