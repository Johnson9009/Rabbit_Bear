import numpy as np
from assertpy import assert_that
from rabbitbear.dataset import minibatch_iterator


dataset_filenames = ['../examples/linear_regression/dataset/test/dataset0.txt',
                     '../examples/linear_regression/dataset/test/dataset1.txt',
                     '../examples/linear_regression/dataset/test/dataset2.txt']


def data_loader(shuffle=True):
    ''' For this exercise we need load dataset from several CSV files. '''
    for dataset_filename in dataset_filenames:
        with open(dataset_filename) as f:
            dataset = np.loadtxt(f, delimiter=',')
        yield (dataset[:, 0:1], dataset[:, 1:2], dataset.shape[0])


def get_whole_golden_dataset():
    golden_features = golden_labels = None
    for dataset_filename in dataset_filenames:
        with open(dataset_filename) as f:
            dataset = np.loadtxt(f, delimiter=',')
        if (golden_features is None):
            golden_features = dataset[:, 0:1]
            golden_labels = dataset[:, 1:2]
        else:
            golden_features = np.concatenate((golden_features, dataset[:, 0:1]), 0)
            golden_labels = np.concatenate((golden_labels, dataset[:, 1:2]), 0)
    return (golden_features.tolist(), golden_labels.tolist())


def main():
    minibatch_size = None
    shuffle = True

    golden_features, golden_labels = get_whole_golden_dataset()
    assert_that(np.shape(golden_features)[0]).is_equal_to( np.shape(golden_labels)[0])

    dropped_count = np.shape(golden_features)[0] % minibatch_size if (minibatch_size is not None) else 0
    dropped_features = golden_features[len(golden_features) - dropped_count : ]
    dropped_labels = golden_labels[len(golden_labels) - dropped_count : ]

    for mini_features, mini_labels, _ in minibatch_iterator(data_loader, minibatch_size, shuffle, drop_tail=True):
        assert_that(mini_features.shape[0]).is_equal_to(mini_labels.shape[0])
        for feature, label in zip(mini_features, mini_labels):
            golden_features.remove(feature)
            golden_labels.remove(label)
    assert_that(np.shape(golden_features)[0]).is_equal_to(np.shape(golden_labels)[0])
    assert_that(np.shape(golden_features)[0]).is_equal_to(np.shape(dropped_features)[0])

    for dropped_feature, dropped_label in zip(dropped_features, dropped_labels):
        golden_features.remove(dropped_feature)
        golden_labels.remove(dropped_label)
    assert_that(np.shape(golden_features)[0]).is_equal_to(0)


if (__name__ == '__main__'):
    main()
