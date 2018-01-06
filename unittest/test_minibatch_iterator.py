import numpy as np
from rabbitbear.dataset import minibatch_iterator


dataset_filenames = ['../examples/dataset0.txt', '../examples/dataset1.txt', '../examples/dataset2.txt']


def data_loader_iterator(shuffle=True):
    ''' For this exercise we need load dataset from several CSV files. '''
    for dataset_filename in dataset_filenames:
        with open(dataset_filename) as f:
            dataset = np.loadtxt(f, delimiter=',')
        yield (dataset[:, 0:1], dataset[:, 1:2])


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
    assert (np.shape(golden_features)[0] == np.shape(golden_labels)[0]), "golden features don't match golden labels!"

    dropped_count = np.shape(golden_features)[0] % minibatch_size if (minibatch_size is not None) else 0
    dropped_features = golden_features[len(golden_features) - dropped_count : ]
    dropped_labels = golden_labels[len(golden_labels) - dropped_count : ]

    for minibatch_features, minibatch_labels in minibatch_iterator(data_loader_iterator, minibatch_size=minibatch_size,
                                                                   shuffle=shuffle, drop_tail=True):
        assert (minibatch_features.shape[0] == minibatch_labels.shape[0]), "minibatch features don't match minibatch labels!"
        for feature, label in zip(minibatch_features, minibatch_labels):
            golden_features.remove(feature)
            golden_labels.remove(label)
    assert (np.shape(golden_features)[0] == np.shape(golden_labels)[0]), "After removing minibatch, golden features don't match golden labels!"
    assert (np.shape(golden_features)[0] == np.shape(dropped_features)[0]), "Remain samples don't match dropped samples!"

    for dropped_feature, dropped_label in zip(dropped_features, dropped_labels):
        golden_features.remove(dropped_feature)
        golden_labels.remove(dropped_label)
    assert (np.shape(golden_features)[0] == 0), 'Impossible!'


if (__name__ == '__main__'):
    main()
