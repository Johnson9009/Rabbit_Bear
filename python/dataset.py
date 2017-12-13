import numpy as np


def minibatch_iterator(data_loader_iterator, minibatch_size=64, shuffle=True):
    ''' This is the mini batch generator which yield one mini batch dataset each time it is called,
    this generator is only responsible for shuffling the dataset given by data loader iterator and
    yielding one mini batch of it, when there isn't any data in this dataset data loader iterator
    will be called again until there isn't any more dataset from data loader. '''
    for features, labels in data_loader_iterator(shuffle=shuffle):
        samples_count = features.shape[0]
        num_complete_minibatches = samples_count // minibatch_size
        sample_indexes = list(np.random.permutation(samples_count) if (shuffle is True) else range(samples_count))

        for i in range(num_complete_minibatches):
            minibatch_features = features[sample_indexes[i * minibatch_size : (i + 1) * minibatch_size]]
            minibatch_labels = labels[sample_indexes[i * minibatch_size : (i + 1) * minibatch_size]]
            yield (minibatch_features, minibatch_labels)

        if (samples_count % minibatch_size != 0):
            minibatch_features = features[sample_indexes[num_complete_minibatches * minibatch_size : samples_count]]
            minibatch_labels = labels[sample_indexes[num_complete_minibatches * minibatch_size : samples_count]]
            yield (minibatch_features, minibatch_labels)


class StandardScarler(object):
    '''Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
    Mean and standard deviation are then stored to be used on later data using the transform method.
    '''
    def __init__(self, data_loader_iterator, sample_axis=0, minibatch_size=64):
        features_shape = list(data_loader_iterator(only_first=True)[0].shape)
        features_shape[sample_axis] = 1
        self._mean = np.zeros(tuple(features_shape))
        self._stddev = np.ones(tuple(features_shape));
        self._samples_count = 0;

        for minibatch_features, _ in minibatch_iterator(data_loader_iterator, minibatch_size, shuffle=False):
            count = minibatch_features.shape[sample_axis]
            self._samples_count += count;
            self._mean += (count / self._samples_count) * (np.sum((1 / count) * minibatch_features, axis=sample_axis) - self._mean)
            # TODO: Incremental iterative standard deviation computation.

    def transform(self, features):
        '''Scaling features of X according to feature_range.'''
        return (features - self._mean) / self._stddev



