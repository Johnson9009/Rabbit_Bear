import logging
import numpy as np
from .common import AxisIndex


logger = logging.getLogger(__name__)


def minibatch_iterator(data_loader, sample_axis=AxisIndex.FIRST,
                       minibatch_size=64, shuffle=True, drop_tail=False):
    ''' This is the mini batch generator which yield one mini batch dataset each time it is called,
    this generator is only responsible for shuffling the dataset given by data loader iterator and
    yielding one mini batch of it, when there isn't any data in this dataset data loader iterator
    will be called again until there isn't any more dataset from data loader. '''
    disable_batch = True if (minibatch_size is None) else False
    remain_features = remain_labels = None
    for features, labels in data_loader(shuffle=shuffle):
        if (remain_features is not None):
            assert (remain_labels.shape[sample_axis] == remain_features.shape[sample_axis]), "Remain features don't match remain labels!"
            features = np.concatenate((remain_features, features), axis=sample_axis)
            labels = np.concatenate((remain_labels, labels), axis=sample_axis)
            logger.debug('Concatenated remain %d' % remain_features.shape[sample_axis])

        samples_count = features.shape[sample_axis]
        minibatch_size = samples_count if (disable_batch is True) else minibatch_size
        minibatches_count = samples_count // minibatch_size
        processable_count = minibatches_count * minibatch_size
        sample_indexes = list(np.random.permutation(processable_count) if (shuffle is True) else range(processable_count))
        logger.debug('samples_count:%d, minibatches_count:%d, processable_count:%d' % (samples_count, minibatches_count, processable_count))

        for i in range(minibatches_count):
            minibatch_features = features.take(sample_indexes[i * minibatch_size : (i + 1) * minibatch_size], axis=sample_axis)
            minibatch_labels = labels.take(sample_indexes[i * minibatch_size : (i + 1) * minibatch_size], axis=sample_axis)
            yield (minibatch_features, minibatch_labels)

        if (processable_count < samples_count):
            remain_features = features.take(range(processable_count, samples_count), axis=sample_axis)
            remain_labels = labels.take(range(processable_count, samples_count), axis=sample_axis)
            logger.debug('remain %d samples' % remain_features.shape[sample_axis])
        else:
            remain_features = remain_labels = None

    if ((drop_tail is False) and (remain_features is not None)):
        assert (remain_labels.shape[sample_axis] == remain_features.shape[sample_axis]), "Remain features don't match remain labels!"
        yield (remain_features, remain_labels)


class StandardScaler(object):
    '''Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
    Mean and standard deviation are then stored to be used on later data using the transform method.
    '''
    def __init__(self, data_loader, sample_axis=AxisIndex.FIRST, minibatch_size=64):
        minibatch_iter = minibatch_iterator(data_loader, sample_axis, minibatch_size, shuffle=False)
        minibatch_features, _ = next(minibatch_iter)
        features_shape = list(minibatch_features.shape)
        features_shape[sample_axis] = 1

        self._mean = np.zeros(tuple(features_shape))
        self._stddev = np.ones(tuple(features_shape));
        self._samples_count = 0;

        while(True):
            try:
                count = minibatch_features.shape[sample_axis]
                self._samples_count += count;
                self._mean += (count / self._samples_count) * (np.sum((1 / count) * minibatch_features,
                                                                      axis=sample_axis, keepdims=True) - self._mean)
                # TODO: Incremental iterative standard deviation computation.
                minibatch_features, _ = next(minibatch_iter)
            except StopIteration:
                break


    def transform(self, features):
        '''Scaling features of X according to feature_range.'''
        return (features - self._mean) / self._stddev
