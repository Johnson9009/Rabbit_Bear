import numpy as np

def minibatch_iterator(data_loader_iterator, minibatch_size=64):
    ''' This is the mini batch generator which yield one mini batch dataset each time it is called,
    this generator is only responsible for shuffling the dataset given by data loader iterator and
    yielding one mini batch of it, when there isn't any data in this dataset data loader iterator
    will be called again until there isn't any more dataset from data loader. '''
    for X, Y in data_loader_iterator():
        m = X.shape[0]
        permutation = list(np.random.permutation(m))
        num_complete_minibatches = m // minibatch_size

        for i in range(num_complete_minibatches):
            minibatch_X = X[permutation[i * minibatch_size : (i + 1) * minibatch_size]]
            minibatch_Y = Y[permutation[i * minibatch_size : (i + 1) * minibatch_size]]
            yield (minibatch_X, minibatch_Y)

        if (m % minibatch_size != 0):
            minibatch_X = X[permutation[num_complete_minibatches * minibatch_size : m]]
            minibatch_Y = Y[permutation[num_complete_minibatches * minibatch_size : m]]
            yield (minibatch_X, minibatch_Y)
