import numpy as np
from .common import AxisIndex


def mean_squared_error(predicts, labels, sample_axis=AxisIndex.FIRST, weight=1):
    '''Calculates the mean squared error between `predicts` and `labels`.

    Parameters
    ----------
    weight:float, default 1
      Global scalar weight for loss
    '''
    samples_count = labels.shape[sample_axis]
    samples_loss = np.square(predicts - labels)
    assert (samples_loss.shape[(sample_axis + 1) % 2] == 1), 'Cost of each sample should be a scalar!'
    return np.sum(np.multiply(weight / (2 * samples_count), samples_loss))
