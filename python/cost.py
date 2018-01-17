import numpy as np
from .common import AxisIndex


def l2_loss(predicts, labels, sample_axis=AxisIndex.FIRST, weight=1):
    '''Calculates the mean squared error between `predicts` and `labels`.

    Parameters
    ----------
    weight:float, default 1
      Global scalar weight for loss
    '''
    # Vectorization like (A - Y).T * (A - Y) only support when dimension of labels is (, 1) or (1, ),
    # So we don't use it.
    samples_loss = predicts - labels
    np.square(samples_loss, out=samples_loss)
    samples_count = labels.shape[sample_axis]
    samples_loss *= weight / (2 * samples_count)
    return np.sum(samples_loss, axis=sample_axis, keepdims=True)


def cross_entropy_loss(predicts, labels, sample_axis=AxisIndex.FIRST):
    loss_matrix = -(labels * np.log(predicts + 1e-12) + (1 - labels) * np.log(1 - predicts + 1e-12))
    samples_count = labels.shape[sample_axis]
    loss_matrix *= 1 / samples_count
    return np.sum(loss_matrix)
