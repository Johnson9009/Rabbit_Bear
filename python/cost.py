import numpy as np
from .common import AxisIndex


def l2(predicts, labels, sample_axis=AxisIndex.FIRST, weight=1):
    '''Calculate the mean squared error between `predicts` and `labels`.

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


def cross_entropy(predicts, labels, sample_axis=AxisIndex.FIRST):
    '''Calculate cross entropy loss using the outputs of sigmoid or softmax.

    Parameters
    ----------
    predicts: prediction tensor with arbitrary shape, it's the output of sigmoid
        or softmax, so all of them must be (0, 1)
    '''
    samples_count = labels.shape[sample_axis]
    # Add a very little number to avoid underflow of log function.
    loss_matrix = -(labels * np.log(predicts + 1e-12) + (1 - labels) * np.log(1 - predicts + 1e-12))
    loss_matrix *= 1 / samples_count
    return np.sum(loss_matrix)


def sigmoid_cross_entropy(logits, labels, sample_axis=AxisIndex.FIRST):
    '''Calculate cross entropy loss using the input of sigmoid directly.

    Parameters
    ----------
    logits: output of net, and it's the input of sigmoid, so the valid range of
        them are (-inf, inf).calculating sigmoid and cross entropy together is
        more numerically stable through log-sum-exp trick.
    '''
    samples_count = labels.shape[sample_axis]
    # The logistic loss is logits - logits * labels + log(1 + exp(-logits))
    # Using the log-sum-exp trick to handle the scenario that logits are very big negative numbers.
    # so for logits < 0, the formula is - logits * labels + log(1 + exp(logits))
    loss_matrix = np.maximum(0, logits) - logits * labels + np.log(1 + np.exp(-np.abs(logits)))
    loss_matrix *= 1 / samples_count
    return np.sum(loss_matrix)
