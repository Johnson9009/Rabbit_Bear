import numpy as np
from assertpy import assert_that
from .common import AxisIndex
from .activation import Sigmoid


class Softmax(object):
    def __init__(self, sample_axis=AxisIndex.FIRST):
        self._sample_axis = sample_axis
        self._class_axis = AxisIndex.LAST if (sample_axis == AxisIndex.FIRST) else AxisIndex.FIRST
        self._latest_forward_result = None

    def forward(self, logits):
        assert_that(np.shape(logits)).is_length(2)
        # In order to avoiding overflow and underflow, every logit need to subtract the maximal logit in its sample.
        logits -= np.max(logits, axis=self._class_axis, keepdims=True)
        exp_logits = np.exp(logits)
        self._latest_forward_result = exp_logits / np.sum(exp_logits, axis=self._class_axis, keepdims=True)
        return self._latest_forward_result

    def backward(self):
        # The farward method must be called before calling backward method.
        assert_that(self._latest_forward_result).is_not_none()
        s = self._latest_forward_result
        return s * (np.eye(s.shape[self._sample_axis]) - s)


class Cost(object):
    def __init__(self, sample_axis):
        self._sample_axis = sample_axis

    def __repr__(self):
        s = '{name}(sample_axis={_sample_axis})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x, labels, *args, **kwargs):
        raise NotImplementedError

    def backward(self, x, labels, *args, **kwargs):
        raise NotImplementedError


class L2(Cost):
    def __init__(self, sample_axis=AxisIndex.FIRST):
        super(L2, self).__init__(sample_axis)

    def forward(self, predicts, labels):
        '''Calculate the mean squared error between `predicts` and `labels`.'''
        # Vectorization like (A - Y).T * (A - Y) only support when dimension of labels is (, 1) or (1, ),
        # So we don't use it.
        samples_loss = predicts - labels
        np.square(samples_loss, out=samples_loss)
        samples_count = labels.shape[self._sample_axis]
        samples_loss /= 2 * samples_count
        return np.sum(samples_loss, axis=self._sample_axis, keepdims=True)

    def backward(self, predicts, labels):
        return (predicts - labels)


class CrossEntropy(Cost):
    def __init__(self, sample_axis=AxisIndex.FIRST):
        super(CrossEntropy, self).__init__(sample_axis)

    def forward(self, predicts, labels):
        '''Calculate cross entropy loss using the outputs of sigmoid or softmax.

        Parameters
        ----------
        predicts: prediction tensor with arbitrary shape, it's the output of sigmoid
        or softmax, so all of them must be (0, 1)
        '''
        samples_count = labels.shape[self._sample_axis]
        # Add a very little number to avoid underflow of log function.
        loss_matrix = -(labels * np.log(predicts + 1e-12) + (1 - labels) * np.log(1 - predicts + 1e-12))
        loss_matrix *= 1 / samples_count
        return np.sum(loss_matrix)

    def backward(self, predicts, labels):
        return (predicts - labels) / (predicts * (1 - predicts))


class SigmoidCrossEntropy(Cost):
    def __init__(self, sample_axis=AxisIndex.FIRST):
        super(SigmoidCrossEntropy, self).__init__(sample_axis)

    def forward(self, logits, labels):
        '''Calculate cross entropy loss using the input of sigmoid directly.

        Parameters
        ----------
        logits: output of net, and it's the input of sigmoid, so the valid range of
        them are (-inf, inf).calculating sigmoid and cross entropy together is
        more numerically stable through log-sum-exp trick.
        '''
        samples_count = labels.shape[self._sample_axis]
        # The logistic loss is logits - logits * labels + log(1 + exp(-logits))
        # Using the log-sum-exp trick to handle the scenario that logits are very big negative numbers.
        # so for logits < 0, the formula is - logits * labels + log(1 + exp(logits))
        loss_matrix = np.maximum(0, logits) - logits * labels + np.log(1 + np.exp(-np.abs(logits)))
        loss_matrix *= 1 / samples_count
        return np.sum(loss_matrix)

    def backward(self, logits, labels):
        return (Sigmoid.forward(logits) - labels)
