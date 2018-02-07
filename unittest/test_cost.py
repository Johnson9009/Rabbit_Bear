import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
from assertpy import assert_that
from rabbitbear import cost
from rabbitbear.activation import Linear, Sigmoid
from rabbitbear.common import AxisIndex, RecurrenceMean
from rabbitbear.dataset import minibatch_iterator


def get_data_loader(sample_axis=AxisIndex.FIRST):
    def data_loader(shuffle=True):
        '''For this exercise we need load dataset from a CSV file.'''
        with open('../examples/logistic_regression/ex2data1.txt') as f:
            dataset = np.loadtxt(f, delimiter=',')
            features = dataset[:, 0:2]
            labels = dataset[:, 2:3]
            count = dataset.shape[0]
            if (sample_axis == AxisIndex.LAST):
                features = np.transpose(features)
                labels = np.transpose(labels)
            yield (features, labels, count)
    return data_loader


def initialize_parameters(features_count, sample_axis=AxisIndex.FIRST):
    ''' Initializing all parameters this model needed. '''
    parameters = {}
    w_shape = (features_count, 1)
    if (sample_axis == AxisIndex.LAST):
        w_shape = (1, features_count)
    parameters['W'] = np.random.normal(0.0, 0.01, w_shape)
    parameters['b'] = np.zeros((1, 1))
    return parameters


def forward_propagation(features, parameters, sample_axis=AxisIndex.FIRST,
                        activator=Linear):
    ''' Using inputs and parameters to compute the final predictions. '''
    W = parameters['W']
    b = parameters['b']
    if (sample_axis == AxisIndex.FIRST):
        Z = np.dot(features, W) + b
    else:
        Z = np.dot(W, features) + b
    return activator.forward(Z)


def main():
    tolerance = 1.e-7
    sample_axis = AxisIndex.LAST
    parameters = initialize_parameters(2, sample_axis)

    # ce standard for cross_entropy
    ce_recur_mean = RecurrenceMean()
    golden_ce_recur_mean = RecurrenceMean()
    golden_ce = mx.gluon.loss.SigmoidBCELoss(True, batch_axis=int(sample_axis))
    ce_coster = cost.CrossEntropy(sample_axis)

    sigmoid_ce_recur_mean = RecurrenceMean()
    golden_sigmoid_ce_recur_mean = RecurrenceMean()
    golden_sigmoid_ce = mx.gluon.loss.SigmoidBCELoss(batch_axis=int(sample_axis))
    sigmoid_ce_coster = cost.SigmoidCrossEntropy(sample_axis)

    for features, labels, count in minibatch_iterator(get_data_loader(sample_axis), sample_axis, minibatch_size=None):
        predicts = forward_propagation(features, parameters, sample_axis, Sigmoid)
        ce_recur_mean(count, ce_coster.forward(predicts, labels))
        golden_ce_recur_mean(count, golden_ce(nd.array(predicts), nd.array(labels)).mean().asscalar())

        logits = forward_propagation(features, parameters, sample_axis)
        sigmoid_ce_recur_mean(count, sigmoid_ce_coster.forward(logits, labels))
        golden_sigmoid_ce_recur_mean(count, golden_sigmoid_ce(nd.array(logits), nd.array(labels)).mean().asscalar())

    assert_that(ce_recur_mean.mean.shape).is_equal_to(golden_ce_recur_mean.mean.shape)
    difference_rate = abs(np.squeeze((ce_recur_mean.mean - golden_ce_recur_mean.mean) / ce_recur_mean.mean))
    assert_that(difference_rate).is_less_than(tolerance)

    assert_that(sigmoid_ce_recur_mean.mean.shape).is_equal_to(golden_sigmoid_ce_recur_mean.mean.shape)
    difference_rate = abs(np.squeeze((sigmoid_ce_recur_mean.mean - golden_sigmoid_ce_recur_mean.mean) / sigmoid_ce_recur_mean.mean))
    assert_that(difference_rate).is_less_than(tolerance)


if (__name__ == '__main__'):
    main()
