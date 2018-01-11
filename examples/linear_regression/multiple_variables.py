import os
import logging
import numpy as np
from rabbitbear import cost
from rabbitbear.common import AxisIndex
from rabbitbear.dataset import minibatch_iterator, StandardScaler
from rabbitbear.visualization import IterativeCostPlot
from rabbitbear.utils.logging import config_logging_yaml


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def get_data_loader(sample_axis=AxisIndex.FIRST):
    def data_loader(shuffle=True):
        '''For this exercise we need load dataset from a CSV file.'''
        with open('ex1data2.txt') as f:
            dataset = np.loadtxt(f, delimiter=',')
            features = dataset[:, 0:2]
            labels = dataset[:, 2:3]
            if (sample_axis == AxisIndex.LAST):
                features = np.transpose(features)
                labels = np.transpose(labels)
            yield (features, labels)
    return data_loader


def initialize_parameters(features_count, sample_axis=AxisIndex.FIRST):
    ''' Initializing all parameters this model needed. '''
    parameters = {}
    w_shape = (features_count, 1)
    if (sample_axis == AxisIndex.LAST):
        w_shape = (1, features_count)
    parameters['W'] = np.random.randn(w_shape[0], w_shape[1])
    parameters['b'] = np.zeros((1, 1))
    return parameters


def forward_propagation(features, parameters, sample_axis=AxisIndex.FIRST):
    ''' Using inputs and parameters to compute the final predictions. '''
    W = parameters['W']
    b = parameters['b']
    if (sample_axis == AxisIndex.FIRST):
        return np.dot(features, W) + b
    else:
        return np.dot(W, features) + b


def compute_cost(predicts, labels, sample_axis=AxisIndex.FIRST, weight=1):
    ''' Compute averaged cost using all samples. '''
    return cost.mean_squared_error(predicts, labels, sample_axis, weight=weight)


def back_propagation(features, labels, predicts, sample_axis=AxisIndex.FIRST):
    ''' Compute the gradients of parameters. '''
    dZ = predicts - labels
    samples_count = labels.shape[sample_axis]
    grads = {}
    if (sample_axis == AxisIndex.FIRST):
        grads['W'] = (1 / samples_count) * np.dot(np.transpose(features), dZ)
    else:
        grads['W'] = (1 / samples_count) * np.dot(dZ, np.transpose(features))
    grads['b'] = (1 / samples_count) * np.sum(dZ)
    return grads


def update_parameters(parameters, grads, learning_rate):
    ''' Update the parameters using its gradients once. '''
    parameters['W'] -= learning_rate * grads['W']
    parameters['b'] -= learning_rate * grads['b']
    return parameters


def predict(features, parameters, sample_axis=AxisIndex.FIRST):
    return forward_propagation(features, parameters, sample_axis)


def main():
    config_logging_yaml()
    epochs_count = 400
    learning_rate = 0.01
    sample_axis = AxisIndex.LAST

    iterCostPlot = IterativeCostPlot(learning_rate, step=5)
    standard_scaler = StandardScaler(get_data_loader(sample_axis), sample_axis, minibatch_size=None)
    parameters = initialize_parameters(2, sample_axis)

    for epoch in range(epochs_count):
        minibatch_costs = []
        for minibatch_features, minibatch_labels in minibatch_iterator(get_data_loader(sample_axis), sample_axis, minibatch_size=None):
            minibatch_features = standard_scaler.transform(minibatch_features)
            predicts = forward_propagation(minibatch_features, parameters, sample_axis)
            minibatch_costs.append(compute_cost(predicts, minibatch_labels, sample_axis))
            grads = back_propagation(minibatch_features, minibatch_labels, predicts, sample_axis)
            parameters = update_parameters(parameters, grads, learning_rate)

        epoch_train_cost = np.mean(minibatch_costs)
        logger.info('Cost after epoch %d: %f' % (epoch, epoch_train_cost))
        iterCostPlot.update(epoch_train_cost)

    test_features = np.array([[1650, 3]]) if (sample_axis == AxisIndex.FIRST) else np.array([[1650], [3]])
    test_features = standard_scaler.transform(test_features)
    test_predicts = predict(test_features, parameters, sample_axis)

    logger.debug(parameters)
    logger.debug(test_predicts)
    iterCostPlot.close()


if (__name__ == '__main__'):
    main()
