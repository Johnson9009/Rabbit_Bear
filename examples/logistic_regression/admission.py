import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from rabbitbear import cost, activations
from rabbitbear.common import AxisIndex, RecurrenceMean
from rabbitbear.dataset import minibatch_iterator, StandardScaler
from rabbitbear.visualization import IterativeCostPlot
from rabbitbear.utils.logging import config_logging_yaml


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


class IterativeFitPlot(object):
    '''Draw the fitting figure using data got gradually, the fitting line will be changed iteratively.'''
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Exam 1 score')
        self.ax.set_ylabel('Exam 2 score')
        self.ax.scatter([], [], c='b', marker='+', label='Admitted')
        self.ax.scatter([], [], c='y', marker='o', label='Not admitted')
        self.line, = self.ax.plot([], [], label='Logistic regression')
        self.ax.legend()

    def scatter_samples(self, features, labels):
        admitted = np.squeeze(labels == 1)
        not_admitted = np.squeeze(labels == 0)
        self.ax.scatter(features[admitted, 0:1], features[admitted, 1:2], c='b', marker='+', label='Admitted')
        self.ax.scatter(features[not_admitted, 0:1], features[not_admitted, 1:2], c='y', marker='o', label='Not admitted')
        plt.pause(0.000001)

    def draw_fit_line(self, parameters):
        x = self.ax.get_xticks().reshape((-1, 1))
        line_parameters = {'W':-parameters['W'][0] / parameters['W'][1], 'b':-parameters['b'] / parameters['W'][1]}
        y = forward_propagation(x, line_parameters)
        self.line.set_data(x, y)
        logger.debug('orig W: {}, orig b: {}'.format(parameters['W'], parameters['b']))
        logger.debug('line W: {}, line b: {}'.format(line_parameters['W'], line_parameters['b']))
        logger.debug('x: {}, y: {}'.format(x, y))
        plt.pause(0.000001)

    def close(self, hold_before_close=True):
        if (hold_before_close is True):
            plt.show()
        plt.close(self.fig)


def get_data_loader(sample_axis=AxisIndex.FIRST):
    def data_loader(shuffle=True):
        '''For this exercise we need load dataset from a CSV file.'''
        with open('ex2data1.txt') as f:
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
    parameters['W'] = np.random.randn(*w_shape)
    parameters['b'] = np.zeros((1, 1))
    return parameters


def forward_propagation(features, parameters, sample_axis=AxisIndex.FIRST,
                        activation=None):
    ''' Using inputs and parameters to compute the final predictions. '''
    W = parameters['W']
    b = parameters['b']
    if (sample_axis == AxisIndex.FIRST):
        Z = np.dot(features, W) + b
    else:
        Z = np.dot(W, features) + b
    return activations.active(activation, Z)


def compute_cost(predicts, labels, sample_axis=AxisIndex.FIRST):
    ''' Compute averaged cost using all samples. '''
    return cost.cross_entropy_loss(predicts, labels, sample_axis)


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
    epochs_count = 200
    learning_rate = 0.001072
    sample_axis = AxisIndex.FIRST

    iterFitPlot = IterativeFitPlot()
    iterCostPlot = IterativeCostPlot(learning_rate, step=5)
    parameters = initialize_parameters(2, sample_axis)

    for epoch in range(epochs_count):
        recurrence_mean = RecurrenceMean()
        for mini_features, mini_labels, mini_count in minibatch_iterator(get_data_loader(sample_axis),
                                                                         sample_axis, minibatch_size=None):
            mini_predicts = forward_propagation(mini_features, parameters, sample_axis, 'sigmoid')
            recurrence_mean(mini_count, compute_cost(mini_predicts, mini_labels, sample_axis))
            grads = back_propagation(mini_features, mini_labels, mini_predicts, sample_axis)
            parameters = update_parameters(parameters, grads, learning_rate)
            if (epoch == 0):
                iterFitPlot.scatter_samples(mini_features, mini_labels)

        epoch_train_cost = np.squeeze(recurrence_mean.mean)
        logger.info('Cost after epoch %d: %f' % (epoch, epoch_train_cost))
        iterFitPlot.draw_fit_line(parameters)
        iterCostPlot.update(epoch_train_cost)

    logger.info(parameters)
    iterFitPlot.close()
    iterCostPlot.close()


if (__name__ == '__main__'):
    main()
