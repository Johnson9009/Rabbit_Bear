import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from rabbitbear import cost, activation
from rabbitbear.evaluation import Metric
from rabbitbear.common import AxisIndex, RecurrenceMean
from rabbitbear.dataset import minibatch_iterator, StandardScaler
from rabbitbear.initializer import Initializer, Zero, RandomNormal, Constant
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


def initialize_parameters(features_count, initializer, sample_axis=AxisIndex.FIRST):
    ''' Initializing all parameters this model needed. '''
    parameters = {}
    w_shape = (features_count, 1)
    if (sample_axis == AxisIndex.LAST):
        w_shape = (1, features_count)
    parameters['W'] = initializer.init_weight(w_shape)
    parameters['b'] = initializer.init_bias((1, 1))
    return parameters


def forward_propagation(features, parameters, sample_axis=AxisIndex.FIRST):
    ''' Using inputs and parameters to compute the final predictions. '''
    W = parameters['W']
    b = parameters['b']
    if (sample_axis == AxisIndex.FIRST):
        Z = np.dot(features, W) + b
    else:
        Z = np.dot(W, features) + b
    return Z


def compute_cost(logits, labels, sample_axis=AxisIndex.FIRST):
    ''' Compute averaged cost using all samples. '''
    return cost.sigmoid_cross_entropy(logits, labels, sample_axis)


def compute_dataset_cost(data_loader, parameters, sample_axis=AxisIndex.FIRST):
    recurrence_mean = RecurrenceMean()
    for features, labels, count in minibatch_iterator(data_loader, sample_axis, minibatch_size=None):
        logits = forward_propagation(features, parameters, sample_axis)
        recurrence_mean(count, compute_cost(logits, labels, sample_axis))
    return np.squeeze(recurrence_mean.mean)


def back_propagation(features, labels, logits, sample_axis=AxisIndex.FIRST):
    ''' Compute the gradients of parameters. '''
    dZ = activation.sigmoid(logits) - labels
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


def hypothesis(features, parameters, sample_axis=AxisIndex.FIRST):
    logits = forward_propagation(features, parameters, sample_axis)
    return activation.sigmoid(logits)


def arbitrator(probas, sample_axis=AxisIndex.FIRST):
    '''Get predicted labels from probabilities.'''
    predicts = np.zeros(probas.shape)
    predicts[probas >= 0.5] = 1
    return predicts


def predict(features, parameters, sample_axis=AxisIndex.FIRST):
    probabilities = hypothesis(features, parameters, sample_axis)
    return arbitrator(probabilities, sample_axis)


def evaluation(data_loader, parameters, sample_axis=AxisIndex.FIRST):
    metric = Metric(arbitrator, 1, sample_axis)
    for features, labels, _ in minibatch_iterator(data_loader, sample_axis, minibatch_size=None):
        probas = hypothesis(features, parameters, sample_axis)
        metric.update_statistics(probas, labels)
    return metric


def main():
    config_logging_yaml()
    epochs_count = 400
    learning_rate = 0.001
    sample_axis = AxisIndex.FIRST

    parameters = initialize_parameters(2, Initializer(Zero()), sample_axis)
    dataset_cost = compute_dataset_cost(get_data_loader(sample_axis), parameters, sample_axis)
    logger.info('Cost at initial parameters (zeros): {}'.format(dataset_cost))

    weight_values = np.array([[0.206], [0.201]])
    if (sample_axis == AxisIndex.LAST):
        weight_values = weight_values.T
    initializer = Initializer(Constant(weight_values), Constant(np.array([[-8.161]])))
    parameters = initialize_parameters(2, initializer, sample_axis)
    dataset_cost = compute_dataset_cost(get_data_loader(sample_axis), parameters, sample_axis)
    logger.info('Cost at test parameters: {}'.format(dataset_cost))

    iterFitPlot = IterativeFitPlot()
    iterCostPlot = IterativeCostPlot(learning_rate, step=5)
    # parameters = initialize_parameters(2, Initializer(RandomNormal(0.0, 0.01)), sample_axis)

    for epoch in range(epochs_count):
        recurrence_mean = RecurrenceMean()
        for mini_batch in minibatch_iterator(get_data_loader(sample_axis), sample_axis, minibatch_size=None):
            mini_features, mini_labels, mini_count = mini_batch
            mini_logits = forward_propagation(mini_features, parameters, sample_axis)
            recurrence_mean(mini_count, compute_cost(mini_logits, mini_labels, sample_axis))
            grads = back_propagation(mini_features, mini_labels, mini_logits, sample_axis)
            parameters = update_parameters(parameters, grads, learning_rate)
            if (epoch == 0):
                iterFitPlot.scatter_samples(mini_features, mini_labels)

        epoch_train_cost = np.squeeze(recurrence_mean.mean)
        logger.info('Cost after epoch {}: {}'.format(epoch, epoch_train_cost))
        iterFitPlot.draw_fit_line(parameters)
        iterCostPlot.update(epoch_train_cost)

    logger.info(parameters)
    training_metric = evaluation(get_data_loader(sample_axis), parameters, sample_axis)
    logger.info('Training accuracy: {}'.format(training_metric.accuracy))
    test_features = np.array([[45, 85]]) if (sample_axis == AxisIndex.FIRST) else np.array([[45], [85]])
    test_probas = hypothesis(test_features, parameters, sample_axis)
    logger.info('The admission probability of student with exam 1 score of 45 and exam 2 score of 85 is {}'.format(test_probas))

    iterFitPlot.close()
    iterCostPlot.close()


if (__name__ == '__main__'):
    main()
