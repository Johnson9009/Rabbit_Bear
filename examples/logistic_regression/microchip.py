import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from rabbitbear import cost
from rabbitbear.activation import Sigmoid
from rabbitbear.evaluation import Metric
from rabbitbear.common import AxisIndex, RecurrenceMean
from rabbitbear.dataset import minibatch_iterator, StandardScaler
from rabbitbear.initializer import Initializer, Zero, RandomNormal, Constant
from rabbitbear.visualization import IterativeCostPlot
from rabbitbear.utils.logging import config_logging_yaml


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


class IterativeFitPlot(object):
    '''Draw the fitting figure using data got gradually, the fitting line will be changed iteratively.'''
    def __init__(self, lamb_da):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('lambda = {}'.format(lamb_da))
        self.ax.set_xlabel('Microchip Test 1')
        self.ax.set_ylabel('Microchip Test 2')
        self.ax.scatter([], [], c='k', marker='+', label='y = 1')
        self.ax.scatter([], [], c='y', marker='o', label='y = 0')
        self.line, = self.ax.plot([], [], label='Decision boundary')
        self.ax.legend()
        self.test1_meshgrid, self.test2_meshgrid = np.meshgrid(np.linspace(-1, 1.5, 50), np.linspace(-1, 1.5, 50))

    def scatter_samples(self, features, labels):
        accepted = np.where(labels == 1)
        rejected = np.where(labels == 0)
        self.ax.scatter(features[accepted, 0:1], features[accepted, 1:2], c='k', marker='+', label='y = 1')
        self.ax.scatter(features[rejected, 0:1], features[rejected, 1:2], c='y', marker='o', label='y = 0')
        plt.pause(0.000001)

    def draw_decision_boundary(self, parameters):
        features = np.c_[self.test1_meshgrid.reshape(-1, 1), self.test2_meshgrid.reshape(-1, 1)]
        features = map_features(features, 6)
        labels_meshgrid = forward_propagation(features, parameters).reshape(self.test1_meshgrid.shape)
        self.ax.contour(self.test1_meshgrid, self.test2_meshgrid, labels_meshgrid, [0])

    def close(self, hold_before_close=True):
        if (hold_before_close is True):
            plt.show()
        plt.close(self.fig)


def map_features(features, degree):
    polynomial_features = features
    for i in range(2, degree + 1):
        for j in range(i + 1):
            polynomial_features = np.c_[polynomial_features, np.power(features[:, 0:1], i - j) * np.power(features[:, 1:2], j)]
    return polynomial_features


def get_data_loader(sample_axis=AxisIndex.FIRST):
    def data_loader(shuffle=True):
        '''For this exercise we need load dataset from a CSV file.'''
        with open('ex2data2.txt') as f:
            dataset = np.loadtxt(f, delimiter=',')
            features = dataset[:, 0:2]
            labels = dataset[:, 2:3]
            count = dataset.shape[0]
            features = map_features(features, 6)
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


def hypothesis(features, parameters, sample_axis=AxisIndex.FIRST):
    logits = forward_propagation(features, parameters, sample_axis)
    return Sigmoid.forward(logits)


def arbitrator(probas, sample_axis=AxisIndex.FIRST):
    '''Get predicted labels from probabilities.'''
    predicts = np.zeros(probas.shape)
    predicts[probas >= 0.5] = 1
    return predicts


def predict(features, parameters, sample_axis=AxisIndex.FIRST):
    probabilities = hypothesis(features, parameters, sample_axis)
    return arbitrator(probabilities, sample_axis)


def compute_cost(logits, labels, coster):
    ''' Compute averaged cost using all samples. '''
    return coster.forward(logits, labels)


def back_propagation(features, logits, labels, parameters, weight_decay, coster, sample_axis=AxisIndex.FIRST):
    ''' Compute the gradients of parameters. '''
    dZ = coster.backward(logits, labels)
    samples_count = labels.shape[sample_axis]
    grads = {}
    if (sample_axis == AxisIndex.FIRST):
        grads['W'] = (1 / samples_count) * (np.dot(np.transpose(features), dZ) + weight_decay * parameters['W'])
    else:
        grads['W'] = (1 / samples_count) * (np.dot(dZ, np.transpose(features)) + weight_decay * parameters['W'])
    grads['b'] = (1 / samples_count) * np.sum(dZ, keepdims=True)
    return grads


def update_parameters(parameters, grads, learning_rate):
    ''' Update the parameters using its gradients once. '''
    parameters['W'] -= learning_rate * grads['W']
    parameters['b'] -= learning_rate * grads['b']
    return parameters


def fit(parameters, features, labels, maxiter, weight_decay):
    '''Train the model using BFGS optimizer instead of mini-batch gradient descent.'''
    def f(x, features, labels):
        '''This function is responsible for cost computation.'''
        x = np.reshape(x, (-1, 1))
        parameters['W'] = x[1:]
        parameters['b'] = x[0]
        logits = forward_propagation(features, parameters, AxisIndex.FIRST)
        return compute_cost(logits, labels, cost.SigmoidCrossEntropy(AxisIndex.FIRST)).flatten()

    def fprime(x, features, labels):
        '''This function is responsible for gradient computation.'''
        x = np.reshape(x, (-1, 1))
        parameters['W'] = x[1:]
        parameters['b'] = x[0]
        logits = forward_propagation(features, parameters, AxisIndex.FIRST)
        grads = back_propagation(features, logits, labels, parameters, weight_decay, cost.SigmoidCrossEntropy(AxisIndex.FIRST), AxisIndex.FIRST)
        return np.r_[grads['b'], grads['W']].flatten()

    x0 = np.r_[parameters['b'], parameters['W']]
    return fmin_bfgs(f, x0, fprime, (features, labels), maxiter=maxiter)



def evaluation(data_loader, parameters, sample_axis=AxisIndex.FIRST):
    metric = Metric(arbitrator, 1, sample_axis)
    for features, labels, _ in minibatch_iterator(data_loader, sample_axis, minibatch_size=None):
        probas = hypothesis(features, parameters, sample_axis)
        metric.update_statistics(probas, labels)
    return metric


def compute_dataset_cost(data_loader, parameters, coster, sample_axis=AxisIndex.FIRST):
    recurrence_mean = RecurrenceMean()
    for features, labels, count in minibatch_iterator(data_loader, sample_axis, minibatch_size=None):
        logits = forward_propagation(features, parameters, sample_axis)
        recurrence_mean(count, compute_cost(logits, labels, coster))
    return np.squeeze(recurrence_mean.mean)


def main():
    config_logging_yaml()
    epochs_count = 400
    learning_rate = 0.7
    weight_decay = 1
    sample_axis = AxisIndex.FIRST
    train_using_bfgs = True
    coster = cost.SigmoidCrossEntropy(sample_axis)

    parameters = initialize_parameters(27, Initializer(Zero()), sample_axis)
    dataset_cost = compute_dataset_cost(get_data_loader(sample_axis), parameters, coster, sample_axis)
    logger.info('Cost at initial parameters (zeros): {}'.format(dataset_cost))

    iterFitPlot = IterativeFitPlot(weight_decay)
    if (train_using_bfgs == True):
        features, labels, count = next(get_data_loader()())
        iterFitPlot.scatter_samples(features, labels)
        xopt = fit(parameters, features, labels, epochs_count, weight_decay)
        xopt = np.reshape(xopt, (-1, 1))
        parameters['b'] = xopt[0]
        parameters['W'] = xopt[1:]
    else:
        iterCostPlot = IterativeCostPlot(learning_rate, step=5)

        for epoch in range(epochs_count):
            recurrence_mean = RecurrenceMean()
            for mini_batch in minibatch_iterator(get_data_loader(sample_axis), sample_axis, minibatch_size=None):
                mini_features, mini_labels, mini_count = mini_batch
                mini_logits = forward_propagation(mini_features, parameters, sample_axis)
                recurrence_mean(mini_count, compute_cost(mini_logits, mini_labels, coster))
                grads = back_propagation(mini_features, mini_logits, mini_labels, parameters, weight_decay, coster, sample_axis)
                parameters = update_parameters(parameters, grads, learning_rate)
                if (epoch == 0):
                    iterFitPlot.scatter_samples(mini_features, mini_labels)

            epoch_train_cost = np.squeeze(recurrence_mean.mean)
            logger.info('Cost after epoch {}: {}'.format(epoch, epoch_train_cost))
            iterCostPlot.update(epoch_train_cost)

        iterCostPlot.close(False)

    iterFitPlot.draw_decision_boundary(parameters)
    logger.info(parameters)
    training_metric = evaluation(get_data_loader(sample_axis), parameters, sample_axis)
    logger.info('Training accuracy: {}'.format(training_metric.accuracy))
    iterFitPlot.close()


if (__name__ == '__main__'):
    main()
