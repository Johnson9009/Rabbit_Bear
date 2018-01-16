import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from assertpy import assert_that
from rabbitbear import cost
from rabbitbear.common import AxisIndex
from rabbitbear.dataset import minibatch_iterator
from rabbitbear.visualization import IterativeCostPlot
from rabbitbear.utils.logging import config_logging_yaml
from mpl_toolkits.mplot3d import Axes3D


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def dataset_plot(x, y):
    ''' Plot the dataset using style of this exercise. '''
    plt.scatter(x, y, c='r', marker='x')
    plt.title('Scatter plot of training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


class IterativeFitPlot(object):
    '''Draw the fitting figure using data got gradually, the fitting line will be changed iteratively.'''
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Scatter plot of training data')
        self.ax.set_xlabel('Population of City in 10,000s')
        self.ax.set_ylabel('Profit in $10,000s')
        self.ax.scatter([], [], c='r', marker='x', label='Training data')
        self.line, = self.ax.plot([], [], label='Linear regression')
        self.ax.legend()

    def scatter_samples(self, x, y):
        self.ax.scatter(x, y, c='r', marker='x', label='Training data')
        plt.pause(0.000001)

    def draw_fit_line(self, parameters):
        x = self.ax.get_xticks().reshape((-1, 1))
        self.line.set_data(x, forward_propagation(x, parameters))
        plt.pause(0.000001)

    def close(self):
        plt.close(self.fig)


def cost_visualization(data_loader, parameters, sample_axis=AxisIndex.FIRST):
    meshgrid_w, meshgrid_b = np.meshgrid(np.linspace(-1, 4, 10), np.linspace(-10, 10, 10))
    meshgrid_costs = []
    for w, b in zip(np.ravel(meshgrid_w), np.ravel(meshgrid_b)):
        cur_parameters = {'W' : w, 'b' : b}
        meshgrid_costs.append(compute_dataset_cost(data_loader, cur_parameters, sample_axis))
    meshgrid_costs = np.reshape(meshgrid_costs, meshgrid_w.shape)

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle('Visualization of Cost Function')

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(meshgrid_b, meshgrid_w, meshgrid_costs)
    ax.set_title('Surface')
    ax.set_xlabel('b')
    ax.set_ylabel('w')

    ax = fig.add_subplot(1, 2, 2)
    ax.contour(meshgrid_b, meshgrid_w, meshgrid_costs, np.logspace(-2, 3, 20))
    ax.set_title('Contour, showing minimum')
    ax.set_xlabel('b')
    ax.set_ylabel('w')

    ax.plot(parameters['b'], parameters['W'], c='r', marker='x')
    plt.show()


def dataset_generator(dataset_size, dataset_per_file):
    ''' Generate dataset used in this exercise according to the dataset in file "ex1data1.txt". '''
    with open('ex1data1.txt') as f:
        dataset = np.loadtxt(f, delimiter=',')
        dataset_plot(dataset[:, 0:1], dataset[:, 1:2])
        generated_X = np.std(dataset[:, 0:1]) * np.random.randn(dataset_size, 1) + np.mean(dataset[:, 0:1])
        generated_Y = 1.1664 * generated_X - 3.6303
        generated_Y += np.random.randn(dataset_size, 1) * np.mean(dataset[:, 1:2])
        dataset_plot(generated_X, generated_Y)
        i = 0
        while (i < dataset_size):
            dataset_filename = 'dataset%d.txt' % (i // dataset_per_file)
            with open(dataset_filename, 'w') as generated_file:
                for _ in range(min(dataset_per_file, dataset_size - i)):
                    generated_file.write('%f,%f' % (generated_X[i, 0], generated_Y[i, 0]) + os.linesep)
                    i += 1


def get_data_loader(dataset_type):
    assert_that(dataset_type).is_in('train', 'validate', 'test')
    dataset_dir = 'dataset' + os.path.sep + dataset_type
    dataset_file_count = len(os.listdir(dataset_dir))
    dataset_filenames = ['%sdataset%d.txt' % (dataset_dir + os.path.sep, x) for x in range(dataset_file_count)]
    logger.debug('dataset_type: %s dataset_filenames: %s' % (dataset_type, dataset_filenames))

    def data_loader(shuffle=True):
        ''' For this exercise we need load dataset from several CSV files. '''
        for dataset_filename in dataset_filenames:
            with open(dataset_filename) as f:
                dataset = np.loadtxt(f, delimiter=',')
            yield (dataset[:, 0:1], dataset[:, 1:2])

    return data_loader


def initialize_parameters(features_count):
    ''' Initializing all parameters this model needed. '''
    parameters = {}
    parameters['W'] = np.random.randn(features_count, 1)
    parameters['b'] = np.zeros((1, 1))
    return parameters


def forward_propagation(features, parameters):
    ''' Using inputs and parameters to compute the final predictions. '''
    W = parameters['W']
    b = parameters['b']
    return np.dot(features, W) + b


def compute_cost(predicts, labels, sample_axis=AxisIndex.FIRST, weight=1):
    ''' Compute averaged cost using all samples. '''
    return cost.l2_loss(predicts, labels, sample_axis, weight=weight)


def compute_dataset_cost(data_loader, parameters, sample_axis=AxisIndex.FIRST):
    samples_count = 0
    costs = []
    for features, labels in minibatch_iterator(data_loader, sample_axis, minibatch_size=None):
        predicts = forward_propagation(features, parameters)
        samples_count += features.shape[sample_axis]
        costs.append(compute_cost(predicts, labels, weight=features.shape[sample_axis]))
    return np.sum(np.divide(costs, samples_count))


def back_propagation(features, labels, predicts):
    ''' Compute the gradients of parameters. '''
    dZ = predicts - labels
    samples_count = labels.shape[0]
    grads = {}
    grads['W'] = (1 / samples_count) * np.dot(np.transpose(features), dZ)
    grads['b'] = (1 / samples_count) * np.sum(dZ)
    return grads


def update_parameters(parameters, grads, learning_rate):
    ''' Update the parameters using its gradients once. '''
    parameters['W'] -= learning_rate * grads['W']
    parameters['b'] -= learning_rate * grads['b']
    return parameters


def main():
    config_logging_yaml()
    # dataset_generator(10000, 550)
    epochs_count = 200
    learning_rate = 0.01

    iterFitPlot = IterativeFitPlot()
    iterCostPlot = IterativeCostPlot(learning_rate, has_validate=True, step=5)
    parameters = initialize_parameters(1)

    for epoch in range(epochs_count):
        minibatch_costs = []
        for minibatch_features, minibatch_labels in minibatch_iterator(get_data_loader('train'), drop_tail=True):
            predicts = forward_propagation(minibatch_features, parameters)
            minibatch_costs.append(compute_cost(predicts, minibatch_labels))
            grads = back_propagation(minibatch_features, minibatch_labels, predicts)
            parameters = update_parameters(parameters, grads, learning_rate)
            if (epoch == 0):
                iterFitPlot.scatter_samples(minibatch_features, minibatch_labels)

        epoch_train_cost = np.mean(minibatch_costs)
        epoch_validate_cost = compute_dataset_cost(get_data_loader('validate'), parameters)
        logger.info('Cost after epoch %d: %f' % (epoch, epoch_train_cost))
        iterFitPlot.draw_fit_line(parameters)
        iterCostPlot.update(epoch_train_cost, epoch_validate_cost)

    logger.debug(parameters)
    iterFitPlot.close()
    iterCostPlot.close()
    cost_visualization(get_data_loader('test'), parameters)


if (__name__ == '__main__'):
    main()
