import os
import numpy as np
import matplotlib.pyplot as plt
from rabbitbear.dataset import minibatch_iterator


def dataset_plot(x, y):
    ''' Plot the dataset using style of this exercise. '''
    plt.scatter(x, y, c='r', marker='x')
    plt.title('Scatter plot of training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


dataset_filenames = []
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
                dataset_filenames.append(dataset_filename)


def data_loader_iterator():
    ''' For this exercise we need load dataset from several CSV files. '''
    for dataset_filename in dataset_filenames:
        with open(dataset_filename) as f:
            dataset = np.loadtxt(f, delimiter=',')
        yield (dataset[:, 0:1], dataset[:, 1:2])


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


def compute_cost(predicts, labels):
    ''' Compute averaged cost using all samples. '''
    samples_count = labels.shape[0]
    return (1 / (2 * samples_count)) * np.dot(np.transpose(predicts - labels), (predicts - labels))


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
    dataset_generator(1000, 350)
    epochs_count = 200
    learning_rate = 0.01

    parameters = initialize_parameters(1)

    for epoch in range(epochs_count):
        for minibatch_features, minibatch_labels in minibatch_iterator(data_loader_iterator):
            predicts = forward_propagation(minibatch_features, parameters)
            minibatch_cost = compute_cost(predicts, minibatch_labels)
            grads = back_propagation(minibatch_features, minibatch_labels, predicts)
            parameters = update_parameters(parameters, grads, learning_rate)
        if (epoch % 3 == 0):
            print('Current average cost: %f' % minibatch_cost)
    print(parameters)


if (__name__ == '__main__'):
    main()
