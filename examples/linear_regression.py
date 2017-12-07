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


def main():
    dataset_generator(1000, 350)
    for minibatch_X, minibatch_Y in minibatch_iterator(data_loader_iterator):
        pass
        # plt.scatter(minibatch_X, minibatch_Y, c='r', marker='x')
        # plt.show()


if (__name__ == '__main__'):
    main()
