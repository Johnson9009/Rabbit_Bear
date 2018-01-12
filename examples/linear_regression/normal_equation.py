import os
import logging
import numpy as np
from rabbitbear.utils.logging import config_logging_yaml


logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def load_dataset():
    '''For this exercise we need load dataset from a CSV file.'''
    with open('ex1data2.txt') as f:
        dataset = np.loadtxt(f, delimiter=',')
        return (dataset[:, 0:2], dataset[:, 2:3])


def normal_equation(features, labels):
    return np.dot(np.dot(np.linalg.inv(np.dot(features.T, features)), features.T), labels)


def main():
    config_logging_yaml()
    features, labels = load_dataset()
    features = np.column_stack((np.ones((features.shape[0], 1)), features))
    parameters = normal_equation(features, labels)

    test_features = np.array([[1, 1650, 3]])
    test_predicts = np.dot(test_features, parameters)

    logger.info(parameters)
    logger.info(test_predicts)


if (__name__ == '__main__'):
    main()
