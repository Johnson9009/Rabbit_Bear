import os
import logging
import matplotlib.pyplot as plt
from assertpy import assert_that


logger = logging.getLogger(__name__)


class IterativeCostPlot(object):
    '''Draw the cost changing along with training epoch increasing using data got gradually,
    costs of the training and validating datasets will be displayed, and all the lines can
    be drawn in real time.'''
    def __init__(self, learning_rate, has_validate=False, step=1):
        self.has_validate = has_validate
        self.step = step
        self.epoch_index = 0
        self.train_costs = []
        self.validate_costs = []
        self.epochs = []
        self.xlim_max = 10 * step
        self.ylim_max = 0
        self.ylim_min = float('inf')
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Cost Changing During Learning' + os.linesep +
                          'learning rate = %f' % learning_rate)
        self.ax.set_xlabel('epoch')
        self.ax.set_ylabel('cost')
        self.ax.set_xlim(0, self.xlim_max)
        self.train_cost_line, = self.ax.plot(self.epochs, self.train_costs, label='Train Set')
        if (has_validate is True):
            self.validate_cost_line, = self.ax.plot(self.epochs, self.validate_costs, label='Validate Set')
        self.ax.legend()

    def update(self, train_cost, validate_cost=None):
        assert_that(((self.has_validate is True) and (validate_cost is not None)) or
                    ((self.has_validate is False) and (validate_cost is None)),
                    'has_validate: {}, validate_cost: {}'.format(self.has_validate, validate_cost)).is_true()
        if (self.epoch_index % self.step == 0):
            self.epochs.append(self.epoch_index)
            self.train_costs.append(train_cost)
            self.train_cost_line.set_data(self.epochs, self.train_costs)

            if (self.has_validate is True):
                self.validate_costs.append(validate_cost)
                self.validate_cost_line.set_data(self.epochs, self.validate_costs)
            else:
                validate_cost = train_cost

            max_cost = max(train_cost, validate_cost)
            min_cost = min(train_cost, validate_cost)
            if (max_cost * 1.3 >= self.ylim_max):
                self.ylim_max = max_cost * 1.3
                self.ax.set_ylim(top=self.ylim_max)
            if (min_cost * 0.7 <= self.ylim_min):
                self.ylim_min = min_cost * 0.7
                self.ax.set_ylim(bottom=self.ylim_min)
            plt.pause(0.000001)

        self.epoch_index += 1
        if ((self.epoch_index + 3 * self.step) >= self.xlim_max):
            self.xlim_max += 10 * self.step
            self.ax.set_xlim(0, self.xlim_max)

    def close(self, hold_before_close=True):
        if (hold_before_close is True):
            plt.show()
        plt.close(self.fig)
