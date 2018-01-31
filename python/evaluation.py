import numpy as np
from assertpy import assert_that
from .common import AxisIndex


class ConfusionMatrix(object):
    def __init__(self):
        self._matrix = np.zeros((2, 2))

    def update(self, predicts, labels):
        predicts = predicts.squeeze()
        labels = labels.squeeze()
        assert_that(predicts.shape).is_length(1)
        assert_that(labels.shape).is_equal_to(predicts.shape)

        for predict, label in zip(predicts, labels):
            self._matrix[int(predict), int(label)] += 1

    @property
    def samples_count(self):
        return np.sum(self._matrix)

    @property
    def tp(self):
        return self._matrix[1, 1]

    @property
    def fp(self):
        return self._matrix[1, 0]

    @property
    def tn(self):
        return self._matrix[0, 0]

    @property
    def fn(self):
        return self._matrix[0, 1]

    @property
    def accuracy(self):
        return ((self.tp + self.tn) / self.samples_count)

    @property
    def precision(self):
        return (self.tp / (self.tp + self.fp))

    @property
    def recall(self):
        return (self.tp / (self.tp + self.fn))

    def f_beta(self, beta=1):
        return ((1 + np.square(beta)) * ((self.precision * self.recall) / (np.square(beta) * self.precision + self.recall)))
    f1 = property(f_beta)

    @property
    def false_positive_rate(self):
        return (self.fp / (self.fp + self.tn))
    true_positive_rate = recall


class Metric(object):
    def __init__(self, arbitrator, classes_count=1, sample_axis=AxisIndex.FIRST):
        self._arbitrator = arbitrator
        self._classes_count = classes_count
        self._sample_axis = sample_axis
        # The shape of confusion matrices is (num_class, (10 + 1)), 10 for (0, 0.1 ~ 0.9), 1 for threshold.
        self._confusion_maxtrices = np.array([[ConfusionMatrix() for _ in range(11)] for _ in range(classes_count)])

    def __get_metrics(self, threshold_index, type):
        assert_that(type).is_in('precision', 'recall', 'accuracy')

        shape = (1, self._classes_count) if (self._sample_axis == AxisIndex.FIRST) else (self._classes_count, 1)
        metrics = list()
        for i in range(self._classes_count):
            if (type == 'precision'):
                metric = self._confusion_maxtrices[i, threshold_index].precision
            elif (type == 'recall'):
                metric = self._confusion_maxtrices[i, threshold_index].recall
            else:
                metric = self._confusion_maxtrices[i, threshold_index].accuracy
            metrics.append(metric)
        return np.array(metrics).reshape(shape)

    def __get_mean_precision(self, threshold_index, average):
        assert_that(average).is_in('macro', 'micro')
        if (average == 'macro'):
            return np.mean(self.__get_metrics(threshold_index, 'precision'))
        else:
            tps, fps = 0, 0
            for i in range(self._classes_count):
                tps += self._confusion_maxtrices[i, threshold_index].tp
                fps += self._confusion_maxtrices[i, threshold_index].fp
            return (tps / (tps + fps))

    def __get_mean_recall(self, threshold_index, average):
        assert_that(average).is_in('macro', 'micro')
        if (average == 'macro'):
            return np.mean(self.__get_metrics(threshold_index, 'recall'))
        else:
            tps, fns = 0, 0
            for i in range(self._classes_count):
                tps += self._confusion_maxtrices[i, threshold_index].tp
                fns += self._confusion_maxtrices[i, threshold_index].fn
            return (tps / (tps + fns))

    def __get_micro_false_positive_rate(self, threshold_index):
        fps, tns = 0, 0
        for i in range(self._classes_count):
            fps += self._confusion_maxtrices[i, threshold_index].fp
            tns += self._confusion_maxtrices[i, threshold_index].tn
        return (fps / (fps + tns))
    __get_mean_true_positive_rate = __get_mean_recall

    def update_statistics(self, probas, labels):
        assert_that(len(labels.take(0, axis=self._sample_axis))).is_equal_to(self._classes_count)
        assert_that(labels.shape).is_length(2)

        class_axis = AxisIndex.LAST if (self._sample_axis == AxisIndex.FIRST) else AxisIndex.FIRST
        predicts = self._arbitrator(probas, self._sample_axis)
        # The sample only can be classified to one kind.
        assert_that((np.sum(predicts, axis=class_axis) <= 1).all()).is_true()

        for i in range(self._classes_count):
            class_probas = probas.take(i, axis=class_axis)
            class_labels = labels.take(i, axis=class_axis)
            for j in range(10):
                threshold = j / 10
                class_predicts = np.zeros(class_probas.shape)
                class_predicts[class_probas >= threshold] = 1
                self._confusion_maxtrices[i, j].update(class_predicts, class_labels)

            class_predicts = predicts.take(i, axis=class_axis)
            self._confusion_maxtrices[i, -1].update(class_predicts, class_labels)

    @property
    def precisions(self):
        return self.__get_metrics(-1, 'precision')

    @property
    def recalls(self):
        return self.__get_metrics(-1, 'recall')

    @property
    def accuracies(self):
        return self.__get_metrics(-1, 'accuracy')

    def precision(self, average='macro'):
        assert_that(average).is_in('macro', 'micro')
        return self.__get_mean_precision(-1, average)

    def recall(self, average='macro'):
        assert_that(average).is_in('macro', 'micro')
        return self.__get_mean_recall(-1, average)

    @property
    def accuracy(self):
        if (self._classes_count == 1):
            return self._confusion_maxtrices[0, -1].accuracy
        else:
            tps = 0
            for i in range(self._classes_count):
                tps += self._confusion_maxtrices[i, -1].tp
            return (tps / self._confusion_maxtrices[0, -1].samples_count)

    def precision_recall_curve(self, class_index):
        precisions = [self._confusion_maxtrices[class_index, i].precision for i in range(10)].append(1).reverse()
        recalls = [self._confusion_maxtrices[class_index, i].recall for i in range(10)].append(0).reverse()
        return (precisions, recalls)

    @property
    def average_precisions(self):
        shape = (1, self._classes_count) if (self._sample_axis == AxisIndex.FIRST) else (self._classes_count, 1)
        aps = list()
        for i in range(self._classes_count):
            precisions, recalls = self.precision_recall_curve(i)
            aps.append(np.sum(np.diff(recalls) * precisions[1:]))
        return np.array(aps).reshape(shape)

    def mean_average_precision(self, average='macro'):
        assert_that(average).is_in('macro', 'micro')
        if (average == 'macro'):
            return np.mean(self.average_precisions)
        else:
            precisions = [self.__get_mean_precision(i, 'micro') for i in range(10)].append(1).reverse()
            recalls = [self.__get_mean_recall(i, 'micro') for i in range(10)].append(0).reverse()
            return np.sum(np.diff(recalls) * precisions[1:])

    def roc_curve(self, class_index):
        '''Receiver Operating Characteristic curve.'''
        false_positive_rates = [self._confusion_maxtrices[class_index, i].false_positive_rate for i in range(10)].append(0).reverse()
        true_positive_rates = [self._confusion_maxtrices[class_index, i].true_positive_rate for i in range(10)].append(0).reverse()
        return (true_positive_rates, false_positive_rates)

    @property
    def area_under_curves(self):
        shape = (1, self._classes_count) if (self._sample_axis == AxisIndex.FIRST) else (self._classes_count, 1)
        aucs = list()
        for i in range(self._classes_count):
            true_positive_rates, false_positive_rates = self.roc_curve(i)
            aucs.append(np.sum(np.diff(false_positive_rates) * true_positive_rates[1:]))
        return np.array(aucs).reshape(shape)

    def mean_area_under_curve(self, average='macro'):
        assert_that(average).is_in('macro', 'micro')
        if (average == 'macro'):
            return np.mean(self.area_under_curves)
        else:
            false_positive_rates = [self.__get_micro_false_positive_rate(i) for i in range(10)].append(0).reverse()
            true_positive_rates = [self.__get_mean_true_positive_rate(i, 'micro') for i in range(10)].append(0).reverse()
            return np.sum(np.diff(false_positive_rates) * true_positive_rates[1:])

