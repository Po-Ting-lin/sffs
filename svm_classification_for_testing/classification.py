import numpy as np


class BinaryClassificationMetrics(object):
    def __init__(self):
        self.recall = None
        self.specificity = None
        self.accuracy = None
        self.area_under_curve = None
        self.tp = None
        self.fp = None
        self.fn = None
        self.tn = None
        self.current_number = None

    def reset_metrics(self):
        self.recall = 0
        self.specificity = 0
        self.accuracy = 0
        self.area_under_curve = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.current_number = 0

    def make_average_metrics(self, other, number):
        self.current_number = number
        self.recall = ((self.current_number - 1) * self.recall + 1 * other[0]) / self.current_number
        self.specificity = ((self.current_number - 1) * self.specificity + 1 * other[1]) / self.current_number
        self.accuracy = ((self.current_number - 1) * self.accuracy + 1 * other[2]) / self.current_number
        self.area_under_curve = ((self.current_number - 1) * self.area_under_curve + 1 * other[3]) / self.current_number
        self.tp = ((self.current_number - 1) * self.tp + 1 * other[4]) / self.current_number
        self.fp = ((self.current_number - 1) * self.fp + 1 * other[5]) / self.current_number
        self.fn = ((self.current_number - 1) * self.fn + 1 * other[6]) / self.current_number
        self.tn = ((self.current_number - 1) * self.tn + 1 * other[7]) / self.current_number

    def __str__(self):
        text = ""
        text += "recall: {}\n".format(np.round(self.recall, 3))
        text += "specificity: {}\n".format(np.round(self.specificity, 3))
        text += "accuracy: {}\n".format(np.round(self.accuracy, 3))
        text += "area_under_curve: {}\n".format(np.round(self.area_under_curve, 3))
        text += "tp: {}\n".format(np.round(self.tp))
        text += "fp: {}\n".format(np.round(self.fp))
        text += "fn: {}\n".format(np.round(self.fn))
        text += "tn: {}".format(np.round(self.tn))
        return text


class Classification(object):
    def __init__(self):
        pass
