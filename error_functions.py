import numpy as np


def mean_square_error(pred, y):
    return np.mean(np.square(y - pred))


def half_mean_square_error(pred, y):
    return 0.5 * mean_square_error(pred, y)


def half_squared_error(pred, y):
    return 0.5 * np.sum(np.square(pred - y))
