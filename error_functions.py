import numpy as np


def mean_square_error(pred, y):
    return np.mean(np.square(y - pred))


def half_mean_square_error(pred, y):
    return 0.5 * mean_square_error(pred, y)


def sum_squared_error(pred, y):
    return np.sum(np.square(pred - y))


def half_squared_error(pred, y):
    return 0.5 * sum_squared_error(pred, y)


def gradient_half_squared_error(pred, y):
    return pred - y


def cross_entropy_loss(pred, y):
    # also called log loss
    sum1 = 0
    for predicted, actual in zip(pred, y):
        if predicted != 0:
            sum1 += actual * np.log(predicted)
    cross_entropy = -sum1
    return cross_entropy


if __name__ == "__main__":
    pred = [0.02, 0.3, 0.45, 0, 0.25, 0.05, 0]
    true = [0, 0, 0, 0, 1, 0, 0]
    print(cross_entropy_loss(pred, true))
