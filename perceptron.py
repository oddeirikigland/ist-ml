import numpy as np
from activation_functions import sgn


def perception_learning(training_data, target_data, weights, bias, learning_rate):
    for i in range(1, 10):
        print("\nEpoch {}".format(i))
        weights_updated = False
        for x, t in zip(training_data, target_data):
            x_with_bias = np.concatenate(([bias], x))
            o = np.dot(x_with_bias, weights)
            pred = sgn(o)
            delta = t - pred
            if delta != 0:
                weights = weights + learning_rate * delta * x_with_bias
                weights_updated = True
            print(
                "o = sgn({}) = {}, delta={}, weights={}".format(o, pred, delta, weights)
            )
        if not weights_updated:
            print("converged")
            break
    return weights


def query(weights, query):
    query_with_w0 = np.concatenate(([1], query))
    o = np.dot(w, query_with_w0)
    pred = sgn(o)
    print(
        "weights: {}, query: {}, pred: o = sgn({}) = {}".format(weights, query, o, pred)
    )
    return pred


x = np.array([[0, 0, 0], [0, 2, 1], [1, 1, 1], [1, -1, 0]])
t = np.array([-1, 1, 1, -1])
w = np.array([1] * (len(x[0]) + 1))
b = 1
learning_rate = 1
w = perception_learning(x, t, w, b, learning_rate)
print("weights: {}".format(w))
query(weights=w, query=[0, 0, 1])
