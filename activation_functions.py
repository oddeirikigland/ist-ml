import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))


def gradient_cross_entropy(x, t, w):
    return -x * (t - sigmoid(np.dot(w, x)))


def hyberbolic_tangent(x):
    return np.tanh(x)


def gradient_hyperbolic_tangent(x):
    return 1 - hyberbolic_tangent(x) ** 2


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def gradient_softmax(x):
    return softmax(x) * (1 - softmax(x))


def sgn(x):
    # if x == 0:
    #     return 0
    if x < 0:
        return -1
    return 1


def step(x):
    return 1 if x >= 0 else 0


if __name__ == "__main__":
    from error_functions import gradient_half_squared_error

    x0 = np.array([1, 1, 0, 0, 0])
    t = np.array([1, 0])
    learning_rate = 1
    activation_function = sigmoid
    gradient_activation_function = sigmoid_gradient
    gradient_error_function = gradient_half_squared_error
    weights_init_value = 0.1
    bias_init_value = 0
    layer_size = [5, 3, 2]

    print("\nForward phase\n")
    w1 = np.full((layer_size[1], len(x0)), fill_value=weights_init_value)
    b1 = np.array([bias_init_value] * layer_size[1])
    z1 = np.array(np.dot(w1, x0) + b1)
    x1 = activation_function(z1)
    print("w1: {}".format(w1))
    print("b1: {}".format(b1))
    print("z1: {}".format(z1))
    print("x1: {}\n".format(x1))

    w2 = np.full((layer_size[2], len(x1)), fill_value=weights_init_value)
    b2 = np.array([bias_init_value] * layer_size[2])
    z2 = np.dot(w2, x1) + b2
    x2 = activation_function(z2)
    print("w2: {}".format(w2))
    print("b2: {}".format(b2))
    print("z2: {}".format(z2))
    print("x2: {}\n".format(x2))

    print("\nBackward phase\n")
    gradient_error = gradient_half_squared_error(x2, t)
    gradient_act_func = gradient_activation_function(z2)
    delta_2 = gradient_error * gradient_act_func
    print("gradient error: {}".format(gradient_error))
    print("gradient act_func: {}".format(gradient_act_func))
    print("delta_2: {}\n".format(delta_2))

    gradient_error = np.dot(w2.T, delta_2)
    gradient_act_func = gradient_activation_function(z1)
    delta_1 = gradient_error * gradient_act_func
    print("gradient error: {}".format(gradient_error))
    print("gradient act_func: {}".format(gradient_act_func))
    print("delta_1: {}\n".format(delta_1))

    print("\nWeight update\n")
    delta_w_1 = (delta_1 * np.array([x0]).T).T
    w1 = w1 - learning_rate * delta_w_1
    b1 = b1 - learning_rate * delta_1
    print("delta_w_1: \n{}".format(delta_w_1))
    print("w1: \n{}".format(w1))
    print("b1: {}\n".format(b1))

    delta_w_2 = (delta_2 * np.array([x1]).T).T
    w2 = w2 - learning_rate * delta_w_2
    b2 = b2 - learning_rate * delta_2
    print("delta_w_2: \n{}".format(delta_w_2))
    print("w2: \n{}".format(w2))
    print("b2: {}".format(b2))
