import numpy as np
from activation_functions import (
    sigmoid,
    sigmoid_gradient,
    hyberbolic_tangent,
    gradient_hyperbolic_tangent,
)
from error_functions import half_mean_square_error

# LAB 4
# Task 1
# def forward_phase(x_0, w_1, b_1, w_2, b_2):
#     z_1 = np.dot(w_1, x_0) + b_1
#     x_1 = sigmoid(z_1)
#     z_2 = np.dot(w_2, x_1) + b_2
#     x_2 = sigmoid(z_2)
#     return [x_0], z_1, [x_1], z_2, x_2


# def backward_phase(x_2, t, z_2, w_2, z_1):
#     delta_2 = (x_2 - t) * sigmoid_gradient(z_2)
#     delta_1 = np.dot(np.transpose(w_2), delta_2) * sigmoid_gradient(z_1)
#     return delta_2, delta_1


# def update_weights(x_0, x_1, w_1, b_1, w_2, b_2, delta_1, delta_2, learning_rate=1):
#     delta_e_w_1 = np.transpose(delta_1 * np.transpose(x_0))
#     w_1 = w_1 - learning_rate * delta_e_w_1
#     b_1 = b_1 - learning_rate * delta_1

#     delta_e_w_2 = np.transpose(delta_2 * np.transpose(x_1))
#     w_2 = w_2 - learning_rate * delta_e_w_2
#     b_2 = b_2 - learning_rate * delta_2
#     return w_1, b_1, w_2, b_2


# def task_1():
#     x_0 = np.array([1, 1, 0, 0, 0])
#     w_1 = np.full((3, 5), 0.1)
#     b_1 = np.array([0, 0, 0])
#     w_2 = np.full((2, 3), 0.1)
#     b_2 = np.array([0, 0])

#     t = np.array([1, 0])
#     x_0, z_1, x_1, z_2, x_2 = forward_phase(x_0, w_1, b_1, w_2, b_2)
#     delta_2, delta_1 = backward_phase(x_2, t, z_2, w_2, z_1)
#     w_1, b_1, w_2, b_2 = update_weights(x_0, x_1, w_1, b_1, w_2, b_2, delta_1, delta_2)

#     _, _, _, _, res = forward_phase([1, 0, 0, 0, 1], w_1, b_1, w_2, b_2)
#     # Label with highest output is the predicted one
#     print(res)


def task_2():
    learning_rate = 0.1

    def two_a(x_0, t, w_1, w_2, w_3, b_1, b_2, b_3):
        z_1 = np.dot(w_1, x_0) + b_1
        x_1 = hyberbolic_tangent(z_1)

        z_2 = np.dot(w_2, x_1) + b_2
        x_2 = hyberbolic_tangent(z_2)

        z_3 = np.dot(w_3, x_2) + b_3
        x_3 = hyberbolic_tangent(z_3)

        delta_3 = (x_3 - t) * gradient_hyperbolic_tangent(z_3)
        delta_2 = (np.dot(np.transpose(w_3), delta_3)) * gradient_hyperbolic_tangent(
            z_2
        )
        delta_1 = (np.dot(np.transpose(w_2), delta_2)) * gradient_hyperbolic_tangent(
            z_1
        )

        w_1_change = np.transpose(delta_1 * np.transpose([x_0]))
        w_1 = w_1 - learning_rate * w_1_change
        b_1 = b_1 - learning_rate * delta_1

        w_2_change = np.transpose(delta_2 * np.transpose([x_1]))
        w_2 = w_2 - learning_rate * w_2_change
        b_2 = b_2 - learning_rate * delta_2

        w_3_change = np.transpose(delta_3 * np.transpose([x_2]))
        w_3 = w_3 - learning_rate * w_3_change
        b_3 = b_3 - learning_rate * delta_3
        return (
            w_1_change,
            w_2_change,
            w_3_change,
            delta_1,
            delta_2,
            delta_3,
            w_1,
            b_1,
            w_2,
            b_2,
            w_3,
            b_3,
            x_3,
        )

    w_1_init = np.full((4, 4), 0.1)
    w_2_init = np.full((3, 4), 0.1)
    w_3_init = np.full((3, 3), 0.1)

    b_1_init = [0.1, 0.1, 0.1, 0.1]
    b_2_init = [0.1, 0.1, 0.1]
    b_3_init = [0.1, 0.1, 0.1]

    (
        w_1_change_a,
        w_2_change_a,
        w_3_change_a,
        delta_1_a,
        delta_2_a,
        delta_3_a,
        w_1_a,
        b_1_a,
        w_2_a,
        b_2_a,
        w_3_a,
        b_3_a,
        _,
    ) = two_a(
        [1, 0, 1, 0],
        [0, 1, 0],
        w_1_init,
        w_2_init,
        w_3_init,
        b_1_init,
        b_2_init,
        b_3_init,
    )

    # Task two b adds on top of two a
    (
        w_1_change_b,
        w_2_change_b,
        w_3_change_b,
        delta_1_b,
        delta_2_b,
        delta_3_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = two_a(
        [0, 0, 10, 0],
        [0, 0, 1],
        w_1_init,
        w_2_init,
        w_3_init,
        b_1_init,
        b_2_init,
        b_3_init,
    )
    w_1 = w_1_init - learning_rate * (w_1_change_a + w_1_change_b)
    b_1 = b_1_init - learning_rate * (delta_1_a + delta_1_b)
    w_2 = w_2_init - learning_rate * (w_2_change_a + w_2_change_b)
    b_2 = b_2_init - learning_rate * (delta_2_a + delta_2_b)
    w_3 = w_3_init - learning_rate * (w_3_change_a + w_3_change_b)
    b_3 = b_3_init - learning_rate * (delta_3_a + delta_3_b)

    # Compare models
    x = [1, 1, 1, 0]
    t = [0, 0, 1]
    _, _, _, _, _, _, _, _, _, _, _, _, res, = two_a(
        x, t, w_1_a, w_2_a, w_3_a, b_1_a, b_2_a, b_3_a,
    )
    print("pred model a: {}".format(res))
    _, _, _, _, _, _, _, _, _, _, _, _, res_model_b, = two_a(
        x, t, w_1, w_2, w_3, b_1, b_2, b_3,
    )
    print("pred model b: {}".format(res_model_b))

    a = 2


task_2()
