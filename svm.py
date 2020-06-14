import numpy as np
from activation_functions import sgn

np.set_printoptions(precision=3)


def svm_weight_vector(weight_vector, bias, query):
    return sgn(np.dot(np.transpose(weight_vector), query) + bias)


def svm_coefficient_formula(training_data, coefficients, targets, bias, query):
    return sgn(
        np.sum(
            [
                t * c * np.dot(np.transpose(query), x)
                for x, c, t in zip(training_data, coefficients, targets)
            ]
        )
        + bias
    )


def get_weight_vector(training_data, coefficients, targets):
    return np.sum(
        [t * c * x for x, c, t in zip(training_data, coefficients, targets)], axis=0
    )


def get_bias(training_example, target, weight_vector):
    return target - np.dot(np.transpose(weight_vector), training_example)


def get_margin(weight_vector):
    return 1 / np.linalg.norm(weight_vector)


def get_coefficients(training_data, targets, bias):
    coeff_matrix = []
    for x in training_data:
        row = []
        for y, t in zip(training_data, targets):
            row.append(t * np.dot(x, y))
        coeff_matrix.append(row)
    a = np.array(coeff_matrix)
    b = targets - bias
    return np.linalg.solve(a, b)


def task_1():
    query = np.array([1, 1, 8])
    res = svm_coefficient_formula(
        training_data=np.array([[0, 1, 8], [1, 0, 6], [1, 1, 7], [1, 1, 3]]),
        coefficients=np.array([1, 0.5, 1, 0.5]),
        targets=np.array([1, -1, -1, 1]),
        bias=-3,
        query=query,
    )
    print("query: {} classified as {}".format(query, res))


def task_2():
    training_data = np.array([[2.07, 0.91], [1.41, 0.51]])
    coefficients = np.array([3.36, 3.36])
    targets = np.array([1, -1])

    weight_vector = get_weight_vector(
        training_data=training_data, coefficients=coefficients, targets=targets,
    )
    print("weight_vector: {}".format(weight_vector))

    bias = get_bias(
        training_example=np.array([2.07, 0.91]), target=1, weight_vector=weight_vector
    )

    print(
        "Boundary equation: {}x1 + {}x2 + {}".format(
            weight_vector[0], weight_vector[1], bias
        )
    )

    margin = get_margin(weight_vector)
    print("Margin: {}".format(margin))

    query = np.array([1, 3])

    classify_coef = svm_coefficient_formula(
        training_data=training_data,
        coefficients=coefficients,
        targets=targets,
        bias=bias,
        query=query,
    )

    classify_w_v = svm_weight_vector(
        weight_vector=weight_vector, bias=bias, query=query
    )

    print(
        "{} classified as {} with coefficient formula, classifed as {} with weight vector".format(
            query, classify_coef, classify_w_v
        )
    )


def task_3():
    training_data = np.array(
        [
            [0.68, 1.26],
            [-0.25, -0.51],
            [0.83, -0.32],
            [1.44, 1.04],
            [1.14, 3.21],
            [-2.19, 5.1],
            [-2.47, 3.76],
            [-1.19, 6.39],
            [-2.3, 3.48],
            [-0.94, 5.04],
        ]
    )
    targets = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    weight_vector = np.array([0.544, -0.474])
    bias = 1.902
    for x, t in zip(training_data, targets):
        pred = np.dot(np.transpose(weight_vector), x) + bias
        if np.abs(pred - t) < 0.01:
            print("{} is a support vector!".format(x))

    coefficients = get_coefficients(
        training_data=np.array([[1.14, 3.21], [-2.3, 3.48], [-0.94, 5.04]]),
        targets=np.array([1, -1, -1]),
        bias=bias,
    )
    print("Coefficients: {}".format(coefficients))

    margin = get_margin(weight_vector)
    print("Margin: {}".format(margin))


if __name__ == "__main__":
    # task_1()
    # task_2()
    # task_3()
    pass
