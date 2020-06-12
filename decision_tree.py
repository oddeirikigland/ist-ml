import numpy as np
from math import log2

np.set_printoptions(precision=2)

# information gain = E_start - E
# want highest value


def entropy(values):
    sum1 = 0
    values = np.array(values)
    for v in values:
        sum1 += v * log2(v)
    entropy = -sum1
    print("E({}) = {}".format(values, entropy))
    return entropy


entropy([0.5, 0.5])
entropy([1])
entropy([1 / 3, 2 / 3])
entropy([1 / 6, 1 / 6, 1 / 3, 1 / 3])
entropy([1 / 3, 1 / 3, 1 / 3])
entropy([0.2, 0.4, 0.2, 0.2])
# print(0.6 * entropy([2 / 3, 1 / 3]) + 0.4 * entropy([0.5, 0.5]))


def gini(values):
    sum1 = 0
    for v in values:
        sum1 += v * v
    return 1 - sum1


# print(gini([0.5, 0.5]))
