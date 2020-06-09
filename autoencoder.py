import numpy as np
from error_functions import half_squared_error

learning_rate = 0.001
x_0 = np.array([8, 0, 4])
print("Original value: {}".format(x_0))

# Forward step
w_1 = np.full((2, 3), 1)
b_1 = np.array([0] * 2)
z_1 = np.dot(w_1, x_0) + b_1
x_1 = z_1

w_2 = np.full((3, 2), 1)
b_2 = np.array([0] * 3)
z_2 = np.dot(w_2, x_1) + b_2
x_2 = z_2

e = half_squared_error(pred=x_2, y=x_0)

# Backward step
delta_2 = (x_2 - x_0) * [1, 1, 1]  # deltaError * deltaX_2
delta_1 = np.dot(np.transpose(w_2), delta_2) * [1, 1]  # deltaZ_2 * delta_2 * deltaX

gradient_w_1 = np.transpose(delta_1 * np.transpose([x_0]))
gradient_b_1 = delta_1
gradient_w_2 = np.transpose(delta_2 * np.transpose([x_1]))
gradient_b_2 = delta_2

# Update weights
w_1 = w_1 - learning_rate * gradient_w_1
b_1 = b_1 - learning_rate * gradient_b_1
w_2 = w_2 - learning_rate * gradient_w_2
b_2 = b_2 - learning_rate * gradient_b_2

# Encode
z_1 = np.dot(w_1, x_0) + b_1
x_1 = z_1
print("Encoded value: {}".format(x_1))

# Decode
z_2 = np.dot(w_2, x_1) + b_2
x_2 = z_2
print("Decoded value: {}".format(x_2))

e = half_squared_error(pred=x_2, y=x_0)
print("Half squared error: {}".format(e))
