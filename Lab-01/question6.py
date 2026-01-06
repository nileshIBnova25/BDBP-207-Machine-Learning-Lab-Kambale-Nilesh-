# Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are three independent variables.
# Compute the gradient of y at a few points and print the values.
import numpy as np

# Define the function
def y(x1, x2, x3):
    return 2*x1 + 3*x2 + 3*x3 + 4

# Gradient of y (constant for all points)
def gradient():
    return np.array([2, 3, 3])

# Sample points
points = [
    (0, 0, 0),
    (1, 2, 3),
    (-1, 4, 0)
]

# Compute and print values
for p in points:
    value = y(*p)
    grad = gradient()
    print(f"At point {p}: y = {value}, gradient = {grad}")

