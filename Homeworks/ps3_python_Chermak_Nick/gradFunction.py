import numpy as np
from sigmoid import sigmoid

def gradFunction(theta, X_train, y_train):
    theta.shape = (len(theta),1)
    m = len(y_train)
    h = sigmoid(X_train @ theta)                 #4x3 * #3x1
    gradient = 1 / m * (X_train.T @ (h-y_train)) #needs tp be 3x4 * (4x1 - 4x1)
    return gradient