import numpy as np
from predict import predict


def nnCost(Theta1, Theta2, X, y, K, lam):
    y_matrix = np.eye(K)[y - 1]
    p, h = predict(Theta1, Theta2, X)
    m = X.shape[0]
    cost = -1/m * (np.sum((y_matrix * np.log(h)) + ((1 - y_matrix) * np.log(1 - h))))
    reg = lam/(2*m) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
    J = cost + reg
    return J