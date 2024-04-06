import numpy as np
from predict import predict
import time


def nnCost(Theta1, Theta2, X, y, K, lam):
    #start = time.time()
    y_matrix = np.eye(K)[y - 1]
    p, h = predict(Theta1, Theta2, X)
    # m = X.shape[0]
    # cost = -1/m * (np.sum((y_matrix * np.log(h)) + ((1 - y_matrix) * np.log(1 - h))))
    # # reg = lam/(2*m) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
    # reg = lam/(2*m) * (np.sum(Theta1[1:,:]**2) + np.sum(Theta2[1:,:]**2))
    # J = cost + reg

    J = (-1/X.shape[0] * (np.sum((y_matrix * np.log(h)) + ((1 - y_matrix) * np.log(1 - h))))) + (lam/(2*X.shape[0]) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2)))
    #end = time.time()
    #print(f'cost time: {end - start}')
    return J