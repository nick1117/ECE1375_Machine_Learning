import numpy as np

def computeCost(X, y, theta):
    theta.shape = (len(theta),1)
    m = len(y)  # get size of matrix which is equal to m
    h = X @ theta  # hypothesis function
    J = (1/(2*m)) * np.sum((h - y)**2) # cost function
    return J