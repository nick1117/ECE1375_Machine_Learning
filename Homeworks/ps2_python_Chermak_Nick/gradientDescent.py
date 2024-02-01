import numpy as np
from computeCost import computeCost

def gradientDescent(X_train, y_train, alpha, iters):
    m, n_plus1 = X_train.shape # get shape of X_train: m x (n+1)
    theta = np.random.rand(n_plus1, 1)  # random initialization of theta
    cost_history = np.zeros((iters, 1)) # initialize vector with the cost hisory
    for x in range(iters):  
        h = X_train @ theta  # hypothesis function
        error = h - y_train # error part
        #import pdb; pdb.set_trace()
        for n in range(n_plus1):
            gradient = (1/m) * (error.T @ X_train[:,n]) # calc gradient
            theta[n] = theta[n] - (alpha * gradient)
        #import pdb; pdb.set_trace()
        cost_history[x] = computeCost(X_train, y_train, theta) # cost function
    return theta, cost_history