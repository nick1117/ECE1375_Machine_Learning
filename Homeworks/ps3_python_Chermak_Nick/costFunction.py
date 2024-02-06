import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X_train, y_train):
    theta.shape = (len(theta),1)
    m = len(y_train)
    #print('c',X_train.shape,theta.shape)
    #print(theta)
    h = sigmoid(X_train @ theta) # 4x3 * 3x1
    J = (-1/m) * ( (y_train.T @ np.log(h)) + (1-y_train.T) @ np.log(1-h) )  # (1x4 * 4x1) + (1x4 * 4x1)
    return J