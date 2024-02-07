import numpy as np


def Reg_normalEqn(X_train, y_train, lam):
    #get shape to make D matrix
    m, n = X_train.shape 
    #create diagonal matrix of Lambdas
    L = lam * np.eye(n,n)
    #set first to 0 for theta0
    L[0, 0] = 0
    #calculate normal equaiton
    theta = np.linalg.pinv(X_train.T @ X_train + L) @ X_train.T @ y_train
    return theta