import numpy as np

def normalEqn(X_train, y_train):
    theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return theta