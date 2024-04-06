import numpy as np
import time

def predict(Theta1, Theta2, X):
    #start = time.time()
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    z2 = X_bias @ Theta1.T  
    a2 = 1 / (1 + np.exp(-z2))

    a2_bias = np.hstack([np.ones((a2.shape[0], 1)), a2])
    z3 = a2_bias @ Theta2.T
    a3 = 1 / (1 + np.exp(-z3))
    p = np.argmax(a3, axis=1) + 1
    #end = time.time()
    #print(f'predict time: {end-start}')
    return p, a3