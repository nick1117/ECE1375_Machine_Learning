import numpy as np
from scipy.spatial.distance import cdist

def weightedKNN(X_train, y_train, X_test, sigma):
    y_predict = np.zeros(len(X_test))
    d = X_test.shape[0]
    classes = np.unique(y_train)
    for i in range(d):
        X = X_test[i,:]
        X = X.reshape(1,-1)
        E_dist = cdist(X_train, X, metric="euclidean")
        w_i = np.exp(-(E_dist**2)/(sigma**2))
        class_vote = np.zeros(len(classes))
       # import pdb; pdb.settrace()
        for j, c in enumerate(classes):
            class_vote[j] = np.sum(w_i[y_train == c])                
        y_predict[i] = classes[np.argmax(class_vote)]

    return y_predict

