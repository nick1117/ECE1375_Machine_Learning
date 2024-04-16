import numpy as np
from scipy.spatial.distance import cdist


def kmeans_single(X, K, iters):
    #initialize
    m, n = X.shape
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    means = np.random.rand(K, n) * (maxs - mins) + mins

    for i in range(iters):
        distances = cdist(X, means, 'euclidean')
        ids = np.argmin(distances, axis=1)
        for k in range(K):
            if np.any(ids == k):
                means[k] = np.mean(X[ids == k], axis=0)
    ssd = sum(np.sum((X - means[ids])**2, axis=1))
    
    return ids, means, ssd
