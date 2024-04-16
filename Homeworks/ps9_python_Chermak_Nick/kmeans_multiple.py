from kmeans_single import kmeans_single
import numpy as np

def kmeans_multiple(X, K, iters, R):
    final_ids, final_means, final_ssd = kmeans_single(X, K, iters)

    for i in range(1, R):
        ids, means, ssd = kmeans_single(X, K, iters)
        
        if ssd < final_ssd:
            final_ssd = ssd
            final_ids = ids
            final_means = means

    return final_ids, final_means, final_ssd