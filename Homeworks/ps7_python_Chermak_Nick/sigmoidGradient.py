import numpy as np
from sigmoid import sigmoid

def sigmoidGradient(z):
    g_prime = sigmoid(z) * (1 - sigmoid(z))
    return g_prime