import numpy as np
#Sigmoid Function
def sigmoid(X):
    return np.reciprocal(1 + np.exp((-1) * X))
