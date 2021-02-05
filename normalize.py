import numpy as np

#Normalize data
def normalize(X):
    means = X.mean(axis = 0)
    std = X.std(axis = 0) + .0000001
    normX = (X - means) / std
    return normX