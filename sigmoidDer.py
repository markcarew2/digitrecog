from sigmoid import sigmoid

#Derivative of Sigmoid
def siggy(X):
    return sigmoid(X) * (1 - sigmoid(X))