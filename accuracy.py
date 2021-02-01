import numpy as np
from sigmoid import sigmoid

def predict(X,thetas):
    activation = np.transpose(X)
    for theta in thetas:
        activation = np.concatenate((np.ones((1, activation.shape[1])), activation))
        activation = sigmoid(np.matmul(theta,activation))
    return activation.argmax(axis = 0)


def accuracy(X, y, hiddenLayerShapes, thetaVector):
    layers = len(hiddenLayerShapes)
    thetas= [0]* layers
    thetaIndex = 0
    for i,shape in enumerate(hiddenLayerShapes):
        hiddenLayerSize = shape[0] * shape[1]
        thetaNextIndex = thetaIndex + hiddenLayerSize
        thetas[i] = np.reshape(thetaVector[thetaIndex:thetaNextIndex], shape)
        thetaIndex = thetaNextIndex
        
    guesses = predict(X, thetas)
    answers = y.argmax(axis=1)
    m = len(guesses)
    err = 0
    for i,j in zip(guesses,answers):
        if i != j:
            err += 1
    return (m-err)/m