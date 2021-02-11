import numpy as np
from sigmoid import sigmoid
from sigmoidDer import siggy

#Cost Function, takes thetas as 1D array,
#hiddenLayerShapes must be list of tuples, column of first shape must be number of features of X plus one
#If two classes, rows of last shape must be one, otherwise, rows must equal number of classes.

def vectorizedLayerGradient(delta, activation):
    d = np.einsum('ij,jk->ik',delta,activation)

    return d


def costFunction(thetaVector, hiddenLayerShapes, X, y,  lam=0):
    
    layers = len(hiddenLayerShapes)
    thetas= [0]* layers
    thetaIndex = 0
    
    for i,shape in enumerate(hiddenLayerShapes):
        hiddenLayerSize = shape[0] * shape[1]
        thetaNextIndex = thetaIndex + hiddenLayerSize
        thetas[i] = np.reshape(thetaVector[thetaIndex:thetaNextIndex], shape)
        thetaIndex = thetaNextIndex
    
        
    #Set Up useful variables
    m = y.size
    activation = np.transpose(X)
    thetaSum = 0
    zs = [0] * len(thetas)
    
    #Find the Cost J, should work with one or more hidden layers
    for i, theta in enumerate(thetas):
        activation = np.concatenate((np.ones((1, activation.shape[1])), activation))
        zs[i] = np.matmul(theta,activation)
        activation = sigmoid(zs[i])
        thetaSum += np.square((np.delete(theta,0,1))).sum()
    J= np.multiply(np.log(activation), np.transpose(y)).sum() + np.multiply(np.log(1-activation), (1-np.transpose(y))).sum()
    J = (J.sum()) / ((-1)*m)
    J += thetaSum * lam / (2*m)

    ds = [0] * (len(thetas))
    ds[len(thetas)-1] = activation - np.transpose(y)
    for i, theta in enumerate(reversed(thetas)):
        if len(thetas)-2-i >= 0:
            ds[len(thetas)-2-i] = np.multiply(np.matmul(np.delete(theta.T,0,0),ds[len(thetas)-1-i]), siggy(zs[len(thetas)-2-i]))
        else:
            break
    
    Deltas = [0] * len(thetas)
    for i, d in enumerate(ds):
        if i > 0:
            Deltas[i] = (vectorizedLayerGradient(d, np.concatenate((np.ones((X.shape[0],1)), sigmoid(zs[i-1].T)),axis=1)))
            Deltas[i][:,1:] = Deltas[i][:,1:] + lam * thetas[i][:,1:]
            Deltas[i] = Deltas[i]/m
                         
        else:
            observations = np.concatenate((np.ones((X.shape[0],1)), X),axis=1)
            Deltas[i] = vectorizedLayerGradient(d, observations)
            Deltas[i][:,1:] = Deltas[i][:,1:] + lam * thetas[i][:,1:]
            Deltas[i] = Deltas[i] / m
            
    tempArray = np.empty((0))
    for d in Deltas:
        tempArray = np.concatenate((tempArray,d.flatten()))
            
    Deltas = tempArray

    return J, Deltas

#Almost Same as above but only returns cost
def cFunction(thetaVector, hiddenLayerShapes, X, y,  lam=0):
    
    layers = len(hiddenLayerShapes)
    thetas= [0]* layers
    thetaIndex = 0
    
    for i,shape in enumerate(hiddenLayerShapes):
        hiddenLayerSize = shape[0] * shape[1]
        thetaNextIndex = thetaIndex + hiddenLayerSize
        thetas[i] = np.reshape(thetaVector[thetaIndex:thetaNextIndex], shape)
        thetaIndex = thetaNextIndex
    
        
    #Set Up useful variables
    m = y.size
    activation = np.transpose(X)
    thetaSum = 0
    zs = [0] * len(thetas)
    
    #Find the Cost J, should work with one or more hidden layers
    for i, theta in enumerate(thetas):
        activation = np.concatenate((np.ones((1, activation.shape[1])), activation))
        zs[i] = np.matmul(theta,activation)
        activation = sigmoid(zs[i])
        thetaSum += np.square((np.delete(theta,0,1))).sum()
    J= np.multiply(np.log(activation), np.transpose(y)).sum() + np.multiply(np.log(1-activation), (1-np.transpose(y))).sum()
    J = (J.sum()) / ((-1)*m)
    J += thetaSum * lam / (2*m)

    return J

#Almost same as above but just returns deltas.
def findDeltas(thetaVector, hiddenLayerShapes, X, y,  lam=0):
    #Find the gradients

    layers = len(hiddenLayerShapes)
    thetas= [0]* layers
    thetaIndex = 0
    
    for i,shape in enumerate(hiddenLayerShapes):
        hiddenLayerSize = shape[0] * shape[1]
        thetaNextIndex = thetaIndex + hiddenLayerSize
        thetas[i] = np.reshape(thetaVector[thetaIndex:thetaNextIndex], shape)
        thetaIndex = thetaNextIndex
    
        
    m = y.size
    activation = np.transpose(X)
    thetaSum = 0
    zs = [0] * len(thetas)

    for i, theta in enumerate(thetas):
        activation = np.concatenate((np.ones((1, activation.shape[1])), activation))
        zs[i] = np.matmul(theta,activation)
        activation = sigmoid(zs[i])
        thetaSum += np.square((np.delete(theta,0,1))).sum()

    ds = [0] * (len(thetas))
    ds[len(thetas)-1] = activation - np.transpose(y)
    for i, theta in enumerate(reversed(thetas)):
        if len(thetas)-2-i >= 0:
            ds[len(thetas)-2-i] = np.multiply(np.matmul(np.delete(theta.T,0,0),ds[len(thetas)-1-i]), siggy(zs[len(thetas)-2-i]))
        else:
            break
    
    Deltas = [0] * len(thetas)
    for i, d in enumerate(ds):
        if i > 0:
            Deltas[i] = (vectorizedLayerGradient(d, np.concatenate((np.ones((X.shape[0],1)), sigmoid(zs[i-1].T)),axis=1)))
            Deltas[i][:,1:] = Deltas[i][:,1:] + lam * thetas[i][:,1:]
            Deltas[i] = Deltas[i]/m
                         
        else:
            observations = np.concatenate((np.ones((X.shape[0],1)), X),axis=1)
            Deltas[i] = vectorizedLayerGradient(d, observations)
            Deltas[i][:,1:] = Deltas[i][:,1:] + lam * thetas[i][:,1:]
            Deltas[i] = Deltas[i] / m
            
    tempArray = np.empty((0))
    for d in Deltas:
        tempArray = np.concatenate((tempArray,d.flatten()))
            
    Deltas = tempArray
            
    return Deltas

#Wrapper Functions that return only cost or only gradients
#For the scipy minimize function
#This solution didn't end up working so I had to go with the silly one above
#Should probably fix this
def costAlone( aFunc ):
    def pick0( thetaVector, hiddenLayerShapes, X, y,  lam=1):
        return aFunc(thetaVector, hiddenLayerShapes, X, y,  lam=0)[0]
    return pick0

def gradAlone( aFunc ):
    def pick1( thetaVector, hiddenLayerShapes, X, y,  lam=1):
        return aFunc(thetaVector, hiddenLayerShapes, X, y,  lam=0)[1]
    return pick1

costy = costAlone(costFunction)
grady = gradAlone(costFunction)