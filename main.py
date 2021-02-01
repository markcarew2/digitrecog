#get data

import numpy as np
import pandas as pd
import random
from keras.utils import to_categorical
from scipy.optimize import minimize
from costFunction import cFunction,findDeltas
from lazySGD import lazyStochasticGradientDescent
from sigmoid import sigmoid
from sigmoidDer import siggy
from accuracy import accuracy, predict
import multiprocessing
import time

with open('trainX.npy', 'rb') as f:
    trainX = np.load(f)

with open('trainY.npy', 'rb') as f:
    trainY = np.load(f)

with open('testX.npy', 'rb') as f:
    testX = np.load(f)

with open('testY.npy', 'rb') as f:
    testY = np.load(f)

validX = trainX[50000:, :]
trainX = trainX[0:50000,:]
validY = trainY[50000:]
trainY = trainY[0:50000]

testY = to_categorical(testY)
trainY = to_categorical(trainY)
validY = to_categorical(validY)

#initialize theta matrices
epsilon = .1
theta1 = np.random.rand(25,28*28)
theta1 = epsilon * (2*theta1 - 1)
theta1 = np.concatenate((np.transpose([[1]*25]), theta1), axis=1)

theta2 = np.random.rand(10,25)
theta2 = epsilon * (2*theta2 - 1)
theta2 = np.concatenate((np.transpose([[1]*10]), theta2), axis=1)

thetas = np.concatenate((theta1.flatten(), theta2.flatten()))

startTime = time.time()
trainedNetwork = lazyStochasticGradientDescent(thetas, [(25,785), (10, 26)], trainX,trainY,1, .999, 3000,10000)
SGDTime = time.time()-startTime

obj_fun = lambda x: cFunction(x, [(25,785), (10, 26)], trainX,trainY,1)
gradd = lambda x: findDeltas(x, [(25,785), (10, 26)], trainX,trainY,1)

startTime = time.time()
rez = minimize(obj_fun, thetas, method="TNC",jac = gradd,options = {"disp":True})
scpyTime = time.time()-startTime

print("SGD Took: %s seconds" % SGDTime)
print("SciPy Took: %s seconds" % scpyTime)
print("***********************")
#print("Gradient Descent Test Accuracy: ", accuracy(testX,testY,[(25,785), (10, 26)],trainedNetwork))
print("SciPy Test Accuracy: ", accuracy(testX,testY,[(25,785), (10, 26)],rez.x))

with open('SGDThetas.npy', 'wb') as f:
    np.save(f, trainedNetwork)

with open('SciPyThetas.npy', 'wb') as f:
    np.save(f, rez.x)
