#get data

import numpy as np
import pandas as pd
import random
from keras.utils import to_categorical
from scipy.optimize import minimize
from costFunction import cFunction,findDeltas
from lazySGD import miniBatchGradientDescent
from sigmoid import sigmoid
from sigmoidDer import siggy
from accuracy import accuracy, predict
import time
from normalize import normalize

with open('trainX.npy', 'rb') as f:
    trainX = np.load(f)

with open('trainY.npy', 'rb') as f:
    trainY = np.load(f)

with open('testX.npy', 'rb') as f:
    testX = np.load(f)

with open('testY.npy', 'rb') as f:
    testY = np.load(f)

validX = trainX[50000:, :]
validX = normalize(validX)
trainX = trainX[0:50000,:]
trainX = normalize(trainX)
testX = normalize(testX)

validY = trainY[50000:]
trainY = trainY[0:50000]

testY = to_categorical(testY)
trainY = to_categorical(trainY)
validY = to_categorical(validY)

#initialize theta matrices
epsilon = .1
theta1 = np.random.rand(50,28*28)
theta1 = epsilon * (2*theta1 - 1)
theta1 = np.concatenate((np.transpose([[1]*50]), theta1), axis=1)

theta2 = np.random.rand(10,50)
theta2 = epsilon * (2*theta2 - 1)
theta2 = np.concatenate((np.transpose([[1]*10]), theta2), axis=1)

thetas = np.concatenate((theta1.flatten(), theta2.flatten()))

#Training the network
startTime = time.time()
trainedNetwork = miniBatchGradientDescent(thetas, [(50,785), (10, 51)], trainX,trainY,.00001, .9, 20,100)
SGDTime = time.time()-startTime

#obj_fun = lambda x: cFunction(x, [(25,785), (10, 26)], trainX,trainY,.001)
#gradd = lambda x: findDeltas(x, [(25,785), (10, 26)], trainX,trainY,.001)

startTime = time.time()
#Very slow, might be because of my functions but I think it's just because it uses the full batch
#rez = minimize(obj_fun, thetas, method="TNC",jac = gradd,options = {"disp":True, "maxfun":100})
scpyTime = time.time()-startTime

print("SGD Took: %s seconds" % SGDTime)
print("SciPy Took: %s seconds" % scpyTime)
print("***********************")
print("Gradient Descent Training Accuracy: ", accuracy(trainX,trainY,[(50,785), (10, 51)],trainedNetwork))
#print("SciPy Training Accuracy: ", accuracy(trainX,trainY,[(25,785), (10, 26)],rez.x))
print("***********************")
print("Gradient Descent Validation Accuracy: ", accuracy(validX,validY,[(50,785), (10, 51)],trainedNetwork))
#print("SciPy Validation Accuracy: ", accuracy(validX,validY,[(25,785), (10, 26)],rez.x))

with open('SGDThetas.npy', 'wb') as f:
    np.save(f, trainedNetwork)
"""
with open('SciPyThetas.npy', 'wb') as f:
    np.save(f, rez.x)
"""