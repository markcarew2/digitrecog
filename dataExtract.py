from keras.datasets import mnist
import numpy as np

(trainX,trainY), (testX,testY) = mnist.load_data()

trainX = trainX.reshape(60000,28*28)
testX = testX.reshape(10000,28*28)

with open('trainX.npy', 'wb') as f:
    np.save(f, trainX)

with open('trainY.npy', 'wb') as f:
    np.save(f, trainY)

with open('testX.npy', 'wb') as f:
    np.save(f, testX)

with open('testY.npy', 'wb') as f:
    np.save(f, testY)

