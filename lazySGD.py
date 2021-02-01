from costFunction import costFunction, costy, grady, findDeltas, cFunction
import random
import numpy as np

def lazyStochasticGradientDescent(thetas, thetaShapes,X, y, lam,  alpha, iterations, batchSize):
    m = y[:,0].size
    if m > batchSize:
        randIndList = random.sample(range(0,m-batchSize),iterations)
        for i, index in enumerate(randIndList):
            Xstoch = X[index:index+batchSize,:]
            ystoch = y[index:index+batchSize]

            DS = findDeltas(thetas,thetaShapes,Xstoch,ystoch, 1)
            J = cFunction(thetas,thetaShapes,Xstoch,ystoch, 1)
            print("The cost at iteration %s is: " % i, J)
            thetas = thetas - np.multiply(alpha,DS)

    else:
        randIndList = random.sample(range(0,m-batchSize),iterations)
        for i, index in enumerate(randIndList):
            J, DS = costFunction(thetas,thetaShapes,X,y, 1)
            thetas = thetas - np.multiply(alpha,DS)
            print("The cost at iteration %s is: " % i,J)
    return thetas