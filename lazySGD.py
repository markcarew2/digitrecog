from costFunction import costFunction, costy, grady, findDeltas, cFunction
import random
import numpy as np

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def miniBatchGradientDescent(thetas, thetaShapes,X, y, lam,  alpha, iterations, batchSize):
    m = y[:,0].size
    for i in range(iterations):
        shuffle_in_unison(X,y)
        if m > batchSize:
            passes = m // batchSize
            for j in range(passes):
                Xstoch = X[j*batchSize:(j+1)*batchSize,:]
                ystoch = y[j*batchSize:(j+1)*batchSize]

                J, DS = costFunction(thetas,thetaShapes,Xstoch,ystoch, lam)
                thetas = thetas - np.multiply(alpha,DS)

        else:
            randIndList = random.sample(range(0,m-batchSize),iterations)
            for i, index in enumerate(randIndList):
                J, DS = costFunction(thetas,thetaShapes,X,y, lam)
                thetas = thetas - np.multiply(alpha,DS)
        
        print("The cost at iteration %s is: " % i, J)
        
    return thetas