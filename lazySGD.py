from costFunction import costFunction, costy, grady, findDeltas, cFunction
import random
import numpy as np



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
        

        J, DS = costFunction(thetas,thetaShapes,X[0:10000,:],y[0:10000], lam)
        print("The cost at iteration %s is approximately: " % i, J)
        
    return thetas

#Thanks to this post for the shuffle in unison method: 
#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)