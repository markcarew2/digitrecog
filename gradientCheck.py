from costFunction import costFunction

#Method to estimate gradients numerically
#Used once to check that gradient function is working correctly
#It passed this and I turned it off
def gradientCheck(thetas, X, y):
    epsilon = .0000001
    gradientEstimate = [0] * len(thetas)
    for i, theta in enumerate(thetas):
        gradientEstimate[i] = theta.flatten()
        for j, ele in enumerate(theta.flatten()):
            thetaPlus = theta.flatten()
            thetaPlus[j] = ele + epsilon
            thetaPlus = thetaPlus.reshape(theta.shape)
            tempPlus = list(thetas)
            tempPlus[i] = thetaPlus
            thetaPlus = tempPlus
            
            
            thetaMinus = theta.flatten()
            thetaMinus[j] = ele - epsilon
            thetaMinus = thetaMinus.reshape(theta.shape)
            
            tempMinus = list(thetas)
            tempMinus[i] = thetaMinus
            thetaMinus = tempMinus
    
            gradientEstimate[i][j] = (costFunction(X,y,thetaPlus,0)[0] - costFunction(X,y,thetaMinus,0)[0]) / (2*epsilon)
        gradientEstimate[i]=gradientEstimate[i].reshape(theta.shape)
    
    return gradientEstimate