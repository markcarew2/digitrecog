Digit Recognition Neural Network

Based on the NN in the Andrew Ng Machine Learning Course.

Fully vectorized over the examples. 

Properly implemented mini batch gradient descent now. It reaches good validation error very quickly.

For alpha = .9, lambda = .00001, batchSize=100, iterations =50, NN = [(25,785),(10,26)] Validation accuracy at 94.68%, training accuracy at 96.96% bias and variance seem to be approaching each other. 

For alpha = .9, lambda = .00001, batchSize=100, iterations =20, NN = [(50,785),(10,51)], Validation accuracy at 95.45%, training accuracy at 96.352%

Need to play with the parameters, can try other initializations of the weights (initializations that skew to the extremes of the sigmoid function), probably should try more sophisticated learning rates (annealing).

Should add a feature to allow for loading the saved parameters. Should probably try to make this object oriented.

Have commented out code related to SciPy minimize, was slow and ineffective.