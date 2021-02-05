Digit Recognition Neural Network

Based on the NN in the Andrew Ng Machine Learning Course.

Fully vectorized over the examples. 

Stochastic Gradient Descent is lazy because it only chooses a random starting point for each batch of examples, instead of choosing every example at random.

Uses both basic gradient descent and an advanced gradient descent method from SciPy.

Instead of having one function that returns both cost and deltas, I have two functions that overlap in work. I did this to get it to work with SciPy optimize but I'm sure there's a better solution.

After some tests the TNC method of minimize doesn't seem to converge faster than SGD. Over 2000 iterations, the test score of SciPY is .9041 and of SGD is .8977. I think the difference is fully explained by the fact that SciPy is running all the data through, this also makes it much, much slower (8 minutes vs 1 hour). A better result can be reached much faster by increasing iterations and batch size of SGD .

Validation accuracy at 94.68%, training accuracy at 96.27% bias and variance seem to be approaching each other. Need to play with the parameters, can try other initializations of the weights (initializations that skew to the extremes of the sigmoid function), probably should try more sophisticated learning rates (annealing).

Should add a feature to allow for loading the saved parameters. Should probably try to make this object oriented.