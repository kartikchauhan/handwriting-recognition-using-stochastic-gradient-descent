# training data-set size: 50,000
# validation data-set size: 10,000
# test data-set size: 10,000

import random
import numpy as np 		# to easily and fastly compute linear algebra functions

class Network(object):
	def __init__(self, sizes):
		self.numLayers = len(sizes)		#get the number of layers in our network
		self.sizes = sizes		# tells number of neurons in each layer
		# assigning random biases and weights to the neurons initially
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]	# size[1:] excludes the first element in the list and iterates through rest of the elements
		self.weights = [np.random.randn(y, x) 		
            for x, y in zip(sizes[:-1], sizes[1:])]		# zip takes two vectors or arrays and is helpful in assigning sliced part of the list to new variables

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent. The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired outputs. If "test_data" is provided then the network will be evaluated
        against the test data after each epoch, and partial progress printed out. This is useful for tracking progress,
        but slows things down substantially."""
        
        if test_data: n_test = len(test_data)	# if test-data present, get the size of the test-data
        n = len(training_data)
        for j in xrange(epochs):		# --not sure about epochs--
            random.shuffle(training_data)	# shuffle the training data everytime minimizing the cost function.
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]		# breaking the training data into mini_batches of size mini_batch_size
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)		# updating mini_batch(biases & weights) 
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta" is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]		# shape returns the dimensions of the list specified
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
        	"""invoking backpropagation algorithm, a fast way of computing gradient of the cost function.
        	--Need to check how backpropagation algo works--"""
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)		
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

# we're using sigmoid function instead of perceptrons because a small change in weights or biases will cause only small change in the final output.
# sigmoid function tells how close our result is. 
# If z->infinity then the final output will be 1 which tells that digit has been correctly recognised. 
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

net = Network([2, 3, 1])
