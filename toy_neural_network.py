# import dependencies
import numpy as np
import matplotlib.pyplot as plt

'''
What:
    -create a toy example of a vanilla neural network.
Why?
    -learn
    -proof of concept

NN consists of 3 parts:
1. representation - define a set of hypothesis space for the learner (score function form)
2. evaluation - distinguish good classifiers from the bad ones (cost function to evaluate a classifier)
3. optimization - search among the classifiers in the hypothesis space for the highest scoring one (GD and backprop)
easy peasy.

what to do?
1) create neural network class
2) run on linearly separable data

'''
# Create neural_network class - a simple 2-layer feed-forward neural network
class neural_network(object):
    def __init__(self, x, y, step, regularization, neurons_per_layer, iter):
        self.x = x
        self.y = y
        self.step = step
        # regularization parameter
        self.regularization = regularization
        # number of neurons on the hidden layer
        self.neurons_per_layer = neurons_per_layer
        # number of iterations
        self.iter = iter
        self.number_of_classes = len(set(self.y.tolist()))
        # initialize weights
        self.W1 = 0.1 * np.random.randn(self.x.shape[1], self.neurons_per_layer)
        self.b1 = np.zeros((1,self.neurons_per_layer))
        self.W2 = 0.1 * np.random.randn(self.neurons_per_layer, self.number_of_classes)
        self.b2 = np.zeros((1,self.number_of_classes))
        # create empty lists to visualize loss and accuracy
        self.viz_loss = []
        self.viz_accuracy = []

    def train(self):
        for i in xrange(self.iter):
            # compute forward pass w/ relu activation function
            f = lambda x: np.maximum(0, x)
            h1 = f(np.dot(self.x, self.W1) + self.b1)
            scores = np.dot(h1, self.W2) + self.b2  # NB that we don't use activation function on the output layer
            # compute the loss
            scores -= np.max(scores)
            probs = np.exp(scores) / np.sum(np.exp(scores),axis=1, keepdims=True)
            log_probs = -np.log(probs[range(self.x.shape[0]),self.y])
            softmax_loss = np.sum(log_probs)/self.x.shape[0]
            regularization_loss = (0.5 * self.regularization * np.sum(self.W1*self.W1)
                                  + 0.5 * self.regularization * np.sum(self.W2 * self.W2))
            loss = softmax_loss + regularization_loss

            self.viz_loss.append(loss)
            self.viz_accuracy.append(np.mean((np.argmax(scores, axis=1)) == y))

            if i % 1000 == 0:
                print "iteration %d: loss %f" % (i, loss)
            # compute gradient
            dscores = probs
            dscores[range(self.x.shape[0]),self.y] -= 1
            dscores /= self.x.shape[0]

            # backpropagation in action:

            # first backprop into parameters W2 and b2
            dW2 = np.dot(h1.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            # next backprop into the hidden layer
            dhidden = np.dot(dscores, self.W2.T)
            # backprop the ReLU non-linearity
            dhidden[h1 <= 0] = 0
            # finally into W,b
            dW1 = np.dot(self.x.T, dhidden)
            db1 = np.sum(dhidden, axis=0, keepdims=True)

            # add regularization gradient contribution
            dW2 += self.regularization  * self.W2
            dW1 += self.regularization  * self.W1

            # perform a parameter update
            self.W1 += -self.step * dW1
            self.b1 += -self.step * db1
            self.W2 += -self.step * dW2
            self.b2 += -self.step * db2

        # output results of the last iteration
        f = lambda x: np.maximum(0, x)
        h1 = f(np.dot(self.x, self.W1) + self.b1)
        scores = np.dot(h1, self.W2) + self.b2  # NB that we don't use activation function on the output layer
        predicted_class = np.argmax(scores, axis=1)
        accuracy = np.mean(predicted_class == y)
        print 'training accuracy: %.2f' % (accuracy)

    def predict(self, x, y):
        # run an input through the score function given trained weights
        f = lambda x: np.maximum(0, x)
        h1 = f(np.dot(x, self.W1) + self.b1)
        scores = np.dot(h1, self.W2) + self.b2  # NB that we don't use activation function on the output layer
        predicted_class = np.argmax(scores, axis=1)
        accuracy = np.mean(predicted_class == y)
        print 'test accuracy: %.2f' % (accuracy)

# create a toy dataset: binary classification of 200 obs with 3 features
# let's try to make the cases separable, so 1/2 of Xs are drawn from a normal distribution N(0,1)
# and another 1/2 is from a distribution N(10,1). Those would be perfectly linearly separable
# so even a cat would be able to classify those. What about a NN, um?

x = np.append(np.random.randn(100, 2),(np.random.randn(100, 2) + 10), axis=0)
y = np.append(np.zeros([100],dtype='uint8'),np.ones([100],dtype='uint8'), axis=0)
plt.scatter(x[:,0],x[:,1], c=y, s=40)
# train the classifier:
nn = neural_network(x,y,1e-2,1e-3,100,3000)
nn.train()

# now predict new results based on unseen data (should produce all zeros)
x_test = np.random.randn(10, 2)
y_test = np.zeros([10],dtype='uint8')
# this baby returns 1 :S
nn.predict(x_test,y_test)

# visualization of loss over iterations:
plt.plot(nn.viz_loss)

# now visialize scatter of loss and prediction accuracy: as loss going down, we can see a drastic increase
# in accuracy, whoa!
plt.scatter(nn.viz_accuracy,nn.viz_loss)