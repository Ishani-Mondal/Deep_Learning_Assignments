import numpy as np
from data_loader import DataLoader

'''Calculate the linear rectified unit scores'''
def relu(x):
    return x * (x > 0)

'''Compute the softmax scores of a 2-D Matrix'''
def softmax(scores):
    exp_scores = np.exp(scores.T-np.max(scores,axis = 1)).T
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

f = open('accuracy.txt','w')
class NN(object):   
    # Initialize the weights and the biases of the hidden and output layer
    def __init__(self,input_dim,Hidden_units,output_dim):
        np.random.seed(1)
        self.W1 = 0.01 *np.random.randn(input_dim,Hidden_units) # (784,500)
        self.W2 = 0.01 *np.random.randn(Hidden_units,output_dim) # (500,10)
        self.b1 = 0.01 *np.zeros((1,Hidden_units)) # (1,500)
        self.b2 = 0.01 *np.zeros((1,output_dim)) # (1,10)

    # Forward scores calculation for the hidden and output unit
    def forward(self,X):
        hidden_layer = relu(np.dot(X, self.W1) + self.b1)
        scores = np.dot(hidden_layer, self.W2) + self.b2
        probs = softmax(scores)
        return hidden_layer, probs

    # Compute the average cross-entropy loss after performing regularization
    def compute_loss(self,probs,X,Y,reg_constant):
        num_examples = X.shape[0]
        corect_logprobs = -np.log(probs[range(num_examples),Y])
        data_loss = np.sum(corect_logprobs)/num_examples
        reg_loss = 0.5*reg_constant*np.sum(self.W1*self.W1) + 0.5*reg_constant*np.sum(self.W2*self.W2)
        loss = data_loss + reg_loss
        return loss

    # Compute the gradients of the parameters for the output and hidden layer
    def backward(self,hidden_layer, probs, X,Y,reg):
        # compute the gradient on scores
        num_examples = X.shape[0]
        dscores = probs
        dscores[range(num_examples),Y] -= 1
        dscores /= num_examples
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, self.W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW1 = np.dot(X.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)
        dW2 += reg * self.W2
        dW1 += reg * self.W1
        return dW1 , dW2, db1 , db2

    # Weight and bias updation after the gradient descent
    def update_parameters(self,dW1, dW2, db1, db2, step_size):
        self.W1 += -step_size * dW1
        self.b1 += -step_size * db1
        self.W2 += -step_size * dW2
        self.b2 += -step_size * db2

    # Compute the predicted accuracy on a dataset after doinh the forward propagation
    def predict(self,X,Y,mode):
        hidden_layer = relu(np.dot(X, self.W1) + self.b1)
        scores = np.dot(hidden_layer, self.W2) + self.b2
        predicted_class = np.argmax(scores, axis=1)
        print('Accuracy: %.2f for %s' % ((np.mean(predicted_class == Y)),mode))
        f.write('Accuracy: %.2f for %s \n' % ((np.mean(predicted_class == Y)),mode))