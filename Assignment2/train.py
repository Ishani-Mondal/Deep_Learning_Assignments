import numpy as np
from data_loader import DataLoader

from module import NN
#f = open('accuracy.txt')

# flags
use_validation_set = False
# The necessary hyperparameters
step_size = 0.001 # Learning Rate
reg = 1e-2 # regularization strength
batch_size = 512 #Number of batches in which the entire dataset will be divided into
n_epochs = 200
Hidden_unit_dimension = 300 # Number of units in the hidden layer
output_dimension = 10 # Number of classes of the image



# Load the training and test Data
dl = DataLoader()
X,Y = dl.load_data('train')
X_test, Y_test = dl.load_data('test')


'''Implement mini-batch SGD here'''
# Create an object of the Neural Network Model
nn = NN(X.shape[1],Hidden_unit_dimension,output_dimension)

for current_epoch in range(n_epochs):
	for current_batch in range(0, X.shape[0], batch_size):

	    X_mini, y_mini = dl.create_batches(X,Y,batch_size, shuffle=True)
	    num_examples = X_mini.shape[0]
	    
	    hidden_layer, probs = nn.forward(X_mini) # Compute the hidden layer and softmax layer scores
	    #print(probs.shape)

	    loss = nn.compute_loss(probs,X_mini,y_mini,reg) # Compute the loss

	    if current_batch % (X.shape[0] - 1) == 0:
	    	print("Epoch %d: loss %f" % (current_epoch+1, loss))
	    dW1, dW2, db1, db2 = nn.backward(hidden_layer, probs, X_mini,y_mini,reg)
	    nn.update_parameters(dW1, dW2, db1, db2, step_size)
	

		
'''Test the Training and Tesy accuracy'''
nn.predict(X,Y,mode='train')
nn.predict(X_test,Y_test, mode='test')
