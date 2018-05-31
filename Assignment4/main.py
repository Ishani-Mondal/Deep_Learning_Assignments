import tensorflow as tf
import numpy as np
from data_loader import DataLoader
import itertools
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer
import argparse
import matplotlib.pyplot as plt



# Create placeholders
def create_placeholders(n_inputs, n_outputs, n_steps):
	X = tf.placeholder(tf.float32,	[None,	n_steps, n_inputs])
	y = tf.placeholder(tf.int32,	[None, n_outputs])    
	return X, y


def compute_loss(logits, labels):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
	return loss


class LSTMCell_test(rnn.RNNCell):
	def __init__(self, n_neurons, input_size, parameters):
		self._n_neurons = n_neurons
		self._n_inputs  = input_size
		self._parameters = parameters
	
	@property
	def input_size(self):
		return self._n_inputs

	@property
	def output_size(self):
		return self._n_neurons

	@property
	def state_size(self):
		return 2 * self._n_neurons


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_C, prev_h = tf.split(state, 2, 1)
			IG_Wx = self._parameters['IG_Wx']
			IG_Uh = self._parameters['IG_Uh']
			IG_b = self._parameters['IG_b']

			FG_Wx = self._parameters['FG_Wx']
			FG_Uh = self._parameters['FG_Uh']
			FG_b = self._parameters['FG_b']

			tanh_Wx = self._parameters['tanh_Wx']
			tanh_Uh = self._parameters['tanh_Uh']
			tanh_b = self._parameters['tanh_b']

			OG_Wx = self._parameters['OG_Wx']
			OG_Uh = self._parameters['OG_Uh']
			OG_b = self._parameters['OG_b']
	
			input_gate  = tf.sigmoid(tf.add( tf.add( tf.matmul(x, IG_Wx), tf.matmul(prev_h, IG_Uh) ) , IG_b))
			output_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, OG_Wx), tf.matmul(prev_h, OG_Uh) ) , OG_b))
			forget_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, FG_Wx), tf.matmul(prev_h, FG_Uh) ) , FG_b))
			tanh_output = tf.tanh(tf.add( tf.add( tf.matmul(x, tanh_Wx), tf.matmul(prev_h, tanh_Uh) ) , tanh_b))
			C_t = tf.add(tf.multiply(forget_gate, prev_C), tf.multiply(input_gate, tanh_output))
			y_t = tf.multiply(output_gate, tf.tanh(C_t))
			return y_t, tf.concat([C_t, y_t], 1)


class LSTMCell_train(rnn.RNNCell):
	def __init__(self, n_neurons, input_size):
		self._n_neurons = n_neurons
		self._n_inputs  = input_size


	@property
	def input_size(self):
		return self._n_inputs

	@property
	def output_size(self):
		return self._n_neurons

	@property
	def state_size(self):
		return 2 * self._n_neurons


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_C, prev_h = tf.split(state, 2, 1)
			IG_Wx = tf.get_variable("IG_Wx", [self._n_inputs, self._n_neurons], initializer=xavier_initializer(seed=42))
			IG_Uh = tf.get_variable("IG_Uh", [self._n_neurons, self._n_neurons], initializer=xavier_initializer(seed=42))
			IG_b  = tf.get_variable("IG_b", [1, self._n_neurons], initializer=tf.ones_initializer())

			FG_Wx = tf.get_variable("FG_Wx", [self._n_inputs, self._n_neurons], initializer=xavier_initializer(seed=42))
			FG_Uh = tf.get_variable("FG_Uh", [self._n_neurons, self._n_neurons], initializer=xavier_initializer(seed=42))
			FG_b  = tf.get_variable("FG_b", [1, self._n_neurons], initializer=tf.ones_initializer())

			tanh_Wx = tf.get_variable("tanh_Wx", [self._n_inputs, self._n_neurons], initializer=xavier_initializer(seed=42))
			tanh_Uh = tf.get_variable("tanh_Uh", [self._n_neurons, self._n_neurons], initializer=xavier_initializer(seed=42))
			tanh_b  = tf.get_variable("tanh_b", [1, self._n_neurons], initializer=tf.ones_initializer())

			OG_Wx = tf.get_variable("OG_Wx", [self._n_inputs, self._n_neurons], initializer=xavier_initializer(seed=42))
			OG_Uh = tf.get_variable("OG_Uh", [self._n_neurons, self._n_neurons], initializer=xavier_initializer(seed=42))
			OG_b  = tf.get_variable("OG_b", [1, self._n_neurons], initializer=tf.ones_initializer())

			input_gate  = tf.sigmoid(tf.add( tf.add( tf.matmul(x, IG_Wx), tf.matmul(prev_h, IG_Uh) ) , IG_b))
			forget_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, FG_Wx), tf.matmul(prev_h, FG_Uh) ) , FG_b))
			output_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, OG_Wx), tf.matmul(prev_h, OG_Uh) ) , OG_b))
			tanh_output = tf.tanh(tf.add( tf.add( tf.matmul(x, tanh_Wx), tf.matmul(prev_h, tanh_Uh) ) , tanh_b))
			C_t = tf.add(tf.multiply(forget_gate, prev_C), tf.multiply(input_gate, tanh_output))
			y_t = tf.multiply(output_gate, tf.tanh(C_t))
			return y_t, tf.concat([C_t, y_t], 1)


class GRUCell_test(rnn.RNNCell):
	def __init__(self, n_neurons, input_size, parameters):
		self._n_neurons = n_neurons
		self._n_inputs  = input_size
		self._parameters = parameters
	
	@property
	def input_size(self):
		return self._n_inputs

	@property
	def output_size(self):
		return self._n_neurons

	@property
	def state_size(self):
		return self._n_neurons


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_h = state
			UG_Wx = self._parameters['UG_Wx']
			UG_Uh = self._parameters['UG_Uh']

			RG_Wx = self._parameters['RG_Wx']
			RG_Uh = self._parameters['RG_Uh']

			tanh_Wx = self._parameters['tanh_Wx']
			tanh_Uh = self._parameters['tanh_Uh']
			tanh_b = self._parameters['tanh_b']
	
			update_gate  = tf.sigmoid(tf.add( tf.matmul(x, UG_Wx), tf.matmul(prev_h, UG_Uh) ))
			reset_gate = tf.sigmoid(tf.add( tf.matmul(x, RG_Wx), tf.matmul(prev_h, RG_Uh) ) )
			tanh_output = tf.tanh( tf.add (tf.add( tf.matmul(x, tanh_Wx), tf.matmul(tf.multiply(prev_h, reset_gate), tanh_Uh) ), tanh_b))
			y_t =  tf.add ( tf.multiply((1.0-update_gate),prev_h) , tf.multiply(update_gate, tanh_output) )			
			return y_t, y_t


class GRUCell_train(rnn.RNNCell):
	def __init__(self, n_neurons, input_size):
		self._n_neurons = n_neurons
		self._n_inputs  = input_size


	@property
	def input_size(self):
		return self._n_inputs

	@property
	def output_size(self):
		return self._n_neurons

	@property
	def state_size(self):
		return self._n_neurons


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_h = state
			UG_Wx = tf.get_variable("UG_Wx", [self._n_inputs, self._n_neurons], initializer=xavier_initializer(seed=42))
			UG_Uh = tf.get_variable("UG_Uh", [self._n_neurons, self._n_neurons], initializer=xavier_initializer(seed=42))

			RG_Wx = tf.get_variable("RG_Wx", [self._n_inputs, self._n_neurons], initializer=xavier_initializer(seed=42))
			RG_Uh = tf.get_variable("RG_Uh", [self._n_neurons, self._n_neurons], initializer=xavier_initializer(seed=42))

			tanh_Wx = tf.get_variable("tanh_Wx", [self._n_inputs, self._n_neurons], initializer=xavier_initializer(seed=42))
			tanh_Uh = tf.get_variable("tanh_Uh", [self._n_neurons, self._n_neurons], initializer=xavier_initializer(seed=42))
			tanh_b  = tf.get_variable("tanh_b", [1, self._n_neurons], initializer=tf.ones_initializer())

			update_gate  = tf.sigmoid(tf.add( tf.matmul(x, UG_Wx), tf.matmul(prev_h, UG_Uh) ))
			reset_gate = tf.sigmoid(tf.add( tf.matmul(x, RG_Wx), tf.matmul(prev_h, RG_Uh) ) )
			tanh_output = tf.tanh( tf.add (tf.add( tf.matmul(x, tanh_Wx), tf.matmul(tf.multiply(prev_h, reset_gate), tanh_Uh) ), tanh_b))
			y_t =  tf.add ( tf.multiply((1.0-update_gate),prev_h) , tf.multiply(update_gate, tanh_output) )			
			return y_t, y_t


def train_model_lstm(n_neurons, n_steps=28, n_inputs = 28,n_outputs=10, learning_rate=0.0001,  n_epochs=100, batch_size=32):
	X, y = create_placeholders(n_inputs, n_outputs, n_steps)
	# lstm-cell
	current_input = tf.unstack(X , n_steps, 1)
	lstm_cell_train = LSTMCell_train(n_neurons, n_inputs)
	outputs, states	= tf.nn.static_rnn(lstm_cell_train, current_input, dtype=tf.float32)  
	# fully-connected layer
	FC_W = tf.get_variable("FC_W", [n_neurons, n_outputs], initializer = tf.contrib.layers.xavier_initializer(seed=42))
	FC_b = tf.get_variable("FC_b", [n_outputs], initializer = tf.zeros_initializer())
	Z = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)
	#optimizatio
	loss = compute_loss(Z, y)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	training_op = optimizer.minimize(loss)
	correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()
	# data-loading
	ld=DataLoader()
	X_train, Y_train =ld.load_data()
	Y_train = np.eye(10)[np.asarray(Y_train, dtype=np.int32)]
	X_Batched, Y_Batched = ld.create_batches(X_train, Y_train, batch_size)	
	X_test,y_test=ld.load_data(mode='test')
	y_test = np.eye(10)[np.asarray(y_test, dtype=np.int32)]
	X_test = X_test.reshape((-1, n_steps, n_inputs))
	saver = tf.train.Saver()
	weight_filepath = "./weights/lstm/hidden_unit" + str(n_neurons)+ "/model.ckpt"
	with tf.Session() as sess:
		init.run()
		#training
		for epoch in range(n_epochs):
			for X_batch, y_batch in itertools.izip(X_Batched, Y_Batched):
				X_batch	= X_batch.reshape((-1,	n_steps, n_inputs))
				sess.run(training_op, feed_dict={X:	X_batch, y: y_batch})	
			acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
			print("Train accuracy after %s epochs: %s" %( str(epoch+1), str(acc_train*100) ))
		acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})				
		print("Test accuracy:  ", acc_test*100)
		# Save parameters in memory		
		saver.save(sess, weight_filepath)
	return acc_test


def train_model_gru(n_neurons, n_steps=28, n_inputs = 28,n_outputs=10, learning_rate=0.0001,  n_epochs=100, batch_size=32):
	X, y = create_placeholders(n_inputs, n_outputs, n_steps)
	# lstm-cell
	current_input = tf.unstack(X , n_steps, 1)
	gru_cell_train = GRUCell_train(n_neurons, n_inputs)
	outputs, states	= tf.nn.static_rnn(gru_cell_train, current_input, dtype=tf.float32)  
	# fully-connected layer
	FC_W = tf.get_variable("FC_W", [n_neurons, n_outputs], initializer = tf.contrib.layers.xavier_initializer(seed=42))
	FC_b = tf.get_variable("FC_b", [n_outputs], initializer = tf.zeros_initializer())
	Z = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)
	#optimizatio
	loss = compute_loss(Z, y)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	training_op = optimizer.minimize(loss)
	correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()
	# data-loading
	ld=DataLoader()
	X_train, Y_train =ld.load_data()
	Y_train = np.eye(10)[np.asarray(Y_train, dtype=np.int32)]
	X_Batched, Y_Batched = ld.create_batches(X_train, Y_train, batch_size)	
	X_test,y_test=ld.load_data(mode='test')
	y_test = np.eye(10)[np.asarray(y_test, dtype=np.int32)]
	X_test = X_test.reshape((-1, n_steps, n_inputs))
	saver = tf.train.Saver()
	weight_filepath = "./weights/gru/hidden_unit" + str(n_neurons)+ "/model.ckpt"
	with tf.Session() as sess:
		init.run()
		#training
		for epoch in range(n_epochs):
			for X_batch, y_batch in itertools.izip(X_Batched, Y_Batched):
				X_batch	= X_batch.reshape((-1,	n_steps, n_inputs))
				sess.run(training_op, feed_dict={X:	X_batch, y: y_batch})	
			acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
			print("Train accuracy after %s epochs: %s" %( str(epoch+1), str(acc_train*100) ))
		acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})				
		print("Test accuracy:  ", acc_test*100)
		# Save parameters in memory		
		saver.save(sess, weight_filepath)
	return acc_test


def test_model_gru(weight_filepath, n_neurons, n_steps=28, n_inputs = 28,n_outputs=10):
	with tf.Session() as sess:
		X, y = create_placeholders(n_inputs, n_outputs, n_steps)
		new_saver = tf.train.import_meta_graph(weight_filepath + "/model.ckpt.meta")
		new_saver.restore(sess, tf.train.latest_checkpoint(weight_filepath))
		FC_W    = sess.run('FC_W:0')
		FC_b    = sess.run('FC_b:0') 
		UG_Wx   = sess.run('rnn/GRUCell_train/UG_Wx:0')
		UG_Uh   = sess.run('rnn/GRUCell_train/UG_Uh:0')
		RG_Wx   = sess.run('rnn/GRUCell_train/RG_Wx:0')
		RG_Uh   = sess.run('rnn/GRUCell_train/RG_Uh:0')
		tanh_Wx = sess.run('rnn/GRUCell_train/tanh_Wx:0')
		tanh_Uh = sess.run('rnn/GRUCell_train/tanh_Uh:0')
		tanh_b  = sess.run('rnn/GRUCell_train/tanh_b:0')
		parameters = {
		"UG_Wx" : UG_Wx,
		"UG_Uh" : UG_Uh,
		"RG_Wx" : RG_Wx,
		"RG_Uh" : RG_Uh,
		"tanh_Wx" : tanh_Wx,
		"tanh_Uh" : tanh_Uh,
		"tanh_b" : tanh_b
		}
	
		# lstm-cell
		current_input = tf.unstack(X , n_steps, 1)
		gru_cell_test = GRUCell_test(n_neurons, n_inputs, parameters)
		outputs, states	= tf.nn.static_rnn(gru_cell_test, current_input, dtype=tf.float32)  
		# fully-connected layer
		Z = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)
		correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		init = tf.global_variables_initializer()
		# data-loading
		ld=DataLoader()
		X_test,y_test=ld.load_data(mode='test')
		y_test = np.eye(10)[np.asarray(y_test, dtype=np.int32)]
		X_test = X_test.reshape((-1, n_steps, n_inputs))
		acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})				
		print("Test accuracy:  ", acc_test*100)
		return acc_test


def test_model_lstm(weight_filepath, n_neurons, n_steps=28, n_inputs = 28,n_outputs=10):
	with tf.Session() as sess:
		X, y = create_placeholders(n_inputs, n_outputs, n_steps)
		new_saver = tf.train.import_meta_graph(weight_filepath + "/model.ckpt.meta")
		new_saver.restore(sess, tf.train.latest_checkpoint(weight_filepath))
		FC_W    = sess.run('FC_W:0')
		FC_b    = sess.run('FC_b:0') 
		IG_Wx   = sess.run('rnn/LSTMCell_train/IG_Wx:0')
		IG_Uh   = sess.run('rnn/LSTMCell_train/IG_Uh:0')
		IG_b    = sess.run('rnn/LSTMCell_train/IG_b:0')
		FG_Wx   = sess.run('rnn/LSTMCell_train/FG_Wx:0')
		FG_Uh   = sess.run('rnn/LSTMCell_train/FG_Uh:0')
		FG_b    = sess.run('rnn/LSTMCell_train/FG_b:0')
		OG_Wx   = sess.run('rnn/LSTMCell_train/OG_Wx:0')
		OG_Uh   = sess.run('rnn/LSTMCell_train/OG_Uh:0')
		OG_b    = sess.run('rnn/LSTMCell_train/OG_b:0')
		tanh_Wx = sess.run('rnn/LSTMCell_train/tanh_Wx:0')
		tanh_Uh = sess.run('rnn/LSTMCell_train/tanh_Uh:0')
		tanh_b  = sess.run('rnn/LSTMCell_train/tanh_b:0') 
		parameters = {
		"IG_Wx" : IG_Wx,
		"IG_Uh" : IG_Uh,
		"IG_b"  : IG_b,
		"FG_Wx" : FG_Wx,
		"FG_Uh" : FG_Uh,
		"FG_b"  : FG_b,
		"tanh_Wx" : tanh_Wx,
		"tanh_Uh" : tanh_Uh,
		"tanh_b" : tanh_b,
		"OG_Wx" : OG_Wx,
		"OG_Uh" : OG_Uh,
		"OG_b"  : OG_b
		}
	
		# lstm-cell
		current_input = tf.unstack(X , n_steps, 1)
		lstm_cell_test = LSTMCell_test(n_neurons, n_inputs, parameters)
		outputs, states	= tf.nn.static_rnn(lstm_cell_test, current_input, dtype=tf.float32)  
		# fully-connected layer
		Z = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)
		correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		init = tf.global_variables_initializer()
		# data-loading
		ld=DataLoader()
		X_test,y_test=ld.load_data(mode='test')
		y_test = np.eye(10)[np.asarray(y_test, dtype=np.int32)]
		X_test = X_test.reshape((-1, n_steps, n_inputs))
		acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})				
		print("Test accuracy:  ", acc_test*100)
		return acc_test

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--test' , action="store_true",help='To test the data')
parser.add_argument('--train', action="store_true",help='To train the data')
parser.add_argument('--hidden_unit', action="store",help='Use model for entered hidden unit size',type=int)
parser.add_argument('--model', action="store",help='Select model',type=str)
parser.add_argument('--plot', action="store_true",help='to generate plot ')
args = parser.parse_args()


n_steps = 28
n_inputs = 28
n_outputs = 10
learning_rate =	0.001
n_epochs = 10
batch_size = 32


if args.model == 'lstm':    
	if args.train:
		n_neurons = int(args.hidden_unit)
		test_acc = train_model_lstm(n_neurons, n_steps, n_inputs ,n_outputs, learning_rate,  n_epochs, batch_size)
   
	elif args.test:
		n_neurons = int(args.hidden_unit)
		weight_filepath = "./weights/lstm/hidden_unit" + str(n_neurons)
		test_acc = test_model_lstm(weight_filepath, n_neurons, n_steps=28, n_inputs = 28,n_outputs=10)

elif args.model == 'gru':    
	if args.train:
		n_neurons = int(args.hidden_unit)
		test_acc = train_model_gru(n_neurons, n_steps, n_inputs ,n_outputs, learning_rate,  n_epochs, batch_size)
   
	elif args.test:
		n_neurons = int(args.hidden_unit)
		weight_filepath = "./weights/gru/hidden_unit" + str(n_neurons)
		test_acc = test_model_gru(weight_filepath, n_neurons, n_steps=28, n_inputs = 28,n_outputs=10)


elif args.plot:
	values = []
	for i in [32]:
		x = int(i)
		n_neurons = x
		weight_filepath = "./weights/gru/hidden_unit" + str(n_neurons)
		test_acc = test_model_gru(weight_filepath, n_neurons, n_steps=28, n_inputs = 28,n_outputs=10)
                values.append((x,test_acc))		
	plt.plot(*zip(*values), label='GRU'.format(1))
	plt.xlabel('hidden_unit_size')
	plt.ylabel('test accuracy')
        values = []
	for i in [32]:
		x = int(i)
		n_neurons = x
		weight_filepath = "./weights/lstm/hidden_unit" + str(n_neurons)
		test_acc = test_model_lstm(weight_filepath, n_neurons, n_steps=28, n_inputs = 28,n_outputs=10)
                values.append((x,test_acc))		
	plt.plot(*zip(*values), label='lSTM'.format(2))
	plt.xlabel('hidden_unit_size')
	plt.ylabel('test accuracy')
	plt.title('accuracy_compare')
	plt.savefig("plot")
       
