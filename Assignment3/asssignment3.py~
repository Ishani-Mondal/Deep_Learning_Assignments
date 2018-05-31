#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:15:39 2018

@author: user
"""

import tensorflow as tf
import data_loader
import sys
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression

data = data_loader.DataLoader()

x_train,y_train = data.load_data('train')
y_train = np.eye(10)[np.asarray(y_train)]

x_test, y_test = data.load_data('test')
y_test = np.eye(10)[np.asarray(y_test)]

# Train and Validation set 
train_data, val_data, train_labels, val_labels = train_test_split(x_train, y_train, test_size=0.1, random_state=123)


# Hyperparameter Definition
num_nodes= 100
batch_size = 100
last_improvement= 0
patience=20
best_validation_accuracy =0.0
learning_rate = 0.01
no_of_classes = 10
num_steps = 30
test= 0

# Relu and Softmax Activation Functions
def relu(x):
    boolx = x > 0
    actx = tf.cast(boolx, tf.float32)
    return x * actx

def softmax(logits, dims=-1):
  """Compute softmax over specified dimensions."""
  exp = tf.exp(logits - tf.reduce_max(logits, dims, keep_dims=True))
  return exp / tf.reduce_sum(exp, dims, keep_dims=True)

# Feed Forward Layer for the entire netwrork
def feedforward(X,weights_1,weights_2,weights_3,weights_4,biases_1,biases_2,biases_3,biases_4):
    logits_1 = tf.matmul(X, weights_1) + biases_1
    relu_layer_1 = relu(logits_1)
    logits_2 = tf.matmul(relu_layer_1, weights_2) + biases_2
    relu_layer_2 = relu(logits_2)
    logits_3 = tf.matmul(relu_layer_2, weights_3) + biases_3
    relu_layer_3 = relu(logits_3)
    logits_4 = tf.matmul(relu_layer_3, weights_4) + biases_4
    return logits_1, logits_2, logits_3, logits_4  

# Feed Forward Layer for the Layer 1
def ff_l1(X,w1,b1):
    logits_1 = tf.matmul(X, w1) + b1
    return logits_1

# Feed Forward Layer for the Layer 2
def ff_l2(X,w1,b1,w2,b2):
    logits_1 = tf.matmul(X, w1) + b1
    relu_layer_1 = relu(logits_1)
    logits_2 = tf.matmul(relu_layer_1, w2) + b2
    return logits_2

# Feed Forward Layer for the Layer 3
def ff_l3(X,w1,b1,w2,b2,w3,b3):
    logits_1 = tf.matmul(X, w1) + b1
    relu_layer_1 = relu(logits_1)
    logits_2 = tf.matmul(relu_layer_1, w2) + b2
    relu_layer_2 = relu(logits_2)
    logits_3 = tf.matmul(relu_layer_2, w3) + b3
    return logits_3 
def prediction(logits):
    pred = softmax(logits)
    return pred




# The compuation Graph Definition
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(100)
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 784))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 10))
    
    tf_valid_dataset = tf.constant(val_data,tf.float32)
    valid_labels = val_labels
    
    tf_test_dataset = tf.constant(x_test,tf.float32)
    test_labels = y_test

    # Variables.
    weights_1 = tf.Variable(tf.truncated_normal([784, num_nodes]),name='w1')
    biases_1 = tf.Variable(tf.zeros([num_nodes]),name='b1')
    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_nodes]),name='w2')
    biases_2 = tf.Variable(tf.zeros([num_nodes]),name='b2')
    weights_3 = tf.Variable(tf.truncated_normal([num_nodes, num_nodes]),name='w3')
    biases_3 = tf.Variable(tf.zeros([num_nodes]),name='b3')
    weights_4 = tf.Variable(tf.truncated_normal([num_nodes, no_of_classes]),name='w4')
    biases_4 = tf.Variable(tf.zeros([10]),name='b4')

    #Custom Layer wise weight and bias variables Layer 2
    weights_1_l1 = tf.Variable(tf.truncated_normal([784, no_of_classes]),name='w11')
    biases_1_l1 = tf.Variable(tf.zeros([no_of_classes]),name='b11')
    
    
    #Custom Layer wise weight and bias variables Layer 2
    weights_1_l2 = tf.Variable(tf.truncated_normal([784, num_nodes]),name='w12')
    biases_1_l2 = tf.Variable(tf.zeros([num_nodes]),name='b12')
    weights_2_l2 = tf.Variable(tf.truncated_normal([num_nodes, no_of_classes]),name='w22')
    biases_2_l2= tf.Variable(tf.zeros([no_of_classes]),name='b22')
    
    weights_1_l3 = tf.Variable(tf.truncated_normal([784, num_nodes]),name='w13')
    biases_1_l3 = tf.Variable(tf.zeros([num_nodes]),name='b13')
    weights_2_l3 = tf.Variable(tf.truncated_normal([num_nodes, num_nodes]),name='w23')
    biases_2_l3= tf.Variable(tf.zeros([num_nodes]),name='b23')
    weights_3_l3 = tf.Variable(tf.truncated_normal([num_nodes, no_of_classes]),name='w33')
    biases_3_l3= tf.Variable(tf.zeros([no_of_classes]),name='b33')
    

    
    # Training computation.
    logits_1_train,logits_2_train,logits_3_train,logits_4_train = feedforward(tf_train_dataset,weights_1,weights_2,weights_3,weights_4,biases_1,biases_2,biases_3,biases_4)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_4_train, labels=tf_train_labels))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
     # Predictions for the training
    #train_prediction = tf.nn.softmax(logits_4)
    train_prediction = prediction(logits_4_train)
    
    # Training computation for layer 1
    logits_1 = ff_l1(tf_train_dataset,weights_1_l1,biases_1_l1)
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_1, labels=tf_train_labels))
    optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(loss1)
    
    train_prediction_layer1 = prediction(logits_1)
    
    # Training computation for layer 2
    logits_2 = ff_l2(tf_train_dataset,weights_1_l2,biases_1_l2,weights_2_l2,biases_2_l2)
    tf.Print(tf.shape(logits_2),[tf.shape(logits_2)])
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=tf_train_labels))
    optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)
    
    train_prediction_layer2 = prediction(logits_2)
    
    # Training computation for layer 3
    logits_3 = ff_l3(tf_train_dataset,weights_1_l3,biases_1_l3,weights_2_l3,biases_2_l3,weights_3_l3,biases_3_l3)
    
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_3, labels=tf_train_labels))
    optimizer3 = tf.train.AdamOptimizer(learning_rate).minimize(loss3)
    
    train_prediction_layer3 = prediction(logits_3)
    
    # Predictions for validation 
    logits_1_train,logits_2_train,logits_3_train,logits_4_train = feedforward(tf_valid_dataset,weights_1,weights_2,weights_3,weights_4,biases_1,biases_2,biases_3,biases_4)
    
    valid_prediction = prediction(logits_4_train)
   



def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
    
with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    argument = str(sys.argv[1])
    parsed_value = argument[2:]
    print("Initialized")
    saver=tf.train.Saver()                                                                                                                                                                          
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_data[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
    
        if parsed_value == 'train':
            print('------------------------------Training the model---------------------------------')
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            print("Minibatch loss at step {}: {}".format(step, l))
            print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_labels)))
            acc = accuracy(valid_prediction.eval(), valid_labels)
            if acc > best_validation_accuracy:
                last_improvement = step
                best_validation_accuracy = acc
                w1,w2,w3,w4 = session.run([weights_1, weights_2, weights_3, weights_4], feed_dict=feed_dict)
                saver.save(session,'../weights/wts')
                b1,b2,b3,b4 = session.run([biases_1, biases_2, biases_3, biases_4], feed_dict=feed_dict)
                saver.save(session,'../weights/biases')
            if step - last_improvement > patience:
                print("Early stopping ...")
                break

        if parsed_value == 'test':
            print('--------------Restoring the model Weights and Biases------------------------')
            saver.restore(session,'../weights/wts')
            saver.restore(session,'../weights/biases')
            feed_dict = {weights_1:'w1:0',weights_2:'w2:0',weights_3:'w3:0',weights_4:'w4:0',biases_1:'b1:0',biases_2:'b2:0',biases_3:'b3:0',biases_4:'b4:0'}
            _,_,_,logits_4_test = feedforward(tf_test_dataset,weights_1,weights_2,weights_3,weights_4,biases_1,biases_2,biases_3,biases_4)
            test_prediction_layer4 = prediction(logits_4_test)
            if test==0:
                print("Test accuracy: {:.1f}".format(accuracy(test_prediction_layer4.eval(), test_labels)))
                break
            
        
        
        if parsed_value == 'layer1_train':
           print('------------------------------Training the model with layer 1---------------------------------')
           feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
           _, l, predictions = session.run([optimizer1, loss1, train_prediction_layer1], feed_dict=feed_dict)
           print("Minibatch loss at step {}: {}".format(step, l))
           print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
         
           if step == num_steps - 1 :
              w1 = session.run([weights_1_l1], feed_dict=feed_dict)
              saver.save(session,'../weights/wts_1')
              b1 = session.run([biases_1_l1], feed_dict=feed_dict)
              saver.save(session,'../weights/biases_1')
           

        if parsed_value == 'layer1_test':
           print('--------------Restoring the model Weights and Biases------------------------')
           saver.restore(session,'../weights/wts_1')
           saver.restore(session,'../weights/biases_1')
           feed_dict = {weights_1:'w11:0',biases_1:'b11:0'}
           logits_1_test = ff_l1(tf_test_dataset,weights_1,biases_1)
           test_prediction_layer1 = prediction(logits_1_test)
           if test==0:
               print("Test accuracy: {:.1f}".format(accuracy(test_prediction_layer1.eval(), test_labels)))
               break
                
            
        if parsed_value == 'layer2_train':
           print('------------------------------Training the model with layer 2---------------------------------')
           feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
           _, l, predictions = session.run([optimizer2, loss2, train_prediction_layer2], feed_dict=feed_dict)
           print("Minibatch loss at step {}: {}".format(step, l))
           print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
          
#           acc = accuracy(valid_prediction_layer2.eval(), valid_labels)
           if step == num_steps - 1 :
              w1,w2 = session.run([weights_1_l2,weights_2_l2], feed_dict=feed_dict)
              saver.save(session,'../weights/wts_2')
              b1,b2 = session.run([biases_1_l2,biases_2_l2], feed_dict=feed_dict)
              saver.save(session,'../weights/biases_2')
          
            
        if parsed_value == 'layer2_test':
           print('--------------Restoring the model Weights and Biases------------------------')
           saver.restore(session,'../weights/wts_2')
           saver.restore(session,'../weights/biases_2')
           feed_dict = {weights_1:'w12:0',biases_1:'b12:0',weights_2:'w22:0',biases_2:'b22:0'}
           logits_2_test = ff_l2(tf_test_dataset,weights_1,biases_1,weights_2,biases_2)
           test_prediction_layer2 = prediction(logits_2_test)
           if test==0:
               print("Test accuracy: {:.1f}".format(accuracy(test_prediction_layer2.eval(), test_labels)))
               break
            
        
        
        if parsed_value == 'layer3_train':
           print('------------------------------Training the model with layer 3---------------------------------')
           feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
           _, l, predictions = session.run([optimizer3, loss3, train_prediction_layer3], feed_dict=feed_dict)
           print("Minibatch loss at step {}: {}".format(step, l))
           print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
#           print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction_layer3.eval(), valid_labels)))
#           acc = accuracy(valid_prediction_layer3.eval(), valid_labels)
#           if acc > best_validation_accuracy:
#              last_improvement = step
#              best_validation_accuracy = acc
           if step == num_steps - 1 :
              w1,w2,w3 = session.run([weights_1_l3,weights_2_l3,weights_3_l3], feed_dict=feed_dict)
              saver.save(session,'../weights/wts_3')
              b1,b2,b3 = session.run([biases_1_l3,biases_2_l3,biases_3_l3], feed_dict=feed_dict)
              saver.save(session,'../weights/biases_3')
#           if step - last_improvement > patience:
#              print("Early stopping ...")
#              break
          
        if parsed_value == 'layer3_test':
           print('--------------Restoring the model Weights and Biases------------------------')
           saver.restore(session,'../weights/wts_3')
           saver.restore(session,'../weights/biases_3')
           feed_dict = {weights_1:'w13:0',biases_1:'b13:0',weights_2:'w23:0',biases_2:'b23:0',weights_3:'w33:0',biases_3:'b33:0'}
           logits_3_test = ff_l3(tf_test_dataset,weights_1,biases_1,weights_2,biases_2,weights_3,biases_3)
           test_prediction_layer3 = prediction(logits_3_test)
           if test==0:
               print("Test accuracy: {:.1f}".format(accuracy(test_prediction_layer3.eval(), test_labels)))
               break   

    session.close()       
