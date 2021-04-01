#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf #tf version 1.14.0 gpu enabled

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import math
import cmath


import os





###### all rest data hcp
n_subj = 447
n_train = 387
n_test = 60


filename = 'all_rest.txt'
data_rest = np.genfromtxt(filename, delimiter =',')
n_vox, n_time = np.shape(data_rest)
data_rest = np.reshape(data_rest, [n_vox, n_subj, int(n_time/n_subj)])
data_rest = data_rest[:,n_test:,:]

# task data hcp concatenated and cut into nearest 50
filename = 'all_task.txt'
data_task = np.genfromtxt(filename, delimiter =' ')
data_task = np.reshape(data_task, [66, 447, 1800]) 


#network parameters

t_time = 1800
n_time = 50
n_infered = 10 
n_scans = 447
n_vox = 66
n_hidden = 150
n_loop = 600
number_of_layers = 4 


n_batches = 60 
tr = 0.72 #as defined from HCP data


#functions to graph batches of data

def nextBatch(myData, num_batch):
    n_vox, n_scans, n_time = np.shape(myData)
    b=np.arange(n_scans)
    np.random.shuffle(b)
    return np.array([myData[:,i,:] for i in b[:num_batch]])

def detBatch(myData, num_batch):
    n_vox, n_scans, n_time = np.shape(myData)
    b=np.arange(n_scans)
    return np.array([myData[:,i,:] for i in b[:num_batch]])




#### load length and weight matirix from DTI tracktrography
from numpy import linalg as LA
f_weights = "weights_tract.txt"


f_lengths = "lengths_tract.txt"
lengths_in = np.loadtxt(f_lengths)
lengths= np.reshape([lengths_in.flatten()[i] if lengths_in.flatten()[i] else 250 for i in range(66*66)], (np.shape(lengths_in)))


weights_in = np.loadtxt(f_weights)


#normalize
w,v = LA.eig(weights_in)
c1 = w[0]
k1 = 0.6/c1

weights = k1*weights_in


##### TENSORFLOW GRAPH


tf.reset_default_graph()

###placeholders for data batch
data_input = tf.placeholder(tf.float32, [None,  n_vox, n_time], name="data_input")
data_series = tf.transpose(data_input, [0, 2, 1])
sequence = tf.split(data_input, n_time , axis=2)
data_sequence =  [tf.squeeze(d) for d in sequence]

##placeholders for initial state of LSTM

init_state = tf.placeholder(tf.float32, [number_of_layers, 2, None, n_hidden],  name="init_state")
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(number_of_layers)]
)





#constants
TR = tf.placeholder(tf.float32, [1], name="TR")
tf_weights = tf.constant(weights, dtype=tf.float32)
tf_lengths = tf.constant(lengths , dtype= tf.float32)


#intermediate variables
state_series = []
output_series = []
pred_bold = []

#sampling 
def sampler(data_vector):
    my_mean, my_logstd  = tf.split(data_vector, num_or_size_splits=2, axis=1)    
    my_out = tf.random_normal(tf.shape(my_mean),mean=my_mean, stddev=tf.exp(my_logstd))
    return my_out

######## LSTM initialization and running 

def getCell(n):
    cell = tf.contrib.rnn.LSTMCell(n, state_is_tuple=True, reuse=tf.AUTO_REUSE)
    #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
    return cell

#### LSTM initialization
stacked_lstm = tf.contrib.rnn.MultiRNNCell([getCell(n_hidden) for _ in range(number_of_layers)], state_is_tuple=True)

#Variables for transformation
output_series = []
W = tf.Variable(np.random.rand(n_hidden, n_vox*2), dtype=tf.float32)
b = tf.Variable(np.zeros((n_vox*2)), dtype=tf.float32)



#### LSTM execution
output_series_dynamic, current_state_tmp = tf.nn.dynamic_rnn(stacked_lstm, data_series, initial_state=rnn_tuple_state)
output_series = tf.split(output_series_dynamic, n_time, axis = 1)



current_state = tf.identity(current_state_tmp, name="current_state")

firing_rate = [sampler(tf.matmul(tf.squeeze(o), W) + b) for o in output_series]
FR = tf.identity(firing_rate, name='FR')

###Brain Network Model ode
ts = tf.linspace(0.0, TR[0], 10)

for singleFiring in firing_rate:
    all_firing = tf.contrib.integrate.odeint(lambda _fr_, ts: tf.subtract(tf.matmul(_fr_, tf_weights), _fr_), singleFiring, ts, rtol = 0.01)
    nextFiring = all_firing[-1, :,:]
    pred_bold.append(nextFiring)

one_step_out = tf.identity(pred_bold, name="one_step_out")


#############################LOSS FUNCTION#########################################################



losses_recon = [tf.losses.mean_squared_error(pred, labels) for pred, labels in zip(pred_bold[0:n_time-1], data_sequence[1:n_time])]  

my_loss = tf.reduce_mean(losses_recon)  
loss = tf.identity(my_loss, name='loss')

#############################LOSS FUNCTION#########################################################


learning_rate = 0.0001

optimizer=tf.train.AdamOptimizer(learning_rate) #gradient descent optimizer with steps scaled by 0.1
train_op= optimizer.minimize(loss) #optimization function




saver = tf.train.Saver()


### Running it
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
session = tf.InteractiveSession(config=config)
session.run(tf.global_variables_initializer())



loss_ct = []
#training loop
for step in range(n_loop):
    data_batch = nextBatch(data_rest, n_batches)

    _current_state = np.zeros((number_of_layers, 2, n_batches, n_hidden))
    t_time = 2400
    ###rest train
    for i in range( int(t_time/n_time) -1):
        batch = data_batch[:,:,i*n_time: (i+1)*n_time]
        init_pt_train =  data_batch[:,:,0]
        session_loss, _train_op , _current_state, WW =session.run(
            [loss, train_op, current_state, tf_weights], #graph variable we want to compute
            feed_dict={data_input: batch, init_state: _current_state, TR:[tr], init_pt: init_pt_train})
    

        if (i == 13 and step % 10 == 0): 
            print('loss', session_loss)
            print(step)
    
    t_time = 1800

    data_batch = nextBatch(data_task, n_batches)     
    ###task train
    for i in range( int(t_time/n_time) -1):
        batch = data_batch[:,:,i*n_time: (i+1)*n_time]
        init_pt_train =  data_batch[:,:,0]
        session_loss, _train_op , _current_state, WW=session.run(
            [loss, train_op, current_state, tf_weights], #graph variable we want to compute
            feed_dict={data_input: batch, init_state: _current_state, TR:[tr], init_pt: init_pt_train})
    

        if (i == 13 and step % 10 == 0):
            saver.save(session, './trained_network')   #### Saving Network
            print('loss', session_loss)
            print(step)
            if(loss_ct != [] ):
                loss_ct = np.vstack((loss_ct, session_loss))
            else:
                loss_ct = session_loss


np.savetxt('loss.txt', np.array(loss_ct))





