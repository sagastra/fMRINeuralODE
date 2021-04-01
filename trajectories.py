#!/usr/bin/env python
# coding: utf-8




import tensorflow as tf #import TensorFlow library version 1.14
import numpy as np

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import math
import cmath



import os
import numpy

from scipy.integrate import odeint



# Polynomial Regression for rsquared calculation
def polyfit(x, y, degree):
    results = {}

    coeffs = numpy.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = numpy.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                      
    ybar = numpy.sum(y)/len(y)         
    ssreg = numpy.sum((yhat-ybar)**2)   
    sstot = numpy.sum((y - ybar)**2)    
    results['determination'] = ssreg / sstot
    rsq = ssreg / sstot
    return rsq


#Next batch of data
def nextBatch(myData, num_batch):
    n_vox, n_scans, n_time = np.shape(myData)
    b=np.arange(n_scans)
    np.random.shuffle(b)
    return np.array([myData[:,i,:] for i in b[:num_batch]])

#Firing rate model with and without noise
def nFRM(x,t,W, n):
    indx = int(t/0.72)
    return -x+np.matmul(W,x)+ n[indx]

def FRM(x,y,W):
    return -x+np.matmul(W,x)

######## rest test data
n_subj = 447
n_train = 387
n_test = 60

filename = 'all_rest.txt'
data_rest = np.genfromtxt(filename, delimiter =',')
n_vox, n_time = np.shape(data_rest)
data_rest = np.reshape(data_rest, [n_vox, n_subj, int(n_time/n_subj)])
data_rest = data_rest[:,:n_test,:]


#### load length and weight matirix
from numpy import linalg as LA
f_weights = "weights_tract.txt"

#normalize
weights_in = np.loadtxt(f_weights)
w,v = LA.eig(weights_in)
c1 = w[0]
k1 = 0.6/c1
weights = k1*weights_in





t_time = 1800
n_time = 50
n_scans = 447
n_vox = 66
n_hidden = 150 
n_loop = 600
number_of_layers = 4 


n_batches = 60 
tr = 0.72 #as defined from HCP


#load trained network
tf.reset_default_graph()

imported_graph = tf.train.import_meta_graph('./trained_network')
print('loaded meta')
with tf.Session() as sess:
    # restore the saved vairable
    real = []
    inferred = []
    rsquared = []
    auto = []
    one_err = []
    oerr = []
    imported_graph.restore(sess, './trained_network')
    print('loaded_graph')
    # print the loaded variable
    train_op = tf.get_collection("train_op")[0]
    
    
    for step in range(1):
        
        data_batch = nextBatch(data_rest, n_batches)
        _current_state = np.zeros((number_of_layers, 2, n_batches, n_hidden))

        t_time = 2400
        for i in range( int(t_time/n_time) -1):
            batch = data_batch[:,:,i*n_time: (i+1)*n_time]
            init_pt_train =  data_batch[:,:,0]
            session_loss, _current_state , out, initial_cond =sess.run(
                ['loss:0', 'current_state:0', 'one_step_out:0','FR:0'], #graph variable we want to compute
                feed_dict={'data_input:0': batch, 'init_state:0': _current_state, 'TR:0':[0.72]})

            
            
           
            ts = np.linspace(0.0, tr*5, 51)
            pred3 = []           

            
              
	    #trajectory from each initial cond across all batches and timepoints
            for a_fr in initial_cond:
                 
                #noise added into integration
                noise = np.random.normal(0, 0.45, 51*66)
                noise = np.reshape(noise, [51,66])
                bnm_trj = np.array([ odeint(nFRM, np.squeeze(a_fr[i,:]), ts, args=(weights,noise))  for i in range(n_batches)])
           

                multi_step_pred = [bnm_trj[:,(x+1)*10,:] for x in range(5)]
                pred3.append(multi_step_pred)
            
            pred3 = np.array(pred3)
   
            #calculate error at every timepoint 1-5 across batch on everytimestep           
            err = np.zeros((45,5))         
            for xx in range(45):
                for yy in range(5):
                    err[xx,yy] = polyfit(np.array(batch[:,:,xx+yy+1]).flatten(), np.array(pred3[xx,yy,:,:]).flatten(),1)

 
            
            
            
            if (i != 0 and i!=24):
                if(oerr != [] ):
                    oerr = np.vstack((oerr, err))
                else:
                    oerr = err


            





np.savetxt('rsq_loss.txt',oerr)
