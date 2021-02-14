# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:46:59 2021

@author: Ibrahima S. Sow
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.model_selection import train_test_split

import time

nb_features = 1 #You can change as you will



#Generate the data

X, y = datasets.make_regression(n_samples=500, n_features=nb_features, noise=30)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)

#need that theta0,x0=1 (simplification)

train_o = np.ones([X_train.shape[0], 1], dtype=X_train.dtype)
test_o = np.ones([X_test.shape[0],1], dtype=X_test.dtype)
X_train = np.concatenate((train_o, X_train), axis=1)
X_test = np.concatenate((test_o, X_test), axis=1)


""" uncomment to show the data

plt.scatter(X_test[:,1], y_test, c='b')
plt.scatter(X_train[:,1], y_train, c='r')
"""

plt.show()



#hypothesis

def hypothesis(theta, x):
    h = np.matmul(theta.T,x)
    return h



#for a single training example 
def batch_update(theta, gradient, learning_rate):
    theta = theta - learning_rate * gradient
    return theta





def batch_gradient_descent(X, Y, learning_rate, precision):
    
    
    start = time.time()

    step_size = 10
    iterations = 1
    
    theta = np.zeros([nb_features+1, 1], dtype=np.float64)
    
    plt.ion() #interactive
    
    while step_size > precision:
        
        prev_th = theta
        gradient = np.zeros([nb_features+1,1], dtype=np.float64)
        for x,y in zip(X,Y):
            h_x = hypothesis(theta,x)
            x = np.reshape(x, theta.shape)
            gradient += (h_x - y)*x
        
        theta = batch_update(theta, gradient, lr)
        step_size = np.sum(abs(theta - prev_th))
        print("Iteration :", iterations, "- Theta :", theta)
        iterations+=1
    
    end = time.time()
    print("\n Minimum found at theta =", theta, "in ", end-start, "sec with", iterations, "iterations")

    return theta


def stochastic_gradient_descent():
    pass



def cost_function(): #To get a quantification of the loss
    pass






#test 
lr = 0.001
p = 0.001

THETA = batch_gradient_descent(X_train, y_train, lr, p)

f_y = THETA[:, 0] * X_test + THETA[0, 0]



plt.scatter(X_test[:, 1], y_test, c='red', marker='o', edgecolors='black')
plt.plot(X_test, f_y, c='orange')
plt.show()
    











