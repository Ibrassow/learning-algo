# -*- coding: utf-8 -*-
"""
@author: Ibrahima S. Sow

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.model_selection import train_test_split

import time

from sklearn.metrics import mean_squared_error

#hypothesis

def hypothesis(theta, x):
    #h = np.matmul(theta.T,x)
    h = x.dot(theta)
    return h


def loss(theta, x, y):

    prediction = np.dot(x,theta)

    #cost function
    #error = np.mean(np.square(y-prediction)) 
    
    if len(x) == 1:
        x = np.array([x])
        y = np.array([y])
    
    error = mean_squared_error(y, prediction)
    return error


def exp_decay(it, init_lr):
   k = 0.01
   lrate = init_lr * np.exp(-k*it)
   return lrate



def batch_gradient_descent(X, Y, learning_rate, accuracy):
    
    
    start = time.time()

    step_size = 10
    iterations = 1
    
    theta = np.zeros([nb_features+1, 1], dtype=np.float64)
    
    cost = 0
    errors = np.array([])
    
    while step_size > accuracy:
        cost = 0
        prev_th = theta
        gradient = np.zeros([nb_features+1,1], dtype=np.float64)
        for x,y in zip(X,Y):
            h_x = hypothesis(theta,x)
            x = np.reshape(x, theta.shape)
            gradient += (h_x - y)*x
        
        
        theta = theta - learning_rate * gradient #batch update
        step_size = np.sum(abs(theta - prev_th))
        
        cost = loss(theta, X, Y)
        
        errors = np.append(errors, cost)
        
        
        print("Iteration %s - parameters %s" % (iterations, theta))
        iterations+=1
         
    
    end = time.time()
    print("\nMinimum found at theta = %s in %s sec with %s iterations" % (theta, end - start, iterations))

    return theta, errors




def stochastic_gradient_descent(X, Y, learning_rate=2, tolerance=0.01, max_iter=1000):
    start = time.time()
    
    #random initialisation of the parameters
    #theta = np.random.normal(size=[nb_features+1, 1], scale=0.2)
    theta = np.zeros([nb_features+1, 1], dtype=np.float64)
    
    size_tr = len(X)
    
    errors = [loss(theta, X, Y)]
    gradient = 0
    lr = learning_rate
    batch_size = 20
    
    cost = 0
    e = np.array([])
    
    for it in range(1, max_iter+1):
        
        #select random training sample
        rd_index = np.random.randint(size_tr, size=batch_size)
        xi = X[rd_index]
        yi = Y[rd_index]
        gradient = np.zeros([nb_features+1,1], dtype=np.float64)
        
        cost = 0
        
        for x,y in zip(xi,yi):
            h_x = hypothesis(theta,x)
            x = np.reshape(x, theta.shape)
            gradient += (y - h_x)*x 
            cost += loss(theta, x.T, y)
        
        
        
        theta = theta + lr * gradient 
    
        
        
        #errors.append(loss(theta, X, Y))
        errors.append(cost)


        error_d = np.linalg.norm(errors[it - 1] - errors[it])
        e = np.append(e, error_d)
        
        print("Iteration %s - error %s - parameters %s - err_d %s" % (it, errors[it], theta, error_d))
        
        
        #TO DO : decaying learning rate
        """if it > 100: 
            lr = exp_decay(it, learning_rate)"""
        
        
        
        if error_d < tolerance:
            print("Convergence!")
            break
    
    
    end = time.time()
    print("\nMinimum found at theta = %s in %s sec with %s iterations" % (theta, end - start, it))

    return theta, errors, e





if __name__ == "__main__":
    
    
    nb_features = 1 #You can change as you will

    #Generate the data
    
    X, y = datasets.make_regression(n_samples=500, n_features=nb_features, noise=50)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
    
    #need that theta0,x0=1 (simplification)
    
    train_o = np.ones([X_train.shape[0], 1], dtype=X_train.dtype)
    test_o = np.ones([X_test.shape[0],1], dtype=X_test.dtype)
    X_train = np.concatenate((train_o, X_train), axis=1)
    X_test = np.concatenate((test_o, X_test), axis=1)
    
    
    """ 
    
    plt.scatter(X_test[:,1], y_test, c='b')
    plt.scatter(X_train[:,1], y_train, c='r')
    
    
    plt.show()
    
    """
    
    
    #test 
    lr = 0.001
    p = 0.001
    
    
    #THETA, err = batch_gradient_descent(X_train, y_train, learning_rate = lr, accuracy = p)
    
    THETA, err, e = stochastic_gradient_descent(X_train, y_train, lr,p)
    
    f_y = THETA[:, 0] * X_test + THETA[0, 0]
    
    
    plt.figure(1)
    plt.scatter(X_test[:, 1], y_test, c='red', marker='o', edgecolors='black')
    plt.xlabel("First feature")
    plt.ylabel("Y")
    plt.plot(X_test, f_y, c='orange')
    
    
    plt.figure(2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(err)
    
    plt.show()
    
    












