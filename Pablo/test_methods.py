# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:31:31 2018

@author: Pablo
"""

#import useful package 
from proj1_helpers import *
from ridge_regression import *
from split_data import *
from gradient_descent import *
from costs import *
from build_polynomial import *
from cross_validation import *
from least_squares import *
from plots import *
from implementations import *
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


y_tr,inputdata,ids=load_csv_data("train.csv")

def accuracys(y, tx, w):
    """Calculate the accuracy of a prediction
    """
    pred=predict_labels(w,tx)
    e=y-pred
    return 1-(sum(abs(e/2))/len(e)) 

def fast_buildpoly(x,ma,degree):
    return np.c_[ma,np.power(x, degree)]

#Take only the columns that are not sparce
#l = [np.count_nonzero(np.transpose(inputdata)[i] ==-999)==0 for i in range (30)] 
#xfilter = np.transpose(np.transpose(inputdata)[l])
#xf=(xfilter-np.mean(xfilter,axis=0))/np.std(xfilter,axis=0) #standardization


''' Test of gradient descent method with polynomial epxansion'''
#plot accuracies for different gammas
#degree=2
#acc=[]
#polyx=build_poly(xf,degree)
#max_iters, gamma = 50, np.linspace(0.1,1.2,25)
#w_init = np.ones(polyx.shape[1])
#for i in gamma:
#    w_opt , losses = gradient_descent(y,polyx,w_init,max_iters,i)
#    acc.append(accuracys(y,polyx,w_opt))
#    print('gamma= ', i,'  loss = ',losses[len(losses)-1]-losses[len(losses)-2],'  ',len(losses),'\n\n')
#plt.plot(gamma, acc, 'r') 
#print(max(acc)) # best acc = 0.72 for gammma = 0.64999 or 12th gamma
#plt.show()


#plot accuracies for different degrees

#degree = np.arange(10)
#acc=[]
#max_iters, gamma = 50, 0.64999
#for i in degree:
#    polyx=build_poly(xf,i)
#    w_init = np.ones(polyx.shape[1])
#    w_opt , losses = gradient_descent(y,polyx,w_init,max_iters,gamma)
#    acc.append(accuracys(y,polyx,w_opt))
#plt.plot(degree, acc, 'ro') 
#print(max(acc)) #best acc = 0.7489 for d=2
#plt.show()


def standardize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x-mean)/std

x_minus999_to_mean = inputdata.copy()
x_minus999_to_mean[x_minus999_to_mean == -999] = np.nan

col_means = np.nanmean(x_minus999_to_mean, axis=0)
inds = np.where(np.isnan(x_minus999_to_mean))
x_minus999_to_mean[inds] = np.take(col_means, inds[1])
#x_minus999_to_mean = standardize(x_minus999_to_mean)

'''Gradient descent with optimization'''

## NAG
#acc=[]
#x_clean= x_minus999_to_mean.copy()
#max_iters, gamma = 50, np.linspace(0.1,1,40)
#acc_g = 0.9
#w_init = np.ones(x_clean.shape[1])
#for i in gamma:
#    w_opt , losses = acc_GD(y,x_clean,w_init,max_iters,i,acc_g)
#    acc.append(accuracys(y,x_clean,w_opt))
#    print('gamma= ', i,'  loss = ',losses[len(losses)-1],'  ',len(losses),'\n\n')
#plt.plot(gamma, acc, 'r') 
#print(max(acc)) 
#plt.show()

''' Ridge regression '''

#e_ridge = []
#lambdas, degree = np.logspace(-8,0,25), 6
##polyx = np.ones((x_minus999_to_mean.shape[0], 1))
##polyx = fast_buildpoly( x_minus999_to_mean, polyx, degree)
#polyx = build_poly(x_minus999_to_mean,degree)
#for l in lambdas:
#    w, loss = ridge_regression(y_tr, polyx, l)
#    e_ridge.append(compute_categorical_loss(y_tr, polyx, w))
#    
#plt.semilogx(lambdas, e_ridge, 'r')
#plt.xlabel('Lambas')
#plt.ylabel('Error')
#plt.show()
#print('Best lambda= ', lambdas[np.argmin(e_ridge)],'  best e = ',np.min(e_ridge))

'''K -fold cross validation with 80-20% split'''
# used method : ridge regression

k_fold, seed = 5, 3
max_iters, gamma, d = 50, 1.07142, 9
k_ind = build_k_indices(y_tr, k_fold, seed)
lambda_ = 1e-8

loss_tr, loss_te = np.zeros(k_fold), np.zeros(k_fold)
for i in range(k_fold):
    loss_tr[i], loss_te[i]= cross_validation(y_tr,x_minus999_to_mean,k_ind,i,lambda_,d)
mu_tr = np.mean(loss_tr)
mu_te = np.mean(loss_te)
print('moy_tr = ',mu_tr,'  moy_te = ',mu_te,'\n\n')


