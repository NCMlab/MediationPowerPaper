#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:18:56 2021

@author: jasonsteffener
"""
import numpy as np

def centered(data):
    cData = data - data.mean()
    return cData

# Make moderated data set
# Make A, B, C, D
# # Each has its own mean and standard deviation. 
# # # It would be interesting to see how the SD of a variable alters effects

def MakeDataModel59(N = 1000, means = [0,0,0,0], stdev = [1,1,1,1], 
                    weights = [0,0,0,0,0,0,0,0]):
    # means = A, B, C, D
    # weights = a1, a2, a3, b1, b2, c1P, c2P, c3P
    # Make sure everything is the correct size
    M = len(means)
    S = len(stdev)
    W = len(weights)
    # Add some error checking
    # try:
    data = np.zeros([N,M])
    # Create independent data
    # columns are A, B, C
    for i in range(M):
        # Columns: A, B, C, D
        data[:,i] = np.random.normal(means[i], stdev[i], N)
    AD = centered(data[:,0])*centered(data[:,3])
    BD = centered(data[:,1])*centered(data[:,3])
    # Make B = A + D + A*D
    data[:,1] = data[:,1] + data[:,0]*weights[0] + data[:,3]*weights[1] + AD*weights[2]
    # Make C = A + B + D + A*D + B*D 
    data[:,2] = data[:,2] + data[:,1]*weights[3] + BD*weights[4] + data[:,0]*weights[5] + data[:,3]*weights[6] + AD*weights[7]
    
    return data


def MakeData():
    data = MakeDataModel59(1000,[1,1,1,1],[1,1,1,1],[1,1,1,1,1,1,1,1])
    
    
def FitModel59(data):
    AD = centered(data[:,0])*centered(data[:,3])
    BD = centered(data[:,1])*centered(data[:,3])
    # Model of B
    X = np.vstack([data[:,0], data[:,3], AD]).transpose()
    beta, intercept = calculate_beta(X, data[:,1])
    # Model of C
    X = np.vstack([data[:,1], BD, data[:,0], data[:,3], AD]).transpose()
    beta, intercept = calculate_beta(X, data[:,2])