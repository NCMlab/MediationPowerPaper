#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:18:56 2021

@author: jasonsteffener
"""
import numpy as np
from sklearn import linear_model
from sklearn.utils import resample
import time
import multiprocessing as mp
import pandas as pd
from scipy.stats import norm

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

def MakeBootResampleList(N, i = 1):
    """ Return a bootstrap ressample array.
    It is super important that the seed is reset properly. This is especially 
    true when sending this out to a cluster. This is why the current time is 
    offset with an index."""
    np.random.seed(int(time.time())+i)
    return np.random.choice(np.arange(N),replace=True, size=N)

def MakeBootResampleArray(N, M, offset = 0):    
    """ Make an array of bootstrap resamples indices """
    data = np.zeros((N,M)).astype(int)
    for i in range(M):
        data[:,i] = MakeBootResampleList(N, i + offset)
    return data

def ResampleData(data, resamples):
    """ resample the data using a list of bootstrap indices """
    return data[resamples,:]
    
def RunAnalyses(index):
    print(index)
    N = 200
    NBoot = 2000
    data = MakeDataModel59(N,[1,1,1,1],[1,1,1,1],[1,1,1,1,1,1,1,1])
    ResampleArray = MakeBootResampleArray(N,NBoot)
    
    PEbetaB, PEbetaC = FitModel59(data)
    BSbetaB, BSbetaC = ApplyBootstrap(data, ResampleArray)
    PEDir, PEInd, PETot = CalculatePathsModel59(PEbetaB, PEbetaC, 1)
    BSDir, BSInd, BSTot = CalculatePathsModel59(BSbetaB, BSbetaC, 1)
    # combine data so CI can be easily calculated
    AllB = []
    count = 0
    for i in PEbetaB:
        AllB.append([PEbetaB[count], BSbetaB[count,]])
        count += 1
    count = 0    
    for i in PEbetaC:
        AllB.append([PEbetaC[count], BSbetaC[count,]])
        count += 1  
    AllB.append([PEDir, BSDir])
    AllB.append([PEInd, BSInd])
    AllB.append([PETot, BSTot])
    AllSign = np.zeros(len(AllB))
    count = 0
    for i in AllB:
        tempPer, tempBC, bias = CalculateCI(i[1], i[0])
        if np.sign(np.array(tempBC).prod()) > 0:
            AllSign[count] = 1
            count += 1
    return AllSign

    
    

def calculate_beta(x,y):
    """Returns estimated coefficients and intercept for a linear regression problem.
    
    Keyword arguments:
    x -- Training data
    y -- Target values
    """
    reg = linear_model.LinearRegression().fit(x, y)
    return reg.coef_,reg.intercept_

    
def FitModel59(data):
    AD = centered(data[:,0])*centered(data[:,3])
    BD = centered(data[:,1])*centered(data[:,3])
    # Model of B
    X = np.vstack([data[:,0], data[:,3], AD]).transpose()
    betaB, interceptB = calculate_beta(X, data[:,1])
    # Model of C
    X = np.vstack([data[:,1], BD, data[:,0], data[:,3], AD]).transpose()
    betaC, interceptC = calculate_beta(X, data[:,2])
    return betaB, betaC  
 #   D = 1
 #   CondDirectEffect, CondIndirectEffect, ConditionalTotalEffect = CalculatePathsModel59(betaB, betaC, D)    
    
    # Output should be a dataframe
    # This will likely make things easier to save
##    columnNames = CreateListOfColumnNames(betaB,betaC)
  #  columnNames.append("D")
  #  results = np.concatenate((betaB, betaC, [CondDirectEffect, CondIndirectEffect, ConditionalTotalEffect, D]))
  #  df = pd.DataFrame([results], columns=columnNames)
    
#    return df

def CalculatePathsModel59(betaB, betaC, D):
    # Take the parameter estimates (beta) and a probe value and 
    # calculate the conditional effects at the probe value
    CondDirectEffect = betaC[2,] + betaC[4,]*D
    CondIndirectEffect = (betaB[0,]+betaB[2,]*D)*(betaC[0,]+betaC[1,]*D)
    ConditionalTotalEffect = CondDirectEffect + CondIndirectEffect
    return CondDirectEffect, CondIndirectEffect, ConditionalTotalEffect

def ApplyBootstrap(data, ResampleArray):
    # Calculate the Point Estimate
    PEbetaB, PEbetaC = FitModel59(data)
    NBoot = ResampleArray.shape[1]
    # Make output arrays
    BSbetaB = np.zeros([PEbetaB.shape[0], NBoot])
    BSbetaC = np.zeros([PEbetaC.shape[0], NBoot])    
    for i in range(NBoot):
        tempB, tempC = FitModel59(ResampleData(data, ResampleArray[:,i]))
        BSbetaB[:,i] = tempB
        BSbetaC[:,i] = tempC
    return BSbetaB, BSbetaC

def CalculateCI(BS, PE, alpha=0.05):
    """Calculate confidence intervals from the bootstrap resamples
    
    Confidence intervals are calculated using two difference methods:
        Percentile
            This method finds the alpha/2 percentiles at both ends of 
            the distribution of bootstratp resamples. First, the 
            index for these limits is found as: NBoot*(alpha/2).
            If this is a non interger value, it is rounded in the more
            conservative direction. Using these indices, the bootstrap 
            values are the confidence intervals.

        Bias-corrected
            It is possible that the distribution of bootstrap resmaples
            are biased with respect to the point estimate. Ideally,
            there whould be an equal number of bootstrap resample values
            above and below the point estimate. And difference is considered
            a bias. This approach adjusts for this bias. If there is no bias
            in the bootstrap resamples, the adjustment factor is zero and no
            adjustment is made so the result is the same as from the percentile
            method.
     
    Parameters
    ----------
    BS : array of length number of bootstrap resamples
        bootstrap resamples.
    PE : float
        the point estimate value.
    alpha : float
        statstical alpha used to calculate the confidence intervals.

    Returns
    -------
    PercCI : array of two floats
        Confidence intervals calculated using the percentile method.
    BCCI : array of two floats
        Confidence intervals calculated using the bias-corrected method.
    Bias : float
        The size of the bias calculated from the distribution of bootstrap
        resamples.
    """
    # If there were no bias, the zh0 would be zero
    # The percentile CI assume bias and skew are zero
    # The bias-corrected CI assume skew is zero
    NBoot = BS.shape[0]
    zA = norm.ppf(alpha/2)
    z1mA = norm.ppf(1 - alpha/2)
    
    # Percentile
    Alpha1 = norm.cdf(zA)
    Alpha2 = norm.cdf(z1mA)
    PCTlower = np.percentile(BS,Alpha1*100)
    PCTupper = np.percentile(BS,Alpha2*100)
    PercCI = [PCTlower, PCTupper]

    # Find resamples less than point estimate
    F = np.sum(BS < PE)
    if F > 0:
        pass
    else:
        F = 1 
    # Estimate the bias in the BS
    zh0 = norm.ppf(F/NBoot)
    # Calculate CI using just the bias correction
    Alpha1 = norm.cdf(zh0 + (zh0 + zA))
    Alpha2 = norm.cdf(zh0 + (zh0 + z1mA))
    PCTlower = np.percentile(BS,Alpha1*100)
    PCTupper = np.percentile(BS,Alpha2*100) 
    BCCI = [PCTlower, PCTupper]
    # Record the amount of bias and the amount of skewness in the resamples
    Bias = F/NBoot
    return PercCI, BCCI, Bias


def CreateListOfColumnNames(betaB,betaC):
    # Create column names based on the length of the beta vectors
    columns = []
    count = 0
    for i in betaB:
        columns.append('coefB%02d'%(count))
        count += 1
    count = 0
    for i in betaC:
        columns.append('coefC%02d'%(count))
        count += 1
    columns.append('CondDir')
    columns.append('CondIndir')
    columns.append('CondTot')
    return columns
    
def save_results(path, results):
    """ Saves the estimated coefficients and intercept into a pickle file.
    Keyword arguments:
    path -- name of the path to store the results
    results -- job list of estimated betas
    """
    new_df = pd.DataFrame()
    print(results)
    for f in results:
        stuff = f.get(timeout=60)
        print(stuff)
        count = 0
        for i in stuff:
            print(count)
            new_df = new_df.append(stuff)
            count += 1

    #(f'results-{os.environ["SLURM_JOBID"]}.pkl')
    new_df.to_pickle(path)

