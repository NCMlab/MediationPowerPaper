#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:02:29 2021

@author: jasonsteffener
"""
import sys
import os

sys.path.append(os.getcwd())


import ModMedTools as t
import multiprocessing as mp
import numpy as np
if __name__ == '__main__':
# Create a pool of workers based on the number cpus 
    pool = mp.Pool(mp.cpu_count())
    # There is one job for each equation that needs to be estimated
    # For simple mediation, there is are two equations: X --> M and X + M --> Y
    jobs1 = []
    jobs2 = []
    N_SAMPLE = 100
    data = t.MakeDataModel59(N_SAMPLE,[1,1,1,1],[1,1,1,1],[1,1,1,1,1,1,1,1])
    N_BOOTSTRAPS = 1000
    # Precalculate all of the bootstrap index arrays
    R1 = t.MakeBootResampleArray(N_SAMPLE, N_BOOTSTRAPS)
    R2 = t.MakeBootResampleArray(N_SAMPLE, N_BOOTSTRAPS, N_BOOTSTRAPS)
    # iterate n_bootstrap times
    for j in range(N_BOOTSTRAPS):
        
        # Calculate the linear regression beta weights by passing two arguments: 
            # the output and the model
        job1 = pool.apply_async(t.FitModel59,args=(t.ResampleData(data,R1[:,j])))
        job2 = pool.apply_async(t.FitModel59,args=(t.ResampleData(data,R2[:,j])))

 #       job2 = pool.apply_async(t.calculate_beta,args=(t.combine_array(_xy[0],formated_M),_xy[1]))

        jobs1.append(job1)
        jobs2.append(job2)
    # Close the pool
    pool.close()
    # By joining the pool this will wait until all jobs are completed
    pool.join()
    t.save_results(f'results-1.pkl',jobs1) 
    t.save_results(f'results-2.pkl',jobs2) 
