#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:38:43 2021

@author: jasonsteffener
"""

import multiprocessing as mp
import time
import os
import numpy as np
import ModMedTools as t
import pandas as pd
def f(x):
    return np.sqrt(x)

if __name__ == '__main__':
    # start 4 worker processes
    jobs1 = []
    a = np.arange(16)
    pool = mp.Pool(mp.cpu_count())
        # print "[0, 1, 4,..., 81]"
    start_time = time.time()
        #print(pool.map(t.RunAnalyses, a))
    job1 = pool.apply_async(t.RunAnalyses, args=(a))
        #jobs1.append(job1)
    # Close the pool
    print("--- %s seconds ---" % (time.time() - start_time))
    pool.close()
    print("--- %s seconds ---" % (time.time() - start_time))
    # By joining the pool this will wait until all jobs are completed
    pool.join()
    print(job1)
    for result in job1:
        res = result.get()
        print(res)
    
   
    
    print("--- %s seconds ---" % (time.time() - start_time))
   # t.save_results('results-1.pkl',jobs1) 
    
    print("--- %s seconds ---" % (time.time() - start_time))
        
    start_time = time.time()
        # print(f(a))
        # print("--- %s seconds ---" % (time.time() - start_time))
        
    start_time = time.time()

    t.RunAnalyses(1)
    print("--- %s seconds ---" % (time.time() - start_time))
        
        # # print same numbers in arbitrary order
        # for i in pool.imap_unordered(f, range(10)):
        #     print(i)

        # # evaluate "f(20)" asynchronously
        # res = pool.apply_async(f, (20,))      # runs in *only* one process
        # print(res.get(timeout=1))             # prints "400"

        # # evaluate "os.getpid()" asynchronously
        # res = pool.apply_async(os.getpid, ()) # runs in *only* one process
        # print(res.get(timeout=1))             # prints the PID of that process

        # # launching multiple evaluations asynchronously *may* use more processes
        # multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        # print([res.get(timeout=1) for res in multiple_results])

        # # make a single worker sleep for 10 secs
        # res = pool.apply_async(time.sleep, (10,))
        # try:
        #     print(res.get(timeout=1))
        # except TimeoutError:
        #     print("We lacked patience and got a multiprocessing.TimeoutError")

        # print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")