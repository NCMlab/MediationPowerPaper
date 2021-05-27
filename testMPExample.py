from multiprocessing import Pool
import ModMedTools as t
import time
from itertools import product
from functools import partial
import numpy as np

def f(x):
    return [x,x*x]

def g(x,N):
    output = t.RunAnalyses(x,N)
    return [output]
from contextlib import contextmanager

@contextmanager    
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()

it2=[]
N = np.arange(20,200,20)
sP = [1,0.66,0.33,0,-0.33, -0.66, -1]
SimParams = product(N,sP,sP,sP,sP,sP,sP,sP,sP)

SimParams = product(N,sP,[1],[1],[1],[1],[1],[1],[1])

if __name__ == '__main__':
    with poolcontext(processes=16) as pool:         # start 4 worker processes
        Npower = 1000
        #start = time.time()   
        #for j in range(N):
        #    result1 = pool.apply_async(f, ((10),)) # evaluate "f(10)" asynchronously in a single process
        #etime = time.time() - start
        #print(etime)          
        #print(result1.get(timeout=1))        # prints "100" unless your computer is *very* slow
             
        #print(pool.map(f, range(1000)))       # prints "[0, 1, 4,..., 81]"

        start = time.time()        
        
        # it1 = pool.map(g, (range(Npower)))
        for i in SimParams:
            # print("Working on Sample size: %d"%(i))
            it2.append(pool.imap(partial(t.RunAnalyses, N=i[0],SimParams=i[1:]), range(Npower)))
            
        pool.close()
        pool.join()
        etime = time.time() - start
        print("Ran %d times in %0.6f"%(Npower,etime))
        # print(next(it2))
        t.SaveMPresults(it2[3], Npower)    
        # t.SaveMPresults(it2, Npower)    
        #print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow

        #result = pool.apply_async(time.sleep, (10,))
        #print(result.get(timeout=1))        # raises multiprocessing.TimeoutError