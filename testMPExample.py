from multiprocessing import Pool
import ModMedTools as t
import time

def f(x):
    return [x,x*x]

def g(x):
    output = t.RunAnalyses(x)
    return [output]
    
if __name__ == '__main__':
    with Pool(processes=16) as pool:         # start 4 worker processes
        N = 10
        #start = time.time()   
        #for j in range(N):
        #    result1 = pool.apply_async(f, ((10),)) # evaluate "f(10)" asynchronously in a single process
        #etime = time.time() - start
        #print(etime)          
        #print(result1.get(timeout=1))        # prints "100" unless your computer is *very* slow
             
        #print(pool.map(f, range(1000)))       # prints "[0, 1, 4,..., 81]"

        start = time.time()        
        
        it = pool.imap(g, range(N))
        pool.close()
        pool.join()
        etime = time.time() - start
        print("Ran %d times in %0.6f"%(N,etime))
        
                  # prints "0"
        print(next(it))                     # prints "1"
        print(next(it))      
        print(next(it))      
        print(next(it))      
        #print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow

        #result = pool.apply_async(time.sleep, (10,))
        #print(result.get(timeout=1))        # raises multiprocessing.TimeoutError