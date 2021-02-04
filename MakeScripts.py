#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:53:26 2020

@author: jasonsteffener
"""
import os
import numpy as np
import pandas as pd
import time
def MakeBatchScripts():
    # Make all of the submission scripts.
    # Also make a dataframe to keep track of all of the expected simulations
    cNames = ['Nboot','NSim','N','typeA', 'a', 'b', 'cP','SimID','Completed']
    dfOut = pd.DataFrame(columns=cNames)

    BaseDir = "/home/steffejr/scratch/Project"
    BaseDir = "/home/steffejr/Data"
    OutDir = "/home/steffejr/Data/Results"
    #OutDir = '/Users/jasonsteffener/Documents/GitHub'
    CodeDir = "/home/steffejr/scratch/ProcessModelsPython"
    SubmissionListFileName = os.path.join(OutDir,'SubmissionList.csv')    
    N = np.arange(10,201,10)
    #N = [100]
    typeA = [99,1,2] # cont, unif, dicotomous     
    aLIST = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]#np.arange(-0.5,0.1,0.5)
    cPLIST = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]# np.arange(-1.0,1.01,0.5)
    bLIST = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]# np.arange(-1.0,1.01,0.5)    
    Nboot = 1000
    Nsim = 1000    
    count = 0
    for i1 in N:
        for i3 in typeA:
            for i8 in aLIST:
                for i9 in cPLIST:
                    #for i10 in BtoC:
                        count += 1
                        if count > -1:
                            # Create the script file
                            SimID = '%06d'%(count)
                            fileName = "submit_Process_%s"%(SimID)
                            f = open(os.path.join(BaseDir, 'jobs', fileName+".sh"), "w")
                            f.write("#!/bin/bash\n")                 
                            f.write("#SBATCH --job-name=%s.job\n"%(os.path.join(BaseDir, 'jobs', fileName)))
                            f.write("#SBATCH --output=%s.out\n"%(os.path.join(BaseDir, 'out', fileName)))
                            f.write("#SBATCH --error=%s.err\n"%(os.path.join(BaseDir, 'out', fileName)))
                            f.write("#SBATCH --time=06:00:00\n")
                            f.write("#SBATCH --account=def-steffejr-ab\n")
                            f.write("#SBATCH --mem-per-cpu=512M\n\n")
                            # Added an array for at least one dimension of simulations
                            f.write("#SBATCH --array=1-11\n")
                            f.write("source ~/ENV/bin/activate\n")
                            f.write("python %s %d %d %d %d %0.2f %0.2f %s %s\n" %(os.path.join(CodeDir, "ProcessTools.py"), Nboot,Nsim,i1,i3, i8,i9, OutDir, '$SLURM_ARRAY_TASK_ID'))
                            f.close()
                            # Add this sim to the dataframe keeping track
                            for j in bLIST:
                                li = [Nboot, Nsim, i1, i3, i8, j, i9, SimID, 0]
                                row = pd.Series(li, index = cNames)
        
                                dfOut = dfOut.append(row, ignore_index = True)
                            
                            # submit the file to the queue
                            # os.system('sbatch %s'%(os.path.join(BaseDir, 'jobs', fileName+".sh")))
                            # time.sleep(0.1)
    print("Saving submission list to file: %s"%SubmissionListFileName)
    dfOut.to_csv(SubmissionListFileName)
    print(count)

if __name__ == "__main__":
    MakeBatchScripts()
