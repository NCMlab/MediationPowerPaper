#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:15:52 2020

@author: jasonsteffener
"""
import pandas as pd
import os
import time
# Where to dind the batch scripts
BaseDir = "/home/steffejr/scratch/Project/jobs"
# Where to find the output files
# OutDir = '/Users/jasonsteffener/Documents/GitHub/PowerMediationResults'
OutDir = '/home/steffejr/Data/Results'
# WHat is the submission list filename
fileName = "SubmissionList.csv"
# read the file
df = pd.read_csv(os.path.join(OutDir, fileName))

CountSubmitted = 0
SubmitLimit = 700
ListOfResubmit = []
for index, row in df.iterrows():
    # Check to see if a row has data
    if not row['Completed'] == 1.0:
        # resubmit the batch
        CurrentSimId = row['SimID']
        
        
        # If this SimID has not already been resubmitted, then do so 
        if not CurrentSimId in ListOfResubmit:
            # Add the submitted job to the list
            ListOfResubmit.append(CurrentSimId)
            # Find the batch file
            CurrentSimId = '%06d'%(CurrentSimId)
            fileName = "submit_Process_%s"%(CurrentSimId)            
            # submit the batch job
            if CountSubmitted < SubmitLimit:
                os.system('sbatch %s'%(os.path.join(BaseDir, fileName+".sh")))
                time.sleep(0.5)
                CountSubmitted += 1
