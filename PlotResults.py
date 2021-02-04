#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 08:58:18 2020

@author: jasonsteffener
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

BaseDir = '/Users/jasonsteffener/Documents/GitHub/PowerMediationResults'
FileName = 'SummaryDataFile.csv'
df = pd.read_csv(os.path.join(BaseDir, FileName))

df.head()


# fig1, ((ax1, ax2, ax3, ax4, a), (ax3, ax4)) = plt.subplots(5,5)



# Plot AtoC
a = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
a = [0.3, -0.3, 0.3, -0.3]
b = [-0, 0.1, 0.2, 0.3, 0.4, 0.5]
b = [-0.3, 0.3, 0.3, -0.3]
cP = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
cP = [ 0]
for k in cP:
    for i in a:
        for j in b:
            Filter = ((df["a"] == i) & (df["b"] == j) & (df["cP"] == k) & (df["typeA"] == 99))
            N = df[Filter]["N"]
            df2 = df[Filter]
            df2 = df2.sort_values("N")
            bStr = "%0.1f, %0.2f"%(j,df2['SbMean'].mean())
            plt.plot(df2["N"], df2["IEBCaPow"], label=bStr)
        plt.legend(title="b")
        plt.xlabel('Sample Size')
        plt.ylabel('Power of Indirect Effect')
        plt.title("a = %0.1f (%0.2f), c' = %0.1f"%(i,df2['SaMean'].mean(),k))
        plt.xlim(0,200)
        plt.ylim(0,1)
        plt.hlines(0.8,0,200,linestyles='dashed')
# Save each figure
        #OutFileName = "PowerPlot_a_%0.1f_cP_%0.1f_Atype_Uniform.png"%(i,k)
        #plt.savefig(os.path.join(BaseDir, OutFileName))        
        plt.show()


# Plot a=0.5, b = 0.3 AND a = 0.3, b = 0.5
a1 = 0.5
b1 = 0.2
a2 = 0.2
b2 = 0.5
Filter1 = ((df["a"] == a1) & (df["b"] == b1) & (df["cP"] == 0.0) & (df["typeA"] == 99))
Str1 = 'a = %0.1f, b = %0.1f'%(a1,b1)
Filter2 = ((df["a"] == a2) & (df["b"] == b2) & (df["cP"] == 0.0) & (df["typeA"] == 99))
Str2 = 'a = %0.1f, b = %0.1f'%(a2,b2)
N1 = df[Filter1]["N"]
N2 = df[Filter2]["N"]
df21 = df[Filter1]
df21 = df21.sort_values("N")
plt.plot(df21["N"], df21["IEBCaPow"], label = Str1)

df22 = df[Filter2]
df22 = df22.sort_values("N")
plt.plot(df22["N"], df22["IEBCaPow"], label = Str2)

plt.legend()
plt.xlabel('Sample Size')
plt.ylabel('Power of Indirect Effect')
plt.xlim(0,200)
plt.ylim(0,1)
plt.hlines(0.8,0,200,linestyles='dashed')
plt.show()



# Plot Compare CIs 

a1 = 0.4
b1 = 0.3
Filter1 = ((df["a"] == a1) & (df["b"] == b1) & (df["cP"] == 0.0) & (df["typeA"] == 99))
Str1 = 'BCa, a = %0.1f, b = %0.1f'%(a1,b1)
Str2 = 'BC'
Str3 = 'Perc'

N1 = df[Filter1]["N"]
N2 = df[Filter2]["N"]
df21 = df[Filter1]
df21 = df21.sort_values("N")
plt.plot(df21["N"], df21["IEBCaPow"], label = Str1)
plt.plot(df21["N"], df21["IEBCPow"], label = Str2)
plt.plot(df21["N"], df21["IEPercPow"], label = Str3)

plt.legend()
plt.xlabel('Sample Size')
plt.ylabel('Power of Indirect Effect')
plt.xlim(0,200)
plt.ylim(0,1)
plt.hlines(0.8,0,200,linestyles='dashed')
plt.show()


# Calculate how many more people are needed for every level of power
# between the BC and Perc methods
dBC = df21["IEBCPow"]
dPC = df21["IEPercPow"]
dN = df21["N"]
# Interpolate between the two curves
# Interpolate these curves to 1000 points
PowerInterp = np.arange(0,1.001,0.001)
NInterp = np.arange(0,200+200/1000,200/1000)
iBC = np.interp(NInterp, dN, dBC)
iPC = np.interp(NInterp, dN, dPC)
# cycle over the values of power and find the closest N values
# This will allow us to make a plot 
DiffPowLevels = np.arange(0.1,1,0.2)
DiffN = np.zeros(DiffPowLevels.shape)
count = 0
for i in DiffPowLevels:
    value = next(x for x in iBC if x > i)
    BCindex = np.where(iBC == value)[0][0]
    value = next(x for x in iPC if x > i)
    PCindex = np.where(iPC == value)[0][0]
    DiffN[count] = np.round(NInterp[PCindex] - NInterp[BCindex])
    count += 1

plt.plot(DiffPowLevels, DiffN)

# Compare A types
a1 = 0.5
b1 = 0.4
Filter1 = ((df["a"] == a1) & (df["b"] == b1) & (df["cP"] == 0.0) & (df["typeA"] == 1))
Filter2 = ((df["a"] == a1) & (df["b"] == b1) & (df["cP"] == 0.0) & (df["typeA"] == 2))
Filter3 = ((df["a"] == a1) & (df["b"] == b1) & (df["cP"] == 0.0) & (df["typeA"] == 99))
Str1 = 'Uniform, a = %0.1f, b = %0.1f'%(a1,b1)
Str2 = 'Dicotomous'
Str3 = 'Continuous'

df21 = df[Filter1]
df21 = df21.sort_values("N")
df22 = df[Filter2]
df22 = df22.sort_values("N")
df23 = df[Filter3]
df23 = df23.sort_values("N")

plt.plot(df21["N"], df21["IEBCaPow"], label = Str1)
plt.plot(df22["N"], df22["IEBCaPow"], label = Str2)
plt.plot(df23["N"], df23["IEBCaPow"], label = Str3)

plt.legend()
plt.xlabel('Sample Size')
plt.ylabel('Power of Indirect Effect')
plt.xlim(0,200)
plt.ylim(0,1)
plt.hlines(0.8,0,200,linestyles='dashed')
plt.show()
        
# TOTAL EFFECTS
# Plot AtoC
# a = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
a = [0.5]
# b = [-0, 0.1, 0.2, 0.3, 0.4, 0.5]
b = [0.5]
cP = [-0.5,-0.4,-0.3,-0.2,-0.1,0, 0.1, 0.2, 0.3, 0.4, 0.5]

for k in cP:
    for i in a:
        for j in b:
            Filter = ((df["a"] == i) & (df["b"] == j) & (df["cP"] == k) & (df["typeA"] == 99))
            N = df[Filter]["N"]
            df2 = df[Filter]
            df2 = df2.sort_values("N")
            bStr = "%0.1f, %0.2f"%(j,df2['SbMean'].mean())
            plt.plot(df2["N"], df2["TEBCaPow"], label=bStr)
        plt.legend(title="b")
        plt.xlabel('Sample Size')
        plt.ylabel('Power of Total Effect')
        plt.title("a = %0.1f (%0.2f), c' = %0.1f"%(i,df2['SaMean'].mean(),k))
        plt.xlim(0,200)
        plt.ylim(0,1)
        plt.hlines(0.8,0,200,linestyles='dashed')
# Save each figure
        #OutFileName = "PowerPlot_a_%0.1f_cP_%0.1f_Atype_Uniform.png"%(i,k)
        #plt.savefig(os.path.join(BaseDir, OutFileName))        
        plt.show()
        
        
# Fix a, fix b, change cP and see how Sb changes
# Plot N versus b and Sb for different cP values
a = [0.5]
# b = [-0, 0.1, 0.2, 0.3, 0.4, 0.5]
b = [-0.5]
cP = [0]


for i in a:
    for j in b:
        for k in cP:
            Filter = ((df["a"] == i) & (df["b"] == j) & (df["cP"] == k) & (df["typeA"] == 99))
            N = df[Filter]["N"]
            df2 = df[Filter]
            df2 = df2.sort_values("N")
            bStr = "%0.1f, %0.2f, %0.2f, %0.2f"%(j,df2['SbMean'].mean(),k,df2['SaMean'].mean())
            plt.plot(df2["N"], df2["SbMean"], label=bStr)
        plt.legend(title="b")
        plt.xlabel('Sample Size')
        plt.ylabel('Power of Total Effect')
        plt.title("a = %0.1f (%0.2f), c' = %0.1f"%(i,df2['SaMean'].mean(),k))
        plt.xlim(0,200)
        plt.ylim(0,1)
        plt.hlines(0.8,0,200,linestyles='dashed')
# Save each figure
        #OutFileName = "PowerPlot_a_%0.1f_cP_%0.1f_Atype_Uniform.png"%(i,k)
        #plt.savefig(os.path.join(BaseDir, OutFileName))        
        plt.show()
# SUPPRESSION
# Plot AtoC

a = [0.3]
b = [0.3]
cP = [-0.1]

for i in a:
    for j in b:
        for k in cP:
            Filter = ((df["a"] == i) & (df["b"] == j) & (df["cP"] == k) & (df["typeA"] == 99))
            N = df[Filter]["N"]
            df2 = df[Filter]
            df2 = df2.sort_values("N")
            bStr = "%0.1f"%(j)
            plt.plot(df2["N"], df2["IEBCaPow"], label='Indirect')
            plt.plot(df2["N"], df2["TEBCaPow"], label='Total')
        plt.legend(title="Effect")
        plt.xlabel('Sample Size')
        plt.ylabel('Power')
        plt.title("a = %0.1f, b = %0.1f, c' = %0.1f"%(i,j,k))
        plt.xlim(0,200)
        plt.ylim(0,1)
        plt.hlines(0.8,0,200,linestyles='dashed')
# Save each figure
        OutFileName = "SUPP_PowerPlot_a_%0.1f_b_%0.1f_cP_%0.1f_Atype_Uniform.png"%(i,j,k)
        plt.savefig(os.path.join(BaseDir, OutFileName))        
        plt.show()        
# FULL AND PARTIAL MEDIATION

a = [0.3]
b = [0.3]
cP = [0.1,0.3,0.5]

for i in a:
    for j in b:
        for k in cP:        
            Filter = ((df["a"] == i) & (df["b"] == j) & (df["cP"] == k) & (df["typeA"] == 99))
            N = df[Filter]["N"]
            df2 = df[Filter]
            df2 = df2.sort_values("N")
            bStr = "%0.1f"%(k)
            plt.plot(df2["N"], df2["IEBCaPow"], label=bStr)
            #plt.plot(df2["N"], df2["TEBCaPow"], label='Total')
        plt.legend(title="c'")
        plt.xlabel('Sample Size')
        plt.ylabel('Power')
        plt.title("a = %0.1f, b = %0.1f"%(i,j))
        plt.xlim(0,200)
        plt.ylim(0,1)
        plt.hlines(0.8,0,200,linestyles='dashed')
# Save each figure
        OutFileName = "FullPart_PowerPlot_a_%0.1f_b_%0.1f_Atype_Uniform.png"%(i,j)
        plt.savefig(os.path.join(BaseDir, OutFileName))        
        plt.show()        
                
# Make plots of collinearity
def PlotCollinearity():
    # Make plot of the power of DIRECT as a changes
    # Pick a sample size
    N = [20,40,60,80,100,120,140,160,180,200]
    
    N = [100]# Pick value

    b = [0.3]

    cP = [0.3]
    
    
    
    for k in cP:
        for j in b:
            for i in N:
                #Filter = ((df["a"] == i) & (df["b"] == j) & (df["cP"] == k) & (df["typeA"] == 99))
                Filter = ((df["N"] == i) &(df["b"] == j) & (df["cP"] == k) & (df["typeA"] == 99))
                
                N = df[Filter]["N"]
                a = df[Filter]["a"]
                df2 = df[Filter]
                df2 = df2.sort_values("a")
                #bStr = "%0.1f, %0.2f, %0.2f, %0.2f"%(j,df2['SbMean'].mean(),k,df2['SaMean'].mean())
                bStr = "%0d"%(i)
                plt.plot(df2["a"], df2["DEBCPow"], label=bStr)
            plt.legend(title="N")
            plt.xlabel('a')
            plt.ylabel('Power of Direct Effect')
            #plt.title("a = %0.1f (%0.2f), c' = %0.1f"%(i,df2['SaMean'].mean(),k))
            plt.title("b = %0.1f, Direct effect = %0.1f"%(j,k))
            plt.xlim(-0.5,0.5)
            plt.ylim(0,1)
            plt.hlines(0.8,-0.5,0.5,linestyles='dashed')
    # Save each figure
            #OutFileName = "PowerPlot_a_%0.1f_cP_%0.1f_Atype_Uniform.png"%(i,k)
            #plt.savefig(os.path.join(BaseDir, OutFileName))        
            plt.show()
    



