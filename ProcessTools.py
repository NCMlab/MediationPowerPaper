
from sklearn import linear_model
from sklearn.utils import resample
import numpy as np
from scipy.stats import norm
import scipy.stats 
import time
import sys
import os
import pandas as pd
import pingouin as pg

def MakeModeratedEffect(data,i = 0, j = 1, effect = 0):
    """Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> func(1, "a")
    True
    """
    # How big is the data
    N,M = data.shape
    # Remove mean of each column
    iD = data[:,i] - data[:,i].mean()
    print(iD.mean())
    jD = data[:,j] - data[:,j].mean()
    print(jD.mean())
    # Create the moderation regressor. 
    # This regressor is a vector and is converted to an array for the multiplication
    mD = np.array(iD*jD)
    print(mD.mean())
    # Add a moderator effect to the DV
    OutData = np.append(data, np.expand_dims(mD, axis = 1), axis = 1)
    OutData[:,-1] = OutData[:,-1] + mD*effect
    # Append the moderator to the data 
    return OutData

def CalculateMediationPEEffect(PointEstimate2, PointEstimate3, ia = 0, ib = 1):
    """Calculate derived effects from simple mediation model.
    
    Given parameter estimates from a simple mediation model,
    calculate the indirect effect, the total effect and the indirect effects
    

    Parameters
    ----------
    PointEstimate2 : array
        This is an array of parameter estimates for the regression equation
        of A on B. With no covariates, this will be an array of length 1
    PointEstimate3 : array
        This is an array of parameter estimates for the regression equation
        of A and B on C. With no covariates, this will be an array of length 2
    ia : int 
        The index of the parameter of interest from the array of beta values
        in PointEstimate2
    ib : int
        The index of the parameter of interest from the array of beta values
        in PointEstimate3

    Returns
    -------
    IE
        The indirect effect, parameter a times b
    TE
        The total effect, which is IE plus DE
    DE
        The direct effect, the effect of A on C, when B is in the model
    a
        The effect of A on B
    b
        The effect of B on C, when A is in the model

    """
    # Indirect effect
    # The model of B with A has one beta which is index 0
    a = PointEstimate2[0][ia]
    # The model of C with A and B has two betas, b has index = 1
    b = PointEstimate3[0][ib]
    IE = a*b
    # Direct effect
    DE = PointEstimate3[0][0] # This is cP
    TE = DE + IE
    return IE, TE, DE, a, b
    
def CalculateMediationResampleEffect(BSbetalist2, BSbetalist3, ia = 0, ib = 1):
    # Indirect effect
    a = np.squeeze(BSbetalist2[:,ia])
    b = np.squeeze(BSbetalist3[:,ib])
    IE = a*b
    DE = np.squeeze(BSbetalist3[:,0])
    TE = DE + IE
    return IE, TE, DE, a, b
    
def JackKnife(func, data):
    N,M = data.shape
    # prepare output arrays
    betalist = np.zeros([N,M-1])
    interlist = np.zeros([N,1])
    temp = np.arange(N)
    for i in range(0,N):
        JKresample = data[np.delete(temp, i),:]
        # randomly resample the dataset with the original set with replacement
        tempP = func(JKresample)
        betalist[i,:] = tempP[0]
        interlist[i] = tempP[1]
    return betalist,interlist

def CalculateCI(BS, JK, PE, alpha):
    """Calculate confidence intervals from the bootstrap resamples
    
    Confidence intervals are calculated using three difference methods:
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
        Bias-corrected, accelerated
            In addition to there being a possible bias in the bootstrap resamples,
            it is also possible that there is some skew. The amount of skew is
            calculated and used to adjust the confidence intervals in this method.
            If there is no skew, this method gives the same as the bias-correct
            approach. If there is no skew and no bias the results are the 
            same as the percentile method.
    Parameters
    ----------
    BS : array of length number of bootstrap resamples
        bootstrap resamples.
    JK : array of length N, the sample size
        Jack-knife resamples.
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
    BCaCI : array of two floats
        Confidence intervals calculated using the bias-correctd, accelerated method.
    Bias : float
        The size of the bias calculated from the distribution of bootstrap
        resamples.
    BSskew : float
        The size of the skew calculated from the distribution of bootstrap
        resamples.
    BSskewStat : float
        The statstic associated with the calculated skew in the 
        distribution of bootstrap resamples.

    """
    # If there were no bias, the zh0 would be zero
    # If there wer no skew, the acc would be zero
    # The percentile CI assume bias and skew are zero
    # The bias-corrected CI assume skew is zero
    # The bias-correct-accelerated assumes nothing
    NBoot = BS.shape[0]
    N = JK.shape[0]
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
    BCaCI = [-1, 1]
    if F > 0:
        # Estimate the bias in the BS
        zh0 = norm.ppf(F/NBoot)
        # Calculate CI using just the bias correction
        Alpha1 = norm.cdf(zh0 + (zh0 + zA))
        Alpha2 = norm.cdf(zh0 + (zh0 + z1mA))

        PCTlower = np.percentile(BS,Alpha1*100)
        PCTupper = np.percentile(BS,Alpha2*100)
        BCCI = [PCTlower, PCTupper]
        # Calculate the skew/acceleration factor
        # Adjust the confidence limits based on the skewness
        ThetaDiff = JK.sum()/N - JK
        acc = ((ThetaDiff**3).sum())/(6*((ThetaDiff**2).sum())**(3/2))
        Alpha1 = norm.cdf(zh0 + (zh0 + zA)/(1 - acc*(zh0 + zA)))
        Alpha2 = norm.cdf(zh0 + (zh0 + z1mA)/(1 - acc*(zh0 + z1mA)))

        PCTlower = np.percentile(BS,Alpha1*100)
        PCTupper = np.percentile(BS,Alpha2*100)
        BCaCI = [PCTlower, PCTupper]
    # Record the amount of bias and the amount of skewness in the 
    # resamples
    Bias = F/NBoot
    BSskew = scipy.stats.skew(BS)
    BSskewStat = scipy.stats.skewtest(BS)[0]
    return PercCI, BCCI, BCaCI, Bias, BSskew, BSskewStat


def CaclulatePercCI(BS, alpha):
    # count the samples
    NBoot = BS.shape[0]
    # sort the resamples
    sBS = np.sort(BS)
    # Find the limits
    Lower = int(np.floor(NBoot*alpha/2))
    Upper = int(NBoot - np.ceil(NBoot*alpha/2))
    percCI = [sBS[Lower], sBS[Upper]]
    return percCI

def ExploreIdea2():
    [a, b, cP, Sa, Sb, ScP, IE, SIE, data] = CalculateSimulatedEffectSizes(1000, 0.5, 0.5, 0.5, 99)
    df = pd.DataFrame(data, columns={'C','A','B'})
    print(np.corrcoef(data.T))
    pg.partial_corr(data=df, x='A', y='C', covar='B').round(3)
    
def ExploringIdea():  
    # How do the variance in the simulated data, the weights, the betas and the Bs 
    # relate to each other?
    N = 1000
    
    data = MakeIndependentData(N, [1,1,1], [0.001,0.001,0.001], [0.25, -0.25, 0.75], 99)

    PointEstimate2 = Calculate_Beta_Sklearn(data[:,[0,1]])
    PointEstimate3 = Calculate_Beta_Sklearn(data)
        # Point estimate mediation effects
    IE, TE, DE, a, b = CalculateMediationPEEffect(PointEstimate2, PointEstimate3)
    print("a: %0.3f"%(a))
    print("b: %0.3f"%(b))
    print("cP: %0.3f"%(DE))
    print("TE: %0.3f"%(TE))
    print("IE: %0.3f"%(IE))
    print(Calculate_standardizedB(data, PointEstimate3[0]))
    # print(CalculateKappaEffectSize(data, a, b))
    
    N=100
    NBoot = 1000
    data = MakeIndependentData(N, [1,1,1], [1,1,1], [0.2, 0.2, 0], 1)
    # data = MakeMultiVariableData(N,means, covs)
    # Point estimates
    #  Model of B with A, parameter is a
    PointEstimate2 = Calculate_Beta_Sklearn(data[:,[0,1]])
    # Model of C with A and B, parameters are cp, b
    PointEstimate3 = Calculate_Beta_Sklearn(data)
    # Point estimate mediation effects
    IE, TE, DE, Act_a, Act_b = CalculateMediationPEEffect(PointEstimate2, PointEstimate3)
    
    # Bootstrap model 2
    BSbetalist2, BSinterlist2 = Bootstrap_Sklearn(NBoot,Calculate_Beta_Sklearn, data[:,[0,1]])
    JKbetalist2, JKinterlist2 = JackKnife(Calculate_Beta_Sklearn, data[:,[0,1]])
    # Bootstrap model 3
    BSbetalist3, BSinterlist3 = Bootstrap_Sklearn(NBoot,Calculate_Beta_Sklearn, data)
    JKbetalist3, JKinterlist3 = JackKnife(Calculate_Beta_Sklearn, data)
    # Bootstrap mediation effects
    BSIE, BSTE, BSDE, BSa, BSb = CalculateMediationResampleEffect(BSbetalist2, BSbetalist3)
    # Jackknifemediation effects
    JKIE, JKTE, JKDE, JKa, JKb = CalculateMediationResampleEffect(JKbetalist2, JKbetalist3)
    
    alpha = 0.05
    [IEPercCI, IEBCCI, IEBCaCI, IEBSskew, IEBSskewStat] = CalculateCI(BSIE, JKIE, IE, alpha)
    
 

    
    
    
def CalculateSimulatedEffectSizes(N, a, b, cP, typeA):
    # Using provided assigned effects, calculate the estimated effects
    # and the standardied effect sizes
    data = MakeIndependentData(N, [1,1,1], [1,1,1], [a, b, cP], typeA)
    PointEstimate2 = Calculate_Beta_Sklearn(data[:,[0,1]])
    PointEstimate3 = Calculate_Beta_Sklearn(data)
    # Point estimate mediation effects
    IE, TE, DE, a, b = CalculateMediationPEEffect(PointEstimate2, PointEstimate3)
    temp2 = Calculate_standardizedB(data[:,[0,1]], PointEstimate2[0])
    temp3 = Calculate_standardizedB(data, PointEstimate3[0])
    temp4 = Calculate_standardizedB(data, PointEstimate3[1])
    Sa = temp2[0]
    Sb = temp3[1]
    ScP = temp4[0]
    cP = DE
    SIE = CalculateKappaEffectSize(data, a, b)
    #return Sa, Sb, ScP, SIE
    return a, b, cP, Sa, Sb, ScP, IE, SIE, data

def RunEffectSizeSimulations(b):
    cNamesAll = ['N','NSim','typeA','Exp_a','Exp_b','Exp_cP', 'mAct_a','stdAct_a','mAct_b','stdAct_b', 'mAct_cP','stdAct_cP','m_IE','std_IE', 'm_Sa','std_Sa','m_Sb','std_Sb', 'm_ScP','std_ScP','m_K','std_K']
    dfOutAll = pd.DataFrame(columns=cNamesAll)
    N = np.arange(10,101,10)
    typeA = [99,1,2] # cont, unif, dicotomous     
    aLIST = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]#np.arange(-0.5,0.1,0.5)
    cPLIST = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]# np.arange(-1.0,1.01,0.5)
    #BtoC = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]# np.arange(-1.0,1.01,0.5)    
    Nsim = 100   
    count = 0
    
    for i1 in N:
        for i3 in typeA:
            for i8 in aLIST:
                for i9 in cPLIST:
                    #for i10 in BtoC:
                    cNames = ['a', 'b', 'cP', 'Sa','Sb','ScP','IE','SIE']
                    dfOut = pd.DataFrame(columns=cNames)
                    for s in range(Nsim):
                        li = CalculateSimulatedEffectSizes(i1, i8, b, i9, i3)
                        row = pd.Series(li, index = cNames)
                        dfOut = dfOut.append(row, ignore_index = True)
                    liAll = [i1, Nsim, i3, i8, b, i9, dfOut['a'].mean(), dfOut['a'].std(), dfOut['b'].mean(), dfOut['b'].std(), dfOut['cP'].mean(), dfOut['cP'].std(), dfOut['IE'].mean(), dfOut['IE'].std()]
                    liAll.append(dfOut['Sa'].mean())
                    liAll.append(dfOut['Sa'].std()) 
                    liAll.append(dfOut['Sb'].mean()) 
                    liAll.append(dfOut['Sb'].std()) 
                    liAll.append(dfOut['ScP'].mean())
                    liAll.append(dfOut['ScP'].std())
                    liAll.append(dfOut['SIE'].mean()) 
                    liAll.append(dfOut['SIE'].std())
                    
                    row = pd.Series(liAll, index = cNamesAll)
                    dfOutAll = dfOutAll.append(row, ignore_index = True)
                    count += 1
                    print(count)
    dfOutAll.to_csv('SimulationsOfEffectSize_%0.1f.csv'%(b))
                        
                        
def CalculateKappaEffectSize(data, a, b):
    # https://github.com/NCMlab/ProcessModelsNeuroImage/blob/master/FinalCode/CalculateKappa2.m
    COV = np.cov(data.T)

    # Calculate the permissible values of a
    perm_a_1 = (COV[2,1] * COV[2,0] + np.sqrt(COV[1,1] * COV[2,2] - COV[2,1]**2) * np.sqrt(COV[0,0] * COV[2,2] - COV[2,0]**2))/(COV[0,0] * COV[2,2])
    perm_a_2 = (COV[2,1] * COV[2,0] - np.sqrt(COV[1,1] * COV[2,2] - COV[2,1]**2) * np.sqrt(COV[0,0] * COV[2,2] - COV[2,0]**2))/(COV[0,0] * COV[2,2])
    perm_a = np.array([perm_a_1, perm_a_2])
    # Calculate the permissible values of b
    perm_b_1 =  np.sqrt(COV[0,0] * COV[2,2] - COV[2,0]**2)/np.sqrt(COV[0,0] * COV[1,1] - COV[0,1]**2)
    perm_b_2 = -np.sqrt(COV[0,0] * COV[2,2] - COV[2,0]**2)/np.sqrt(COV[0,0] * COV[1,1] - COV[0,1]**2)
    perm_b = np.array([perm_b_1, perm_b_2])
    # Check the a values and find the one in the same direction as the point estimate of a  
    if a > 0:
        temp = perm_a[perm_a > 0]
        max_a = max(temp)
    elif a < 0:
        temp = perm_a[perm_a < 0]
        max_a = min(temp)
    else:
        max_a = 0
    # Check the b values and find the one in the same direction as the point
    # estimate of b
    if b > 0:
        temp = perm_b[perm_b > 0]
        max_b = max(temp)
    elif b < 0:
        temp = perm_b[perm_b < 0]
        max_b = min(temp)
    else:
        max_b = 0
    
    # calculate kappa    
    perm_ab = max_a*max_b
    return (a*b)/perm_ab
    

def CalculateRegressionEffectSizes():
    pass
    

def Calculate_Beta_Sklearn(data):
    # using linear regression model from sklearn 
    lm = linear_model.LinearRegression()
    # M = data.shape[1]
    #fit the x,y to the model,x will be a 2d matrix and y is a array
    # if M > 2:
    model = lm.fit(data[:,0:-1], data[:,-1])
    # else:
    #     model = lm.fit(data[:,[0]], data[:,-1])
    return model.coef_, model.intercept_

def Calculate_standardizedB(data, beta):
    return beta*data[:,0:-1].std(0)/data[:,-1].std()


def Bootstrap_Sklearn(NBoot, func, data):
    # How big is the data
    N,M = data.shape
    # prepare output arrays
    betalist = np.zeros([NBoot,M-1])
    interlist = np.zeros([NBoot,1])
    for i in range(0,NBoot):
        # randomly resample the dataset with the original set with replacement
        a = resample(data, n_samples=N, replace=True, random_state=i)
        temp = func(a)
        betalist[i,:] = temp[0]
        interlist[i] = temp[1]
    return betalist,interlist


def MakeIndependentData(N = 1000, means = [0,0,0], stdev = [1,1,1], weights = [0, 0, 0], Atype = 99):
    # means = A, B, C
    # weights = a, b, cP
    # Make sure everything is the correct size
    M = len(means)
    S = len(stdev)
    W = len(weights)
# try:
    data = np.zeros([N,M])
    # Create independent data
    # columns are A, B, C
    for i in range(M):
        data[:,i] = np.random.normal(means[i], stdev[i], N)
    if Atype == 1:
        data[:,0] = np.random.uniform(20,80,N)
    if Atype == 2:
        data[:,0] = np.concatenate((np.zeros(int(N/2)), np.ones(int(N/2))))
    # Add weights between predictors to DV

    # Make C data
    # Make weighted combo of A and B
    data[:,1] = data[:,1] + data[:,0]*weights[0]
    data[:,2] = data[:,2] + data[:,0]*weights[2] + data[:,1]*weights[1]
    
    return data


def CalculateIndPower(NBoot, NSimMC, N, typeA, alpha, a, b, cP):
    print("Starting simulations...")
    t = time.time()
    # Prepare a matrix for counting significant findings
    PercList = np.zeros((NSimMC,5))
    BCList = np.zeros((NSimMC,5))
    BCaList = np.zeros((NSimMC,5))
    SaList = np.zeros((NSimMC,1))
    SbList = np.zeros((NSimMC,1))
    ScPList = np.zeros((NSimMC,1))
    SIEList  = np.zeros((NSimMC,1))
    
    IEBiasList = np.zeros((NSimMC,1))
    IEBSskewList = np.zeros((NSimMC,1))
    IEBSskewStatList = np.zeros((NSimMC,1))
    
    # Repeatedly generate data for Monte Carlo simulations 
    for i in range(NSimMC):
        print("%d of %d"%(i+1,NSimMC))
        # Make data
        data = MakeIndependentData(N, [1,1,1], [1,1,1], [a, b, cP], typeA)
        # data = MakeMultiVariableData(N,means, covs)
        # Point estimates
        #  Model of B with A, parameter is a
        PointEstimate2 = Calculate_Beta_Sklearn(data[:,[0,1]])
        # Model of C with A and B, parameters are cp, b
        PointEstimate3 = Calculate_Beta_Sklearn(data)
        # Point estimate mediation effects
        IE, TE, DE, Act_a, Act_b = CalculateMediationPEEffect(PointEstimate2, PointEstimate3)
        
        # Bootstrap model 2
        BSbetalist2, BSinterlist2 = Bootstrap_Sklearn(NBoot,Calculate_Beta_Sklearn, data[:,[0,1]])
        JKbetalist2, JKinterlist2 = JackKnife(Calculate_Beta_Sklearn, data[:,[0,1]])
        # Bootstrap model 3
        BSbetalist3, BSinterlist3 = Bootstrap_Sklearn(NBoot,Calculate_Beta_Sklearn, data)
        JKbetalist3, JKinterlist3 = JackKnife(Calculate_Beta_Sklearn, data)
        # Bootstrap mediation effects
        BSIE, BSTE, BSDE, BSa, BSb = CalculateMediationResampleEffect(BSbetalist2, BSbetalist3)
        # Jackknifemediation effects
        JKIE, JKTE, JKDE, JKa, JKb = CalculateMediationResampleEffect(JKbetalist2, JKbetalist3)
        
        # Calculate confidence intervals for all effects in the model
        IEPercCI, IEBCCI, IEBCaCI, IEBiasList[i], IEBSskewList[i], IEBSskewStatList[i] = CalculateCI(BSIE, JKIE, IE, alpha)
        TEPercCI, TEBCCI, TEBCaCI, TEBias, TEBSskew, TEBSskewStat = CalculateCI(BSTE, JKTE, TE, alpha)
        DEPercCI, DEBCCI, DEBCaCI, DEBias, DEBSskew, DEBSskewStat = CalculateCI(BSDE, JKDE, DE, alpha)
        aPercCI, aBCCI, aBCaCI, aBias, aBSskew, aBSskewStat = CalculateCI(BSa, JKa, Act_a, alpha)
        bPercCI, bBCCI, bBCaCI, bBias, bBSskew, bBSskewStat = CalculateCI(BSb, JKb, Act_b, alpha)
        # Make boolean list of whether the confidence intervals include zero        
        PercList[i,:] = DoCIIncludeZero(IEPercCI,TEPercCI, DEPercCI, aPercCI, bPercCI)
        BCList[i,:] = DoCIIncludeZero(IEBCCI,TEBCCI, DEBCCI, aBCCI, bBCCI)
        BCaList[i,:] = DoCIIncludeZero(IEBCaCI,TEBCaCI, DEBCaCI, aBCaCI, bBCaCI)
        # Calculate the effect size for each simulation
        SaList[i] = Calculate_standardizedB(data[:,[0,1]], PointEstimate2[0])[0]
        [ScPList[i],SbList[i]]  = Calculate_standardizedB(data, PointEstimate3[0])
        SIEList[i] = CalculateKappaEffectSize(data, PointEstimate2[0], PointEstimate3[1])[0]
    # Now that simulations have been done, how many simulations
    # resulted in significant results
    PercPower = PercList.sum(0)/NSimMC
    BCPower = BCList.sum(0)/NSimMC
    BCaPower = BCaList.sum(0)/NSimMC
 
    # Prepare output data
    # Add parameters
    outdata = [NBoot, NSimMC, N, a, b, cP, typeA]
    # Add the power estimates from the difference approaches    
    outdata.extend(PercPower)
    outdata.extend(BCPower)
    outdata.extend(BCaPower)
    # Add mean and std of effect sizes
    EffectSizeList = [SaList.mean(), SaList.std(), SbList.mean(), SbList.std(), ScPList.mean(), ScPList.std(), SIEList.mean(), SIEList.std()]
    outdata.extend(EffectSizeList)
   # What are the skew and bias estimates
    BiasSkewList = [IEBiasList.mean(), IEBiasList.std(), IEBSskewList.mean(), IEBSskewList.std(), IEBSskewStatList.mean(), IEBSskewStatList.std()]
    outdata.extend(BiasSkewList)
    
        

    print("Run time was: %0.2f"%(time.time() - t))
    # For each run the following should be calculated also
    # standardized parameters
    # skewnPess in the BS resmple
    # statistical test of whether the skewness is large
    # confidence intervals using 
    # percentile
    # BCa
    return outdata



def PrintPowerSimResult(outdata):
    print("Power for IE with Perc: %0.3f"%(outdata[7]))
    print("Power for IE with BC: %0.3f"%(outdata[12]))
    print("Power for IE with BCa: %0.3f"%(outdata[17]))    
    
    
def DoCIIncludeZero(IE, TE, DE, a, b):
    return int(np.prod(IE)>0), int(np.prod(TE)>0), int(np.prod(DE)>0), int(np.prod(a)>0), int(np.prod(b)>0)

def main():
    #   make sure this script and make submussions match
    if len(sys.argv[1:]) != 8:
        print("ERROR")
    else:
        print("Getting ready")
        # Pass the process ID to use for setting the seed
        pid = os.getpid() 
        # Set the seed
        np.random.seed(pid + int(time.time()))
        # Run the sim
        NBoot = int(sys.argv[1:][0])
        NSim = int(sys.argv[1:][1])
        N = int(sys.argv[1:][2])
        Atype = int(sys.argv[1:][3])
        a = float(sys.argv[1:][4])
        cP = float(sys.argv[1:][5])
        OutDir = sys.argv[1:][6]
        b = float(sys.argv[1:][7])
        bLIST = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # If this parameter is too big then assume that the sbatch array was used
        if b > 0.9:
            b = bLIST[int(b)-1]
            
        
        alpha = 0.05
        print("Calling the simulator")
        outdata = CalculateIndPower(NBoot,NSim, N, Atype, alpha, a, b, cP)
        print("Done with the simulations")
        # Make outputfile name
        clock = time.localtime()
        OutFileName = "SimData_NB_%d_NSim_%d_"%(NBoot,NSim)
        OutFileName = OutFileName+str(clock.tm_hour)+"_"+str(clock.tm_min)+"__"+str(clock.tm_mon)+"_"+str(clock.tm_mday)+"_"+str(clock.tm_year)
        OutFileName = OutFileName+'_pid'+str(pid)+'.csv'
        np.savetxt(os.path.join(OutDir, OutFileName), outdata, delimiter = ',')

def main2():
    index = int(sys.argv[1:][0])
    bLIST = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
    RunEffectSizeSimulations(bLIST[index])
        
if __name__ == "__main__":
# #     #MakeBatchScripts()
    main()
#     main2()

       
