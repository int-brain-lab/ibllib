# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:46:34 2020

@author: Steinmetz Lab User
"""
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import brainbox as bb

def max_acceptable_cont(FR, RP, rec_duration,acceptableCont, thresh ):
    trueContRate = np.linspace(0,FR,100)
    timeForViol = RP*2*(FR-acceptableCont)*rec_duration
   
    
    expectedCountAtFullCont = FR*timeForViol
    obsContCountVec = np.arange(0,expectedCountAtFullCont,1)
    
    probUnaccept=np.empty(len(obsContCountVec))
    obsidx=0
    for obsContCount in obsContCountVec:
        pObs = poisson.pmf(obsContCount,trueContRate*timeForViol)
        pObsNorm = pObs/sum(pObs)*100
        probUnaccept[obsidx]=sum(pObsNorm[np.where(trueContRate>acceptableCont)])
        obsidx +=1
   
    
    if(np.where(probUnaccept/100<thresh)[0].size == 0):
        max_acceptable = -1 #if there are no places where the probability is below the acc
    else:     
        max_acceptable_ind = np.where(probUnaccept/100<(thresh))[0][-1] #changed to thresh*2
        max_acceptable = 2*obsContCountVec[max_acceptable_ind]
    return max_acceptable