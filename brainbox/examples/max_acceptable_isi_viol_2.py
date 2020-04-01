# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:46:34 2020

@author: Steinmetz Lab User
"""
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import brainbox as bb

def max_acceptable_cont_2(FR, RP, rec_duration,acceptableCont, thresh ):
    trueContRate = np.linspace(0,FR,100)
    timeForViol = RP*2*FR*rec_duration
   
    
    expectedCountForAcceptableLimit = acceptableCont*timeForViol
    
    max_acceptable = poisson.ppf(thresh,expectedCountForAcceptableLimit)
    
    if max_acceptable==0 and poisson.pmf(0,expectedCountForAcceptableLimit)>0:
        max_acceptable=-1
    
#    obsContCountVec = np.arange(0,expectedCountAtFullCont,1)
    
#    probUnaccept=np.empty(len(obsContCountVec))
#    obsidx=0max
#    for obsContCount in obsContCountVec:
#        pObs = poisson.pmf(obsContCount,trueContRate*timeForViol)
#        pObsNorm = pObs/sum(pObs)*100
#        probUnaccept[obsidx]=sum(pObsNorm[np.where(trueContRate>acceptableCont)])
#        obsidx +=1
#   
#    
#    if(np.where(probUnaccept/100<thresh)[0].size == 0):
#        max_acceptable = -1 #if there are no places where the probability is below the acc
#    else:     
#        max_acceptable_ind = np.where(probUnaccept/100<(thresh))[0][-1] #changed to thresh*2
#        max_acceptable = 2*obsContCountVec[max_acceptable_ind]
    return max_acceptable


def genST(rate, duration):
#    % function st = genST(rate, duration)
#    %
#    % generates a spike train with specified rate and duration. Rate and
#    % duration should have the same units. 
    
    
    mu = 1/rate
    n = rate*duration 
    isi = np.random.exponential(mu, int(np.ceil(n*2))) # % generate twice as many spikes as likely required
    
    while sum(isi)<duration:
#        % this part will be extremely slow if it needs to be invoked, but in
#        % general it should not be 
        np.append(isi,np.random.exponential(mu))
    
    
    st = np.cumsum(isi)
    st = st[0:np.where(st<duration)[0][-1]]
    
    return st