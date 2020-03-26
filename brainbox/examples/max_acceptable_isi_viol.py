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
    max_acceptable = acceptableCont*timeForViol
    
    return max_acceptable