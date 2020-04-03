# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:59:40 2020

@author: Steinmetz Lab User
Testing the new max_acceptable:
    test reject rate for different levels of contamination
"""

import numpy as np
from max_acceptable_isi_viol_2 import max_acceptable_cont_2
from max_acceptable_isi_viol_2 import genST
import cv2
from phylib.stats import correlograms
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from phylib.stats import correlograms

nSim = 100

# binSize = 0.0005
# b = np.arange(1e-6,0.0055,binSize)


b = np.arange(0,.0125,binSize)+1e-6
bTestIdx = [2, 3, 4, 5, 6, 8, 10, 15,  20]
bTest = [b[i] for i in bTestIdx]

thresh = 0.2
acceptThres=0.1

baseRates = np.logspace(-0.3,1.3,20)
recDur = 3600

rpvec=np.arange(0.001,0.005,0.0005)
for rp in rpvec:
    
    contPct = np.arange(0.01,0.2,0.02) 

    
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(baseRates)))
    
    passPct = np.zeros([len(baseRates), len(contPct)])
    #colors = hsv(numel(baseRates));  this needs to be fixed, made into python with cv2. 
    
    start_time=time.time()
    bidx=0
    for baseRate in baseRates:
        cidx=0
        for c in contPct:
            contRate = baseRate*c
            rpidx=np.where(b<rp)[0][-1]+1
            mfunc =np.vectorize(max_acceptable_cont_2)
            m = mfunc(baseRate,bTest,recDur,baseRate*acceptThresh,thresh)

            
            simRes = np.zeros([nSim,len(bTestIdx)])
            for n in range(nSim):
                st = genST(baseRate,recDur)
                isi = np.diff(np.insert(st,0,0)) 
                isi = np.delete(isi,np.where(isi<rp)[0]) 
                st = np.cumsum(isi)
                if c>0:
                    contST = genST(contRate,recDur)
                else:
                    contST=[]
                combST = np.sort(np.concatenate((st, contST)))
                c0 = correlograms(combST,np.zeros(len(combST),dtype='int8'),cluster_ids=[0],bin_size=binSize,sample_rate=20000,window_size=.05,symmetrize=False)
                simRes[n,:] = np.cumsum(c0[0,0,bTest])
                len(simRes)
            passPct[bidx,cidx]=sum(np.any(np.less_equal(simRes[:,0:],m),axis=1))/nSim*100
            cidx+=1
    #        print(time.time() - start_time)
    
        plt.plot(contPct,passPct[bidx,:],'o-',color=colors[bidx],label=str(baseRates[bidx]))
        bidx+=1
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left") 
    plt.show()



#
# = 1:numel(baseRates)
#    baseRate = baseRates(bidx);
#    for c = 1:numel(contPct)
#        
#        contRate = baseRate*contPct(c);
#        
#        m = arrayfun(@(xx)maxAcceptableISIviol2(baseRate, xx, recDur, baseRate/10, thresh), b(2:end-1));
#        
#        simRes = zeros(nSim, numel(b)-1);
#        for n = 1:nSim
#            
#            st = genST(baseRate,recDur);
#            isi = diff([0; st]); isi(isi<rp) = [];
#            st = cumsum(isi);
#            
#            contST = genST(contRate,recDur);
#            
#            combST = sort([st; contST]);
#            
#            [nComb,xACG] = histdiff(combST, combST, b);
#            
#            simRes(n,:) = cumsum(nComb);
#        end
#        
#        passPct(bidx,c) = sum(any(simRes(:,1:end-1)<=repmat(m,nSim,1),2))/nSim*100;
#        
#        
#    end
#    legH(bidx) = plot(contPct, passPct(bidx,:), 'o-', 'Color', colors(bidx,:), 'MarkerFaceColor', colors(bidx,:)); 
#        
#    hold on; drawnow;
#end
#addX(0.1);
#legend(legH, array2stringCell(baseRates));
#set(gcf, 'Color', 'w'); 
#box off; 
#xlabel('True contamination proportion'); 
#ylabel('Percentage of simulations that pass'); 




#
#rate = 5; n = 10000;
#st = genST(rate,n);
#
#binSize = 0.0005;
#b = binSize:binSize:0.005; 
#
#[nACG,xACG] = histdiff(st, st, b);
#% nACG = nACG./binSize; 
#
#figure; 
#stairs(xACG-binSize/2, nACG, 'LineWidth', 2.0);
#box off; 
#ylim([0 max(ylim())])