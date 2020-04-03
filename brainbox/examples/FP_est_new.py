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
binSize=0.25 #in ms
b= np.arange(0,10.25,binSize)/1000 + 1e-6 #bins in seconds
bTestIdx = [5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
bTest = [b[i] for i in bTestIdx]

thresh = 0.2
acceptThresh=0.1

baseRates = np.logspace(-0.3,1.3,20)
recDur = 3600

rpvec=[.001,.0015,0.002,0.0025,0.003,0.004,0.005,0.0075,0.01] #true RPs in seconds 
fig, axs = plt.subplots(1,len(rpvec), figsize=(15, 6), facecolor='w', edgecolor='k') #initialize subplots
rpidx=0
for rp in rpvec:
    #print(rpidx)  
    contPct = np.arange(0.01,0.2,0.02) 
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(baseRates)))
    
    passPct = np.zeros([len(baseRates), len(contPct)])
    classicPassPct = np.zeros([len(baseRates), len(contPct)])
    
    start_time=time.time()
    baseidx=0
    for baseRate in baseRates:
        cidx=0
        for c in contPct:
            contRate = baseRate*c
            mfunc =np.vectorize(max_acceptable_cont_2)
            m = mfunc(baseRate,bTest,recDur,baseRate*acceptThresh,thresh)
            #m = mfunc(baseRate,b[1:-1],recDur,baseRate*acceptThresh,thresh)
            simRes = np.zeros([nSim,len(bTest)])
            #simRes = np.zeros([nSim,len(b)-2])
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
                c0 = correlograms(combST,np.zeros(len(combST),dtype='int8'),cluster_ids=[0],bin_size=binSize/1000,sample_rate=20000,window_size=.05,symmetrize=False)               
                #c0 = correlograms(combST,np.zeros(len(combST),dtype='int8'),cluster_ids=[0],bin_size=binSize,sample_rate=20000,window_size=.05,symmetrize=False)
                cumsumc0 = np.cumsum(c0[0,0,:])
                simRes[n,:] = cumsumc0[bTestIdx]
                #simRes[n,:] = np.cumsum(c0[0,0,bTestIdx[1:-1]])
                #simRes[n,:] = np.cumsum(c0[0,0,1:(len(b)-1)])

                len(simRes)
            passPct[baseidx,cidx]=sum(np.any(np.less_equal(simRes[:,0:],m),axis=1))/nSim*100
            classicPassPct[baseidx,cidx] = sum((np.less_equal(simRes[:,4],m[4])))/nSim*100
            cidx+=1
    
        #plt.plot(contPct,passPct[baseidx,:],'o-',color=colors[baseidx],label=str(baseRates[baseidx]))
        
        #print(rpidx)
        axs[rpidx].plot(contPct,passPct[baseidx,:],'o-',color=colors[baseidx],label=str(baseRates[baseidx]))
        #axs2[rpidx].plot(contPct,classicPassPct[baseidx,:],'o-',color=colors[baseidx],label=str(baseRates[baseidx]))
        # plt.plot(contPct,classicPassPct[])
        baseidx+=1
    print(np.mean(passPct[:,:]))    
    axs[rpidx].title.set_text('True RP is {}'.format(np.round(rp,4))) 
    rpidx+=1
    print(time.time() - start_time)

#fig.legend(bbox_to_anchor=(1.04,1), loc="upper left") 
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

# plt.plot(m1,x1,color='b')
# plt.plot(m2,x2,color='r')
# plt.plot(np.arange(0,2500,100),np.arange(0,2500,100),'--',color='k')
# plt.show()