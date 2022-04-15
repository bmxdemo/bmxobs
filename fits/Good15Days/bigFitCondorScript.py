#!/usr/bin/env /cvmfs/astro.sdcc.bnl.gov/SL73/packages/bacon/2021.12/bin/python
#Used with condorfile.job. 15 good days
#Cut is [].
#Format for call:
#python bigFit.py {input filename}

import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
from bmxobs.TheoryPredictor import TheoryPredictor
import fitsio
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.special import j1
from scipy.optimize import least_squares
import copy
from numba import jit
import multiprocessing
import time
import sys


def getBigFit(Theory):
    detectorSet = [[1,2,3,4],[5,6,7,8]]
    channelSet = [[11,12,13,14,22,23,24,33,34,44],
                  [55,56,57,58,66,67,68,77,78,88]]
    
    paramsOut = {}
    
    TASKS = []
    for detectors,ch in zip(detectorSet,channelSet):
        names = []
        for d in detectors:
            names += [#'D{}_pos_x'.format(d),
                       #'D{}_pos_y'.format(d),
                       'D{}_beam_center_x'.format(d), #Varies beam center x and y
                       'D{}_beam_center_y'.format(d),
                       'D{}_beam_sigma_x'.format(d),
                       'D{}_beam_sigma_y'.format(d),
                    #'D{}_beam_smooth_x'.format(d),
                    #'D{}_beam_smooth_y'.format(d)
                     ]
            
            for i,D in enumerate(Theory.data):
                names += ["A{}_{}_{}".format(d,n,i) for n in Theory.satNames]
            #if d!= 1 and d!= 5:
            #    names += ['D{}_pos_x'.format(d),
            #              'D{}_pos_y'.format(d)]
            #    names += ['D{}_phi_{}'.format(d,j) for j in range(len(Theory.data))]
        for c in ch:
            if c%11 == 0:
                names += ['CH{}_offset_r{}'.format(c,i) for i in range(len(Theory.data))]
        TASKS.append((names, 'all', ch, list(range(len(Theory.data))), [0,-1]))
        
            
    with multiprocessing.Pool(len(TASKS)) as pool:
        imap_it = pool.imap(Theory.fit_parallel, TASKS)
        paramsOut = {}
        
        print('Ordered results using pool.imap():')
        for i,x in enumerate(imap_it):
            for n,p in zip(TASKS[i][0],x):
                paramsOut[n] = p
                
    Theory.setParameters(paramsOut)
    
    return paramsOut
        
astroObj= {}

print("check1")

if len(sys.argv)>2: # Load starting parameters from file named in 1st argument
    fileIn = sys.argv[1]
    f = open(fileIn,'r')
    startData = f.read()
    f.close()
    exec(startData)
    
else: # Default starting parameters without loading
    Data_ids = ['pas/'+ sys.argv[1]]

    startParams = {}

    bins = (580, 600) #Frequency 1258 MHz
    
    zeroSats = []
    
    astroObj={'Cygnus_A': [5.233686582997465, 0.7109409436737796]}

Data = [bmxobs.BMXSingleFreqObs(ids, freq_bins=bins) for ids in Data_ids]

Theory = TheoryPredictor(Data, Geometry = SingleFreqGeometry(len(Data), freq=Data[0].freq), params = startParams, zeroSats=zeroSats, astroObj=astroObj)

print('Begin Big Fit') #Fit Amplitudes
print(getBigFit(Theory))
    
if len(sys.argv)>1: # Save end parameters to file named in 2nd argument
    fileOut = '15Refit/' + sys.argv[2]
    f = open(fileOut,'w+')
    f.write('Data_ids = {}\n\nstartParams = {}\n\nbins = {}\n\nzeroSats = {}\n\nastroObj={}\n'.format(Data_ids, Theory.readParameters(), bins, zeroSats, astroObj))
    #startdata=f.read()
    f.close()

print("Write and read has been completed.")

#Insert fit code

fileIn=fileOut
f = open(fileIn,'r')
startData = f.read()
f.close()

exec(startData)
Data = []
for ids in Data_ids:
    print(ids)
    Data.append(bmxobs.BMXSingleFreqObs(ids, freq_bins=bins))
Theory = TheoryPredictor(Data, Geometry = SingleFreqGeometry(len(Data), freq=Data[0].freq), params = startParams, zeroSats=zeroSats, astroObj=astroObj, thresh=0.03)

#Graphs theory predictions vs data

cut = []
channels = [11,12,13,14,22,23,24,33,34,44,55,56,57,58,66,67,68,77,78,88]
mode = 'all'

print("check2")
#Insert code to extract array values

fitdatearraydat, fitdatearrayfit=Theory.showFit(channels = channels, cut=cut,mode=mode, perSat=False)
print("check3")
#print(fitdatearraydat)
#print(fitdatearrayfit)

print("check4")

fitdatearraydat=fitdatearraydat.astype('int')
fitdatearrayfit=fitdatearrayfit.astype('int')
fitdatearraydat = np.array(fitdatearraydat, dtype=np.float64)
fitdatearrayfit = np.array(fitdatearrayfit, dtype=np.float64)

#Insert code to perform chi squared
#for i in range(len(fit1210arraydat)):
#    a,p=stats.chisquare(fit1210arraydat[i],fit1210arrayfit[i])
#    print(a)
#    print(stats.chisquare(fit1210arraydat[i],fit1210arrayfit[i]))

#fileOut = 'ChiSquared/ChiSquaredout.' + sys.argv[1] + '.txt'
#f = open(fileOut,'w')
#for i in range(len(fitdatearraydat)):
#    a,p=stats.chisquare(fitdatearraydat[i],fitdatearrayfit[i])
#    print(a)
#    a=str(a)
#    f.write(a + "\n")
#f.close()
    
fileOut = 'ChiSquared/Mean15RefitcentersigmaChiSquaredout.' + sys.argv[1] + '.txt'
f = open(fileOut,'w')
chitotal=0
chifinal=0
for i in range(len(fitdatearraydat)):
    for j in range(len(fitdatearraydat[i])):
        #a,p=stats.chisquare(fit1210arraydat[i],fit1210arrayfit[i])
        #print(a)
        #a=str(a)
        #f.write(a + "\n")
        chitotal=chitotal+((fitdatearraydat[i][j]-fitdatearrayfit[i][j])**2)
    chifinal=chitotal/len(fitdatearraydat[i])
    print(chifinal)
    chifinal=str(chifinal)
    f.write(chifinal + "\n")
    chitotal=0
    chifinal=0
f.close()   

print("check5")

#fileOut = 'ChiSquared/14 Days/Median4DivChiSquaredout.' + sys.argv[1] + '.txt'
#f = open(fileOut,'w')
#chitotal=0
#chifinal=0
#for i in range(len(fitdatearraydat)):
#    for j in range(len(fitdatearraydat[i])):
#        #a,p=stats.chisquare(fit1210arraydat[i],fit1210arrayfit[i])
#        #print(a)
#        #a=str(a)
#        #f.write(a + "\n")
#        chitotal=chitotal+(((fitdatearraydat[i][j]-fitdatearrayfit[i][j])**2)/(fitdatearraydat[i][j])**2)
#    chifinal=chitotal/len(fitdatearraydat[i])
#    print(chifinal)
#    chifinal=str(chifinal)
#    f.write(chifinal + "\n")
#    chitotal=0
#    chifinal=0
#f.close() 




