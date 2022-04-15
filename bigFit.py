#Format for call:
#python bigFit.py {output filename} {input filename}

import sys
import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
from bmxobs.TheoryPredictor import TheoryPredictor
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import multiprocessing

def getBigFit(Theory):
    detectorSet = [[1,2,3,4],[5,6,7,8]]
    channelSet = [[11,12,13,14,22,23,24,33,34,44],
                  [55,56,57,58,66,67,68,77,78,88]]
    
    paramsOut = {}
    
    TASKS = []
    for detectors,ch in zip(detectorSet,channelSet):
        names = []
        for d in detectors:
            names += ['D{}_beam_center_x'.format(d),
                    'D{}_beam_center_y'.format(d),
                    'D{}_beam_smooth_x'.format(d),
                    'D{}_beam_smooth_y'.format(d)]
            for i,D in enumerate(Theory.data):
                names += ["A{}_{}_{}".format(d,n,i) for n in Theory.satNames]
            if d!= 1 and d!= 5:
                names += ['D{}_pos_x'.format(d),
                          'D{}_pos_y'.format(d)]
                names += ['D{}_phi_{}'.format(d,j) for j in range(len(Theory.data))]
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

if len(sys.argv)>2: # Load starting parameters from file named in 2nd argument
    fileIn = 'bmxobs/fits/' + sys.argv[2]
    f = open(fileIn,'r')
    startData = f.read()
    f.close()
    exec(startData)
    
else: # Default starting parameters without loading
    Data_ids = ['pas/211025_2000']

    startParams = {}

    bins = (580, 600) #Frequency 1258 MHz
    
    zeroSats = []
    
    astroObj={'Cygnus_A': [5.233686582997465, 0.7109409436737796]}

Data = [bmxobs.BMXSingleFreqObs(ids, freq_bins=bins) for ids in Data_ids]

Theory = TheoryPredictor(Data, Geometry = SingleFreqGeometry(len(Data), freq=Data[0].freq), params = startParams, zeroSats=zeroSats, astroObj=astroObj)

print('Begin Big Fit') #Fit Amplitudes
print(getBigFit(Theory))
    
if len(sys.argv)>1: # Save end parameters to file named in 1st argument
    fileOut = 'bmxobs/fits/' + sys.argv[1]
    f = open(fileOut,'w')
    f.write('Data_ids = {}\n\nstartParams = {}\n\nbins = {}\n\nzeroSats = {}\n\nastroObj={}\n'.format(Data_ids, Theory.readParameters(), bins, zeroSats, astroObj))
    f.close()
    