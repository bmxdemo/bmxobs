#Format for call:
#python ampFit.py {output filename} {input filename}

import sys
import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
from bmxobs.TheoryPredictor import TheoryPredictor
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import multiprocessing

def getAmpFit(Theory):
    detectorSet = [[1,2,3,4],[5,6,7,8]]
    channelSet = [[11,12,13,14,22,23,24,33,34,44],
                  [55,56,57,58,66,67,68,77,78,88]]
    offsetReal = [[11,12,13,14,22,23,24,33,34,44],
                  [55,56,57,58,66,67,68,77,78,88]]
    offsetImag = [[12,13,14,23,24,34],
                  [56,57,58,67,68,78]]
    
    paramsOut = {}
    
    cuts,sats = Theory.findCuts()
    print(cuts)
    print(sats)
    
    TASKS = []
    for i,detectors in enumerate(detectorSet):
        ch = channelSet[i]
        for j in range(len(Theory.data)):
            for k,cut in enumerate(cuts[j]):
                names = []
                for d in detectors:
                    names += ["A{}_{}_{}".format(d,n,j) for n in sats[j][k]]
                TASKS.append((names, 'all', ch, [j], cut))
            
    with multiprocessing.Pool(min(len(TASKS),50)) as pool:
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

print('Begin Amp Fit') #Fit Amplitudes
print(getAmpFit(Theory))
    
if len(sys.argv)>1: # Save end parameters to file named in 1st argument
    fileOut = 'bmxobs/fits/' + sys.argv[1]
    f = open(fileOut,'w')
    f.write('Data_ids = {}\n\nstartParams = {}\n\nbins = {}\n\nzeroSats = {}\n\nastroObj={}\n'.format(Data_ids, Theory.readParameters(), bins, zeroSats, astroObj))
    f.close()
    