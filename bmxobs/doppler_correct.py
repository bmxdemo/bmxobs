import os
import bmxobs as bmx
import matplotlib.pyplot as plt 
import numpy as np
import math
import statistics as stats
from bmxobs.vlsr import vlsr 

def correct_doppler(patch, sub_div,freq,ra,mjd):
    
    #Define function to calculate doppler factors
    '''
    input: patch: 2D temperature map: freq x ra in axis
    sub_div: how many sub division you want in each data point
    freq: frequency array
    ra: ra array
    mjd: mjd array
    freq ra and mjd need to be from the date the map is taken
    '''
    def doppler_fact(rel_vel):
        beta = rel_vel/299792.458
        nans, x= np.isnan(beta), lambda z: z.nonzero()[0]
        beta[nans]= np.interp(x(nans), x(~nans), beta[~nans])
        return np.nan_to_num(np.sqrt((1+beta)/(1-beta)))
    
    def dopplershift(patch, subdiv, shift):
        doppshifted=np.zeros((len(patch),len(patch[0])*subdiv))
        shift=shift*subdiv
        extend=[]
        for i in range(len(patch)):
            extendline=[]
            for x in patch[i]:
                for i in range(subdiv):
                    extendline.append(x)
            extend.append(extendline)
        extend=np.array(extend)
        for i in range(len(patch)):
            for j in range(len(extend[i])):
                if (j+int(shift[i])>=0) and (j+int(shift[i])<len(extend[i])):
                    doppshifted[i,j+int(shift[i])]=extend[i,j]
        #print('yes')
        return doppshifted

    
    #Extract velocity data
    vel = vlsr(ra*180/np.pi, 40.8, mjd)
    v_rad = vel[0]+vel[1]+vel[2]

    #Calculate the doppler factors for all times
    doppler_facts = doppler_fact(v_rad)
    shift=-(doppler_facts-1)*np.mean(freq)/(freq[1]-freq[0])
    return dopplershift(patch, sub_div, shift)