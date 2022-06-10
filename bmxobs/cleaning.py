import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
from bmxobs.MultiFreqGeometry import MultiFreqGeometry
from bmxobs.TheoryPredictor import TheoryPredictor
from sklearn.decomposition import PCA
import fitsio
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1
from scipy.optimize import least_squares
import copy
from numba import jit
import multiprocessing
import time
from fitsio import FITS,FITSHDR
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.time import Time
import astropy.units as u
import csv
from numpy import genfromtxt
from scipy.interpolate import interp1d

class Cleaning:
    def __init__(self,data=bmxobs.BMXObs("pas/210903_0000/",channels="110"),template=genfromtxt('11fake1.csv', delimiter=','),ch='11'):
        self.d=data
        self.template=template
        self.ch=ch
        if self.ch=='11':
            trainpatch=self.d[110][20000:40000,759:1200]
        if self.ch=='22':
            trainpatch=self.d[220][20000:40000,759:1200]
        if self.ch=='44':
            trainpatch=self.d[440][20000:40000,759:1200]
        if self.ch=='55':
            trainpatch=self.d[550][20000:40000,759:1200]
        m=np.mean(trainpatch,axis=0)
        ave=trainpatch/m-1
        M=np.dot(ave.T,ave)
        w,self.v=np.linalg.eig(M)
        o=[]
        for j in range(20000):
            f=np.copy(trainpatch[j])
            f/=m
            f-=1
            for i in range(30):
                weight=np.dot(f,self.v[:,i])/np.dot(self.v[:,i],self.v[:,i])
                f-=weight*self.v[:,i]
            o.append(f)
        self.cleantrain=np.array(o)
            
    def cleandata(self):
        if self.ch=='11':
            datapatch=self.d[110][50003:86003,759:1200]
        if self.ch=='22':
            datapatch=self.d[220][50003:86003,759:1200]
        if self.ch=='44':
            datapatch=self.d[440][50003:86003,759:1200]
        if self.ch=='55':
            datapatch=self.d[550][50003:86003,759:1200]
        
        m=np.mean(datapatch,axis=0)
        s=[]
        for j in range(36000):
            f=np.copy(datapatch[j])
            f/=m
            f-=1
            for i in range(30):
                weight=np.dot(f,self.v[:,i])/np.dot(self.v[:,i],self.v[:,i])
                f-=weight*self.v[:,i]
            s.append(f)
        s=np.array(s)
        ave=s.reshape(-1, 100, 441).mean(axis = 1)
        return ave
    
    def cleantemplate(self):
        #freq=self.d.freq[0][759:1200]
        temp=self.template[:,5000:8600].T
        #temp=temp[43:].T
        m=np.mean(temp,axis=0)
        """
        nu=np.arange(1305,1421.51,1.25)
        x=np.array([])
        for i in range(30):
            f=interp1d(freq, self.v[:,i])
            fitted=f(nu)
            x=np.append(x,fitted)
        x=np.reshape(x, (30, 94))
        """
        s=[]

        for j in range(3600):
            p=np.copy(temp[j])
            p/=m
            p-=1
            for i in range(30):
                weight=np.dot(p,self.v[:,i])/np.dot(self.v[:,i],self.v[:,i])
                p-=weight*self.v[:,i]
            s.append(p)
        s=np.array(s)
        ave=s.reshape(-1, 10, 441).mean(axis = 1)
        return ave