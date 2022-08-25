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
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

class Cleaning:
    def __init__(self,data=bmxobs.BMXObs("pas/210903_0000/",channels="110"),template=genfromtxt('11fake1.csv', delimiter=','),ch='11',a=0):
        #generating noise components
        self.d=data
        self.template=template
        self.ch=ch
        self.a=a
        
        if self.ch=='11':
            trainpatch=self.d[110][:40000,759:1200]#.reshape(-1, 10, 441).mean(axis = 1)#+self.a*np.real(self.template[:,:4000].T)
        if self.ch=='22':
            trainpatch=self.d[220][:40000,759:1200]#.reshape(-1, 10, 441).mean(axis = 1)
        if self.ch=='44':
            trainpatch=self.d[440][:40000,759:1200]#.reshape(-1, 10, 441).mean(axis = 1)
        if self.ch=='55':
            trainpatch=self.d[550][:40000,759:1200]#.reshape(-1, 10, 441).mean(axis = 1)
        
        #for k in range(3):
         
        trainpatch=self.spikeremoval(trainpatch,train=True)
            
        trainpatch=trainpatch.reshape(-1, 10, 441).mean(axis = 1)
        trainpatch+=self.a*np.real(self.template[:,:4000].T)
        
        ave=self.regularize(trainpatch)
        
        M=np.dot(ave.T,ave)
        w,self.v=np.linalg.eig(M)
        
        
    def cleandata(self):
        #clean up data patch
        if self.ch=='11':
            datapatch=self.d[110][50000:86000,759:1200].reshape(-1, 10, 441).mean(axis = 1)#+self.a*np.real(self.template[:,5000:8600].T)
        if self.ch=='22':
            datapatch=self.d[220][50000:86000,759:1200].reshape(-1, 10, 441).mean(axis = 1)
        if self.ch=='44':
            datapatch=self.d[440][50000:86000,759:1200].reshape(-1, 10, 441).mean(axis = 1)
        if self.ch=='55':
            datapatch=self.d[550][50000:86000,759:1200].reshape(-1, 10, 441).mean(axis = 1)
        #for k in range(3):
        datapatch=self.spikeremoval(datapatch,train=False)
        datapatch+=self.a*np.real(self.template[:,5000:8600].T)
        ave=self.regularize(datapatch)
        aver=self.clean(ave)
        return aver
    
    def cleantemplate(self):
        #clean up template patch
        #freq=self.d.freq[0][759:1200]
        temp=self.template[:,5000:8600].T
        
        ave=self.regularize(temp)
        aver=self.clean(ave)
        return aver
    
    def regularize(self,o):
        m=np.mean(o,axis=0)
        g=np.mean(o,axis=1)
        
        ave=o/np.outer(g,m)*np.mean(o)-1
        #for i in range(441):
         #   ave[:,i]=ave[:,i]/np.std(ave[:,i])
        return ave
    
    def clean(self,a):
        s=[]
            
        for j in range(len(a)):
            p=np.copy(a[j])
            for i in range(30):
                weight=np.dot(p,self.v[:,i])/np.dot(self.v[:,i],self.v[:,i])
                p-=np.real(weight*self.v[:,i])
            s.append(p)
        s=np.array(s)
        c=s.reshape(-1, 10, 441).mean(axis = 1)
        return c
    
    def spikeremoval(self,patch,train=True):
        """
        peaks, _ = find_peaks(np.mean(patch,axis=0),width=(0,5))
        shift=[-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7]
        con=peaks
        for i in shift:
            n=peaks+i
            con=np.concatenate((con, n))
        peaks=con
        for i in peaks:
            for j in range(len(patch)):
                patch[j,i]=np.nan
        for j in range(len(patch)):
            nans, x= np.isnan(patch[j]), lambda z: z.nonzero()[0]
            patch[j][nans]= np.interp(x(nans), x(~nans), patch[j][~nans])

        for j in range(83,441):
            mean=np.mean(patch[:,j])
            std=np.std(patch[:,j])
            for i in range(len(patch)):
                if patch[i,j]>mean+std:
                    patch[i,j]=np.nan
        """
        newpatch = patch.copy()
        for idt in range(patch.shape[0]):
            der = np.divide(np.abs(np.diff(patch[idt,:])),patch[idt,:-1])
            badpixs = np.where(der > 0.01)[0]
            for badpix in badpixs:
                newpatch[idt,badpix-2:badpix+2] = np.median(patch[idt,badpix-6:badpix+6])
        
        bad=[83,84,85,86,87,88,283,284,285,286,287,288,289,290,291,293]
        for i in range(len(newpatch)):
            for j in bad:
                newpatch[i,j]=np.nan
        if train==False:
            lim=8e12
        else:
            lim=9e12
        for i in range(len(newpatch)):
            for j in range(80,441):
                if newpatch[i,j]>lim:
                    newpatch[i,j]=np.nan
        for j in range(len(patch)):
            nans, x= np.isnan(newpatch[j]), lambda z: z.nonzero()[0]
            newpatch[j][nans]= np.interp(x(nans), x(~nans), newpatch[j][~nans])
        #for idt in range(patch.shape[1]):
            #der = np.divide(np.abs(np.diff(patch[:,idt])),patch[:-1,idt])
            #badpixs = np.where(der > 0.01)[0]
            #for badpix in badpixs:
                #newpatch[badpix-2:badpix+2,idt] = np.median(patch[badpix-5:badpix+5,idt])
        if (train):
            for j in range(0,441):
                for i in range(5640,11660):
                    newpatch[i,j]=np.nan
        
        for j in range(0,441):
            nans, x= np.isnan(newpatch[:,j]), lambda z: z.nonzero()[0]
            newpatch[:,j][nans]= np.interp(x(nans), x(~nans), newpatch[:,j][~nans])
            
        return newpatch
        
class SNR:
    def __init__(self,cross=np.load('crossreal.npy',allow_pickle=True),li=[]):
        #calculating SNR map
        self.crossreal=cross
        self.li=li
        crossrealmatprim=[]
        for name in self.li:
    
            crossrealmatprim.append(self.crossreal.item().get(name))
        crossrealmatprim=np.array(crossrealmatprim)
        #deleting outliers
        cut=4
        time=np.arange(0,720,10)
        freq=np.arange(0,442,10)
        variance=[]
        for j in range(720):
            va=[]
            for k in range(442):
                va.append(np.sqrt(np.nanvar(crossrealmatprim[:,j,k])))
            variance.append(va)
        variance=np.array(variance)

    
        for i in range(52):
            for j in time:
                for k in freq:
                    if np.abs(crossrealmatprim[i,j,k])>cut*variance[j,k]:
                        crossrealmatprim[i,j,k]=np.nan
        time=np.arange(0,720,25)
        freq=np.arange(0,442,25)
        variance=[]
        for j in range(720):
            va=[]
            for k in range(442):
                va.append(np.sqrt(np.nanvar(crossrealmatprim[:,j,k])))
            variance.append(va)
        variance=np.array(variance)

    
        for i in range(52):
            for j in time:
                for k in freq:
                    if np.abs(crossrealmatprim[i,j,k])>cut*variance[j,k]:
                        crossrealmatprim[i,j,k]=np.nan
        
        time=np.arange(0,720,12)
        freq=np.arange(0,442,12)
        variance=[]
        for j in range(720):
            va=[]
            for k in range(442):
                va.append(np.sqrt(np.nanvar(crossrealmatprim[:,j,k])))
            variance.append(va)
        variance=np.array(variance)

    
        for i in range(52):
            for j in time:
                for k in freq:
                    if np.abs(crossrealmatprim[i,j,k])>cut*variance[j,k]:
                        crossrealmatprim[i,j,k]=np.nan
    
        crossrealmat=[]
        for i in range(52):
            crossrealmat.append(crossrealmatprim[i])
        crossrealmat=np.array(crossrealmat)
        snrr=np.nanmean(crossrealmat,axis=0)[180:580,100:]/np.sqrt(np.nanvar(crossrealmat,axis=0)[180:580,100:])
        for i in range(400):
            for j in range(342):
                num=52
                for k in range(52):
                    if crossrealmat[k,i+180,j+100]==np.nan:
                        num-=1
                snrr[i,j]=snrr[i,j]*np.sqrt(num)
        self.snr=snrr
        self.mean=np.nanmean(self.snr)
        self.var=np.sqrt(np.nanvar(snrr)/400/342)
        self.detect=self.mean/self.var
