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
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.time import Time
import astropy.units as u
import csv
import scipy
from numpy import genfromtxt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import signal
from sklearn.decomposition import FastICA
from scipy import interpolate

class Galaclean:
    """
    This code cleans up the satellite emissions and remove gain from all the fine patches in auto-correlation map of BMX.
    satellite emissions: satcen
    gain: self.cen
    return: (fine auto patch-satcen)./self.cen
    
    input: date: string in form of 'pas/some bmx date code'
           channel: two digit channel, each digit can go from 1-4 or 5-8
    """
    def __init__(self,date,channel):
        ch1=int(channel*10)
        ch2=int(channel*10+1)
        data=bmxobs.BMXObs(date,channels=str(ch1))
        self.coarse=np.copy(data[ch1])
        data2=bmxobs.BMXObs(date,channels=str(ch2))
        self.fine=np.copy(data2[ch2])
        self.sat=np.copy(data[ch1][:,269:749])
        self.freqtot=np.array([np.mean(data.freq[1][:15]),np.mean(data.freq[1][15:30]),np.mean(data.freq[1][-30:-15]),np.mean(data.freq[1][-15:])])
        self.freqtotlong=np.concatenate((data.freq[1][:30],data.freq[1][-30:]))
        self.freqcen=data.freq[1]
        
        p=[]
        for i in range(len(self.sat[0])):
            peaks, _ = find_peaks(self.sat[:,i], prominence=10*np.mean(self.sat[:,i]))#3e13)
            for x in peaks:
                p.append(x)
        
        p.sort()
        p=np.array(p)*0.001
        p=p.astype(int)
        p.sort()
        p=np.array(p[1:])*1000
        p=[*set(p)]
        p.sort()
        p.append(85000)
        self.p=p
        tot=[]
        b,a = signal.butter(1, 0.01)
        self.sattot=[]
        fitdown=[]
        fitup=[]
        down=self.spikeremove(np.mean(data2[ch2][:,:15],axis=1))
        tot.append(down)
        for i in range(15):
            test=scipy.signal.filtfilt(b, a,self.fine[:,i])-down
            fitdown.append(test)
        fitdown=np.mean(fitdown,axis=0)
        self.sattot.append(fitdown)
        down=self.spikeremove(np.mean(data2[ch2][:,15:30],axis=1))
        tot.append(down)
        fitdown=[]
        for i in range(15,30):
            
            test=scipy.signal.filtfilt(b, a,self.fine[:,i])-down
            fitdown.append(test)
        fitdown=np.mean(fitdown,axis=0)
        self.sattot.append(fitdown)
        up=self.spikeremove(np.mean(data2[ch2][:,-30:-15],axis=1))
        tot.append(up)
        for i in range(15):
            test=scipy.signal.filtfilt(b, a,self.fine[:,-30+i])-up
            fitup.append(test)
        fitup=np.mean(fitup,axis=0)
        self.sattot.append(fitup)
 
        up=self.spikeremove(np.mean(data2[ch2][:,-15:],axis=1))
        tot.append(up)
        fitup=[]
        for i in range(15,30):
            test=scipy.signal.filtfilt(b, a,self.fine[:,-30+i])-up
            fitup.append(test)
        fitup=np.mean(fitup,axis=0)
        self.sattot.append(fitup)
        #average them
        #self.down=down
        #self.up=up
        
        
        
        tot=np.array(tot).T
        cen=self.twoDfit(tot,self.freqtot,self.freqcen)
        self.cen=cen
        




        self.sattot=np.array(self.sattot).T
        satcen=self.twoDfit(self.sattot,self.freqtot,self.freqcen)
        self.clean=(self.fine-satcen)/cen
        
        
    def isinsat(self,peak):
        s=False
        for x in self.p:
            s=s or (abs(x-peak)<2000)
        #200 pixels is 167s
        return s
    def satfit(self,test):
        fitt=np.zeros(len(test))
        x=np.linspace(0,len(test)-1,len(test))
        peaks, _ = find_peaks(test,width=300)
        results_half = scipy.signal.peak_widths(test, peaks)
        width=results_half[0]
        height=results_half[1]
        for i in range(len(peaks)):
            if self.isinsat(peaks[i]):
            #print(i)
                def func(x, a, b):
                    return a * np.exp(-b * (x-peaks[i])**2)
                xdata=np.arange(peaks[i]-500,peaks[i]+500)
                ydata=test[peaks[i]-500:peaks[i]+500]
                if len(ydata)>2:
                    popt, pcov = curve_fit(func, xdata, ydata,p0 = [height[i], 1/(2*np.pi*500**2)], maxfev=5000)
                    h=popt[0]
                    if abs(peaks[i]-85000)<2000:
                        h=h/1.5
                    w=abs(popt[1])
                    fitt +=h*np.exp(-(x-peaks[i])**2*w)
        return fitt
        
    def spikeremove(self,patch):
        #new=[]
        
        b,a = signal.butter(1, 0.01)
        filtered=scipy.signal.filtfilt(b, a,patch)
        #filtered=np.array(filtered).T
        test=filtered
        peaks, _ = find_peaks(test,width=(300,5000))
        results_half = scipy.signal.peak_widths(test, peaks)
        width=results_half[0]
        rm=[]
        for j in range(len(peaks)):
            if self.isinsat(peaks[j]):
                for k in range(int(width[j])):
                    if peaks[j]+k<len(test):
                        rm.append(int(peaks[j]+k))
                    if peaks[j]-k>0:
                        rm.append(int(peaks[j]-k))
        

        test[rm]=np.nan
        nans, x= np.isnan(test), lambda z: z.nonzero()[0]
        test[nans]= np.interp(x(nans), x(~nans), test[~nans])
            
            #peaks, _ = find_peaks(test,prominence=1e10)
            #results_half = scipy.signal.peak_widths(test, peaks)
            #width=results_half[0]
            
        peaks, _ = find_peaks(test, prominence=1e10)
        rm2=[]
        for j in range(len(peaks)):
            for k in range(1000):
                if peaks[j]+k<80000:
                    rm2.append(int(peaks[j]+k))
                if peaks[j]-k>0:
                    rm2.append(int(peaks[j]-k))
        test[rm2]=np.nan
        nans, x= np.isnan(test), lambda z: z.nonzero()[0]
        test[nans]= np.interp(x(nans), x(~nans), test[~nans])
            
        return test
        
    def twoDfit(self,mat,freq,freqnew):
        matcen=[]
        for i in range(len(mat)):
            f = interpolate.interp1d(freq,mat[i],fill_value="extrapolate")
            matcen.append(f(freqnew))
        return np.array(matcen)

    def fitpeak(self,patch):
        fitted=[]
        x=np.linspace(0,len(patch)-1,len(patch))
        for i in range(len(patch[0])):
            test=patch[:,i]/np.mean(patch[:,i])
            peaks, _ = find_peaks(test, prominence=np.max(test)/12)
            results_half = scipy.signal.peak_widths(test, peaks)
            width=results_half[0]
            height=results_half[1]
            fit=np.zeros(len(patch))
        
            for i in range(len(peaks)):
                fit +=height[i]*np.exp(-(x-peaks[i])**2/np.sqrt(np.pi*width[i]))
            fitted.append(fit)
        return np.array(fitted).T