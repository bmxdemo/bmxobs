import numpy as np
import os, pickle
from .bmxobs import BMXObs
from scipy.special import j1

class SingleBeam:
    def __init__ (self, center = (0.,0.), sigma=(0.05,0.05), smooth=(0.05,0.05), airy = False, fixAmp = True):
        self.center = np.array(center)
        self.sigma2 = np.array(sigma)**2
        self.smooth2 = np.array(smooth)**2
        self.airy = airy #Airy vs Gaussian
        self.fixAmp = fixAmp #Sets each peak to have height 1
    
    def __call__ (self, track):
        
        def airy(x):
            return 2*j1(x)/x #if x!=0 else 1
        track = np.atleast_2d(track)
        r = (track-self.center)
        ra = np.sqrt((r*r/self.smooth2).sum(axis=-1))
        gk = np.exp(-0.5*((r*r)/self.smooth2).sum(axis=-1))
        if self.airy:
            beam = airy(ra)
        else:
            beam = gk
        if self.fixAmp:
            return beam/np.maximum(np.max(beam,axis=-1,keepdims=True),1e-20)
        else:
            return beam
        
class MultiFreqGeometry:

    def __init__(self, dataLen, freq=np.arange(1250,1421.25,1.25), airy=False, fixAmp = True, z=1):
        self.ant_pos = np.array([[0.,4.],[4.,0.],[0.,-4.],[-4.,0.],
                             [0.,4.],[4.,0.],[0.,-4.],[-4.,0.]])
        self.ant_beam = [SingleBeam(airy=airy, fixAmp=fixAmp) for i in range(8)]
        self.phi = np.zeros((dataLen, 8))
        self.freq = freq
        print(self.freq)
        #freq in MHz
        
    def mult_point_source(self, channel, A, track, datNum,z=[0]):
        geometry=[]
        #print(self.freq)
        s=len(z)
        for freq in self.freq:
            g=0*self.point_source(freq, channel, A, track[0], datNum)
            for i in range(s):
                r=z[i]
                t=track[i]
                gauss=np.exp(-(freq-1420/(1+r))**2/(2*(1.42/(1+r))**2))/((1.42/(1+r))*np.sqrt(2*np.pi))
                g+=self.point_source(freq, channel, A, t, datNum)*gauss
            geometry.append(g)
        geometry=np.array(geometry)
        return geometry

    def point_source (self, freq, channel, A, track, datNum):
        ch1 = channel // 10 - 1
        ch2 = channel % 10 - 1
        beams = self.ant_beam[ch1](track)*self.ant_beam[ch2](track)
        baseline = (self.ant_pos[ch2]-self.ant_pos[ch1]) * (freq*1e6 / 3e8) #freq*1e6/3e8 = 1/lambda
        if ch1==ch2:
            fringe = 1.0
        else:
            phase = (track*baseline).sum(axis=-1)*2*np.pi+self.phi[datNum][ch1]-self.phi[datNum][ch2]
            fringe = np.exp(1j*phase)
        return A*fringe*beams