import numpy as np
import os, pickle
from .bmxobs import BMXObs
from scipy.special import j1

class SingleBeam:
    def __init__ (self, center = (0,0), sigma=(0.05,0.05), smooth=(0.2,0.2)):
        self.center = np.array(center)
        self.sigma2 = np.array(sigma)**2
        self.smooth**2 = np.array(smooth)**2

    def __call__ (self, track):
        def airy(x):
            return 2*j1(x)/x if x!=0 else 1.0
        track = np.atlest_2d(track)
        r = (track-self.center[None,:])
        ra = np.sqrt((r*r/self.sigma2).sum())
        gk = np.exp(-0.5*((r*r)/self.smooth2).sum())
        return airy(ra)*gk


class SingleFreqGeometry:

    def __init__(self,freq=1205.0):
        self.ant_pos = np.array([[0,4],[4,0],[0,-4],[-4,0],
                             [0,4],[4,0],[0,-4],[-4,0]])
        self.ant_beam = [SingleBeam() for i in range(8)]
        self.phi = np.zeros(8)
        self.freq = freq
        self.lambda = 3e8/(self.freq*1e6) # freq in MHz

    def point_source (channel, A, track):
        # track is a Ntimes x 2 size vector in x,y from zenith
        ch1 = channel / 10
        ch2 = channel % 10
        beams = self.ant_beam[ch1](track)*self_ant_beam[ch2](track)
        baseline = self.ant_pos(ch2)-self.ant_pos(ch1)/self.lambda
        if ch1==ch2:
            fringe = 1.0
        else:
            phase = np.sum(track*baseline[None,:])*2*np.pi+self.phi[ch1]-self.phi[ch2]
            fringe = np.exp(1j*phase)
        return A*phase*beams


