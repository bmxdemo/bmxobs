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
        gk = np.exp(-0.5*((r*r)/self.sigma2).sum(axis=-1))
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
        self.ant_beam = []
        self.ant_beam.append(SingleBeam(center = (0.009,-0.015), sigma=(0.044,0.039), airy=airy, fixAmp=fixAmp))
        self.ant_beam.append(SingleBeam(center = (0.015,-0.02), sigma=(0.043,0.04), airy=airy, fixAmp=fixAmp))
        self.ant_beam.append(SingleBeam(center = (-0.001,-0.009), sigma=(0.036,0.047), airy=airy, fixAmp=fixAmp))
        self.ant_beam.append(SingleBeam(center = (-0.043,0.016), sigma=(0.046,0.033), airy=airy, fixAmp=fixAmp))
        self.ant_beam.append(SingleBeam(center = (0.016,-0.014), sigma=(0.037,0.04), airy=airy, fixAmp=fixAmp))
        self.ant_beam.append(SingleBeam(center = (0.007,-0.028), sigma=(0.055,0.035), airy=airy, fixAmp=fixAmp))
        self.ant_beam.append(SingleBeam(center = (0.007,-0.022), sigma=(0.057,0.036), airy=airy, fixAmp=fixAmp))
        self.ant_beam.append(SingleBeam(center = (-0.04,0.016), sigma=(0.058,0.034), airy=airy, fixAmp=fixAmp))
        self.phi = np.zeros((dataLen, 8))
        self.freq = freq
        print(self.freq)
        #freq in MHz
    """
    def mult_point_source(self, channel, A, track, datNum,z=[0]):
        geometry=[]
        #print(self.freq)
        k=1.38e-23#J/K
        s=len(z)
        for freq in self.freq:
            g=0*self.point_source(freq, channel, A, track[0], datNum)[0]
            for i in range(s):
                r=z[i]
                t=track[i]
                if 1412.9/(1+r)<=freq<=1427.1/(1+r):
                    gauss=np.exp(-(freq-1420/(1+r))**2/(2*(1.42/(1+r))**2))/((1.42/(1+r))*np.sqrt(2*np.pi))
                    l=0.3/freq#m
                    a=self.point_source(freq, channel, A, t, datNum)
                    g+=a[0]*gauss*(l**2)/2/k#/a[1]
                else:
                    g+=0
            geometry.append(g)
        geometry=np.array(geometry)
        return geometry
    """
    def mult_point_source(self, channel, A, track, datNum,z=[0]):
        
        k=1.38e-23#J/K
        geometry=np.zeros((len(self.freq),len(track[0])))
        #print(self.freq)
        s=len(z)
        if channel==11:
            solid=1.133*0.044*0.039#pi/4/ln(2)~1.133
        elif channel==22:
            solid=1.133*0.043*0.04
        elif channel==33:
            solid=1.133*0.036*0.047
        elif channel==44:
            solid=1.133*0.046*0.033
        elif channel==55:
            solid=1.133*0.037*0.04
        elif channel==66:
            solid=1.133*0.055*0.035
        elif channel==77:
            solid=1.133*0.057*0.036
        elif channel==88:
            solid=1.133*0.058*0.034
        else:
            print('error')
        
        #first source then freq
        for i in range(s):
            r=z[i]
            t=track[i]
            #g=np.zeros(len(track[0]))#np.zeros
            gg=[]
            for freq in self.freq:
                if 1412.9/(1+r)<=freq<=1427.1/(1+r):
                    gauss=np.exp(-(freq-1420/(1+r))**2/(2*(1.42/(1+r))**2))/((1.42/(1+r))*np.sqrt(2*np.pi))
                #change to -5sigma to 5sigma
                    l=300/freq#m
                    a=self.point_source(freq, channel, A, t, datNum)
                    g=a*gauss*(l**2)/2/k/np.pi/solid#0.0025#/a[1]
                else:
                    g=np.zeros(len(track[0]))
                gg.append(g)
            gg=np.array(gg)
            geometry=np.add(geometry,gg)
        #geometry=np.array(geometry)
        
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
        return A*fringe*beams#,beams
