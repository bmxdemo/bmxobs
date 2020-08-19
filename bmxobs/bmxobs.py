""" BMX observations

This class works on reduced BMX datasets 
and provides a set of utility functions to access and 
manipulate those """

import numpy as np
import os
import os.path as path
import fitsio
import glob
import functools

class BMXObs:
    """This is our main 
    


    """
    root_dir = (os.environ['BMX_DATA'] if "BMX_DATA" in os.environ
                else "/gpfs02/astro/workarea/bmxdata/reduced/")

    def __init__ (self, obs_dir, channels=None):
        """Creates a basic object for manipulation of reduced BMX
           data

        Parameters:
        ----------
        obs_dir : str
          Directory from which to pull data
        channels : str
          Channels to load in string, either: comma-separated list of 
          channels descriptions, "all", or "all_auto"
          

        Note:
        -----
        One is not suppoed to load the data after the object has been created.
        This is because various operations like averaging will modify data in 
        place and we cannot ensure consistency otherwise
        

        """
        self.root = path.join(BMXObs.root_dir, obs_dir)
        self._load_aux()
        if channels:
            self._load_data(channels)
        self.N = len(self.mjd) ## all of the same size
        self.diode_buf_samp = 1

    def _load_aux(self):
        """ Loads auxiliary data"""
        def readtxt(fn):
            lines = open(path.join(self.root,fn)).read().splitlines() #returns without \n
            return lines

        def readfits (fn):
            return fitsio.read(path.join(self.root,fn))
        
        self.meta = set(readtxt('meta'))
        self.stage = int(readtxt('stage')[0])
        self.tags = readtxt('tags')
        self.wires = {}
        self.iwires = {}
        for i,n in enumerate(readtxt('wires')):
            self.wires[n] = i+1
            self.iwires[i+1] = n
        self.mjd = readfits('mjd.fits')['mjd']
        self.diode = readfits ('diode.fits')
        coords = readfits ('coords.fits')
        self.ra = coords['ra']
        self.dec = coords['dec']
        self.lgal = coords['lgal']
        self.bgal = coords['bgal']
        ## determine number of cuts
        self.ncuts = len(glob.glob(path.join(self.root,"cut*")))
        self.nchan = 8 ## let's fix this for now
        self.freq = [readfits("cut%i/freq.fits"%i) for i in range(self.ncuts)]
        for i,freq in enumerate(self.freq):
            setattr(self,"freq%i"%i, freq)
        
        #New variables added for Jesse O. remove_spikes functions
        self.spikeStart = []
        self.spikeEnd = []
        self.spikeWidth = []
        for i in range(self.nchan):
            self.spikeStart.append([])
            self.spikeEnd.append([])
            self.spikeWidth.append([])
        
    def _load_data(self,channels):
        if channels == 'all':
            chlist = ['%i%i%i'%(i+1,j+1,c) for i in range(4) for j in range(i,4) for c in range(self.ncuts)]
            chlist += ['%i%i%i'%(i+1,j+1,c) for i in range(4,8) for j in range(i,8) for c in range(self.ncuts)]
        elif channels == 'all_auto':
            chlist = ['%i%i%i'%(i+1,i+1,c) for i in range(8) for c in range(self.ncuts)]
        else:
            chlist = [s.strip() for s in channels.split(",")]

        self.data = {}
        for entry in chlist:
            i=int(entry[0])
            j=int(entry[1])
            cut = int(entry[2:])
            ientry = int(entry)
            auto = (i==j)
            fname = ("cut%i/auto_%i.fits"%(cut,i) if auto else
                     "cut%i/cross_%i%i.fits"%(cut,i,j))
            if auto:
                data = fitsio.read(path.join(self.root,fname))
                self.data[ientry] = data
            else:
                datareal = fitsio.read(path.join(self.root,fname))
                dataimag = fitsio.read(path.join(self.root,fname),ext=1)
                self.data[ientry] = datareal + 1j *dataimag
        for name,data in self.data.items():
            setattr(self,"d"+str(name),data)
            
    def __getitem__ (self, d):
        return self.data[d]

    def __len__(self):
        return self.N

    def cache_clear(self):
        self.diode_pulses.cache_clear()
        del self.diode_r
        
    @functools.lru_cache()
    def diode_pulses (self):
        donoff=[]
        i=0
        while (self.diode[i]>0):
            i+=1
        while True:
            while (self.diode[i]==0):
                i+=1
                if i==self.N: break
            if i==self.N: break
            st=i
            while (self.diode[i]>0):
                i+=1
                if i==self.N: break
            if i==self.N: break
            donoff.append((st,i))
        donoff=donoff[1:-1] ## skip first and last one
        return donoff


    
    def process_diode(self):
        buf_samp = self.diode_buf_samp
        if hasattr(self,"diode_r"):
            return
        donoff = self.diode_pulses()
        self.diode_r = {}
        for cut in range(self.ncuts):
            for chan in range(0,self.nchan):
                id = int("%i%i%i"%(chan+1,chan+1,cut))
                if id in self.data:
                    da = self.data[id]
                    di = []
                    for i,j in donoff:
                        h=(j-i)
                        a=i-h ## we skip one transient one
                        b=j+h+buf_samp
                        diff=da[i+buf_samp:j].mean(axis=0)-0.5*(da[a:i].mean(axis=0)+da[j+buf_samp:b].mean(axis=0))
                        di.append(diff)
                    di=np.array(di)
                    self.diode_r [id] = di

    def remove_diode_pulses(self, to_self=True):
        self.process_diode()
        buf_samp=self.diode_buf_samp
        donoff = self.diode_pulses()
        outdata = {}
        for cut in range(self.ncuts):
            for chan in range(0,self.nchan):
                id = int("%i%i%i"%(chan+1,chan+1,cut))
                if id in self.data:
                    da = np.copy(self.data[id])
                    gain = self.diode_r[id].mean(axis=0)
                    for (i,j),val in zip(donoff,self.diode_r[id]):
                        da[i+buf_samp:j,:]-=val
                        for k in range(buf_samp):
                            da[i+k,:] = 0.5*(da[i-1,:]+da[i+buf_samp,:])
                            da[j+k,:] = 0.5*(da[j-1,:]+da[j+buf_samp,:])
                    if to_self:
                        self.data[id]= da
                    else:
                        outdata[id] = da

        if to_self:
            return self.data
        else:
            return outdata
    
    #Function to remove narrow spikes from data set
    #This version runs each d.data[xxy] plot and removes any
    #spikes that it finds, prepping the d.data[xxy].mean(axis=0) plot
    def remove_narrow_spikes_V1(self, *args, to_self=True):
        outdata = {}
        chan = args[0]
        print("Channel input = %i" % chan)
        cut = args[1]
        print("Cut input = %i" % cut)
        id = int("%i%i%i"%(chan,chan,cut))
        if id in self.data:
            da = np.copy(self.data[id])
            for j in range(self.N):
                if (j % 10000) == 0:
                    print("Program is %f percent complete" % (100*j/self.N))
                thresholdStart = 1.01*da[j][0]
                threshold = 0
                narrowSpikesStart = []
                narrowSpikesEnd = []
                narrowSpikesWidth = []
                dataSlope = (da[j][len(da[j])-1] - da[j][0]) / ((len(da[j])-1) - 0)
                for i in range(len(da[j])):
                    #Adjust for slanted baseline value of data
                    threshold = thresholdStart + (i*dataSlope)
                    if i == 0:
                        if (da[j][i] >= threshold):
                            narrowSpikesStart.append(i)
                    if i>=1:
                        #Previous data points (i-1) must be compared to previous threshold (threshold-1*dataSlope)
                        if ((da[j][i] >= threshold) & (da[j][i-1] < (threshold)-1*dataSlope)):
                            narrowSpikesStart.append(i)
                        if ((da[j][i] < threshold) & (da[j][i-1] >= (threshold-1*dataSlope))):
                            narrowSpikesEnd.append(i)
                for i in range(len(narrowSpikesEnd)):
                    narrowSpikesWidth.append(narrowSpikesEnd[i] - narrowSpikesStart[i])
                    if (narrowSpikesEnd[i] - narrowSpikesStart[i]) < 8:
                        for k in range(narrowSpikesStart[i], narrowSpikesEnd[i]):
                            da[j][k] = da[j][0] + k*dataSlope
                if to_self:
                    self.spikeStart[chan-1].append(narrowSpikesStart)
                    self.spikeEnd[chan-1].append(narrowSpikesEnd)
                    self.spikeWidth[chan-1].append(narrowSpikesWidth)
            if to_self:
                self.data[id] = da
            else:
                outdata[id] = da

        if to_self:
            return self.data
        else:
            return outdata


    #Function to remove narrow spikes from data set
    #This version runs over d.data[xxy].mean(axis=0) plots to find spikes, 
    #and then goes back to remove them from the d.data[xxy] plots
    def remove_narrow_spikes_V4(self, *args, to_self=True):
        outdata = {}
        chan = args[0]
        print("Channel input = %i" % chan)
        cut = args[1]
        print("Cut input = %i" % cut)
        id = int("%i%i%i"%(chan,chan,cut))
        if id in self.data:
            da = np.copy(self.data[id].mean(axis=0))
            thresholdStart = 1.03*da[0]
            threshold = 0
            narrowSpikesStart = []
            narrowSpikesEnd = []
            narrowSpikesWidth = []
            dataSlope = (da[len(da)-1] - da[0]) / ((len(da)-1) - 0)
            for i in range(len(da)):
               #Adjust for slanted baseline value of data
                threshold = thresholdStart + (i*dataSlope)
                #print("threshold = %f" % threshold)
                if i == 0:
                    if (da[i] >= threshold):
                        narrowSpikesStart.append(i)
                if i>=1:
                    #Previous data points (i-1) must be compared to previous threshold (threshold-1*dataSlope)
                    if ( (da[i] >= threshold) & (da[i-1] < (threshold-1*dataSlope)) ):
                        narrowSpikesStart.append(i)
                    if ( (da[i] < threshold) & (da[i-1] >= (threshold-1*dataSlope)) ):
                        narrowSpikesEnd.append(i)
            for i in range(len(narrowSpikesEnd)):
                narrowSpikesWidth.append(narrowSpikesEnd[i] - narrowSpikesStart[i])
            da2 = np.copy(self.data[id])      
            for j in range(self.N):
                #if (j % 10000) == 0:
                    #print("Program is %f percent complete" % (100*j/self.N))
                dataSlope2 = (da2[j][len(da2[j])-1] - da2[j][0]) / ((len(da2[j])-1) - 0)   
                for i in range(len(narrowSpikesEnd)):
                    if (narrowSpikesEnd[i] - narrowSpikesStart[i]) < 8:
                        for k in range(narrowSpikesStart[i], narrowSpikesEnd[i]):
                            da2[j][k] = np.nan
            if to_self:
                self.spikeStart[chan-1].append(narrowSpikesStart)
                self.spikeEnd[chan-1].append(narrowSpikesEnd)
                self.spikeWidth[chan-1].append(narrowSpikesWidth)                
            if to_self:
                self.data[id] = da2
            else:
                outdata[id] = da2

        if to_self:
            self.data
        else:
            return outdata   
            
            
