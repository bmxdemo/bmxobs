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
    
                    
        
                    

            
