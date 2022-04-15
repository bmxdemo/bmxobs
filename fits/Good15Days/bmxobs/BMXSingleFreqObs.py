import numpy as np
import os, pickle
from .bmxobs import BMXObs

class BMXSingleFreqObs (BMXObs):

    def __init__(self, obs_dir, freq_bins = (280,300), time_avg=10):

        BMXObs.__init__(self,obs_dir,None)

        self.freq_bins = freq_bins
        self.time_avg = time_avg

        cache_fn = self.get_cache_fn()
        if os.path.isfile(cache_fn):
            self.data = pickle.load(open(cache_fn,"rb"))
        else:
            self.generate_data()
            if not os.path.exists("cache"):
                os.makedirs("cache")
            pickle.dump(self.data,open(cache_fn,"wb"))
        self.N = len(self.data)
        self.bw = np.abs(self.freq[0][self.freq_bins[0]]-self.freq[0][self.freq_bins[1]])
        self.freq = self.freq[0][self.freq_bins[0]:self.freq_bins[1]].mean()
        self.mjd = self.time_avg_vec(self.mjd)
        self.ra = self.time_avg_vec(self.ra)
        self.dec = self.time_avg_vec(self.dec)
        for i,s in enumerate(self.sat):
            self.sat[i]= np.rec.fromarrays(
                [self.time_avg_vec(s['alt']),self.time_avg_vec(s['az'])],
                dtype=[('alt','f4'),('az','f4')])




    def time_avg_vec(self,s):
        s = s[: (len(s)//self.time_avg)*self.time_avg ]
        s = s.reshape((-1,self.time_avg)).mean(axis=1)
        return s

    def get_cache_fn(self):
        s = self.root
        s = s[s.rfind("_")-6:s.rfind("_")+5]
        fn = 'cache/%s_%i_%i_%i.dat'%(s,
          self.freq_bins[0], self.freq_bins[1], self.time_avg)
        return fn

    def generate_data(self):
        channels = ['%i%i0'%(i+1,j+1) for i in range(4) for j in range(i,4) ]
        channels += ['%i%i0'%(i+1,j+1) for i in range(4,8) for j in range(i,8) ]
        channels= ",".join(channels)
        self._load_data(channels)
        ndata={}
        for channel in self.data.keys():
            nchannel = int(channel)/10
            sdata = self[channel][:,self.freq_bins[0]:self.freq_bins[1]].mean(axis=1)
            sdata = self.time_avg_vec(sdata)
            ndata[nchannel]=sdata

        del self.data
        self.data = ndata
        

