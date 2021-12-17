import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
import fitsio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize
import copy
import time
from numba import jit
import multiprocessing
from scipy.signal import find_peaks

class TheoryPredictor:
    def __init__(self, Data, Geometry, astroObj = {}, params = {}, thresh=0.03, zeroSats = []):
        #Data: BMXSingleFreqObs object
        #Geometry: SingleFreqGeometry object
        #astroObj: dictionary with format {'[Astronomical Source Name]': (RA, DEC)}
        #params: dictionary with format {'[Paramter]': value}; starting values of fit parameters
        #thresh: int or float; satellites will be excluded from fitting if cos(altitude)^(-2) doesn't go below this threshhold.
        #zeroSats: 1D list of strings; any satellites named in this will be ignored by TheoryPredictor
        if type(Data) != list:
            Data = [Data]
        self.data = copy.deepcopy(Data) #copy of Data
        self.geometry = copy.deepcopy(Geometry) #copy of Geometry
        
        self.names = [] #1D list of strings; names of parameters for dictionary input
        self.parameterBounds = {} #dictionary with format {'[parameter]': (low_bound, high_bound)}; dictionary of bounds for named parameters
        self.offsets_r = [] #list of dictionaries with format {int channel: int or float}; offset added to signal, organized first by dataset and then by channel 
        self.offsets_i = [] #same as above, but is multiplied by 1j when added to signal
        for i in range(len(self.data)):
            self.offsets_r.append({
                11:sum(self.data[i][11]-abs(self.data[i][12])*abs(self.data[i][13])/abs(self.data[i][23]))/len(self.data[i][11]),
                12:0,
                13:0,
                14:0,
                22:sum(self.data[i][22]-abs(self.data[i][12])*abs(self.data[i][23])/abs(self.data[i][13]))/len(self.data[i][22]),
                23:0,
                24:0,
                33:sum(self.data[i][33]-abs(self.data[i][13])*abs(self.data[i][23])/abs(self.data[i][12]))/len(self.data[i][33]),
                34:0,
                44:sum(self.data[i][44]-abs(self.data[i][14])*abs(self.data[i][24])/abs(self.data[i][12]))/len(self.data[i][44]),
                55:sum(self.data[i][55]-abs(self.data[i][56])*abs(self.data[i][57])/abs(self.data[i][67]))/len(self.data[i][55]),
                56:0,
                57:0,
                58:0,
                66:sum(self.data[i][66]-abs(self.data[i][56])*abs(self.data[i][67])/abs(self.data[i][57]))/len(self.data[i][66]),
                67:0,
                68:0,
                77:sum(self.data[i][77]-abs(self.data[i][57])*abs(self.data[i][67])/abs(self.data[i][56]))/len(self.data[i][77]),
                78:0,
                88:sum(self.data[i][88]-abs(self.data[i][58])*abs(self.data[i][68])/abs(self.data[i][56]))/len(self.data[i][88]),
            })
            self.offsets_i.append({
                11:0,
                12:0,
                13:0,
                14:0,
                22:0,
                23:0,
                24:0,
                33:0,
                34:0,
                44:0,
                55:0,
                56:0,
                57:0,
                58:0,
                66:0,
                67:0,
                68:0,
                77:0,
                78:0,
                88:0
            })
        
        astroNames = list(astroObj.keys()) #Names of astronomical objects included in the fit
        astroTracks = self.astTrack(np.array([astroObj[n] for n in astroNames])) #tracks of astronomical objects for each data set
        
        self.satTracks = [] #list of numpy arrays, shape (number of singal sources, len(self.data), number of samples in data, 2)
        self.satNames = [] #list of strings, shape (number of singal sources); names of satellites and astroObjs
        self.satAmps = [] #list of numpy arrays, shape (number of signal sources, len(self.data), number of detectors)
        sats = set()
        for i,D in enumerate(self.data):
            channels = [12,13,14,23,24,34,56,57,58,67,68,78]
            for j,n in enumerate(D.sat_id):
                if "COS" not in n and n not in zeroSats:
                    cos2 = np.cos(D.sat[j]['alt'])**(-2)
                    peaks = find_peaks(cos2,height=1/thresh)[0]
                    for p in peaks:
                        if np.any([abs(D[ch][p])>max(abs(D[ch]))/1e3 for ch in channels]):
                            sats = sats | {n}
        for n in sats:
            self.satNames.append(n)
            tracks = []
            for i,D in enumerate(self.data):
                if n in D.sat_id:
                    s = np.array(D.sat)[np.array(D.sat_id)==n][0]
                    tracks.append(np.array([np.cos(s['alt'])*np.sin(s['az']),np.cos(s['alt'])*np.cos(s['az'])]).T)
                else:
                    s = D.sat[0]
                    tracks.append(np.array([np.sin(s['az']),np.cos(s['az'])]).T)
                self.names += ['A{}_{}_{}'.format(k+1, n, i) for k in range(8)]
                self.names += ['A_{}_{}'.format(n, i)]
                self.parameterBounds['A_{}_{}'.format(n, i)] = (0,np.sqrt(min([max(D[(k+1)*11]) for k in range(8)]))*1.5e-6)
                for k in range(8):
                    self.parameterBounds['A{}_{}_{}'.format(k+1, n, i)] = (0,np.sqrt(max(D[(k+1)*11]))*1.5e-7)
            self.satTracks.append(tracks)
                        
        for j,n in enumerate(astroNames):
            self.satNames.append(n)
            tracks = []
            for i,D in enumerate(self.data):
                tracks.append(astroTracks[i][j])
                self.names += ['A{}_{}_{}'.format(k+1, n, i) for k in range(8)]
                self.names += ['A_{}_{}'.format(n, i)]
                self.parameterBounds['A_{}_{}'.format(n, i)] = (0,np.sqrt(min([max(D[(k+1)*11]) for k in range(8)]))*1.5e-6)
                for k in range(8):
                    self.parameterBounds['A{}_{}_{}'.format(k+1, n, i)] = (0,np.sqrt(max(D[(k+1)*11]))*1.5e-7)
            self.satTracks.append(tracks)
        self.satTracks = np.array(self.satTracks)
        self.satNames = np.array(self.satNames)
        self.satAmps = np.zeros((len(self.satNames), len(self.data), 8))
            
        self.approxPeaks()
        
        self.names += ['freq','airy','fix_amplitude','time_offset_all','D_all_dist','beam_sigma','beam_sigma_x','beam_sigma_y','beam_smooth','beam_smooth_x','beam_smooth_y']
        
        
        for i,ant_pos in enumerate(self.geometry.ant_pos):
            self.names += ['D{}_pos_x'.format(i+1),
                           'D{}_pos_y'.format(i+1)]
            self.parameterBounds['D{}_pos_x'.format(i+1)] = (-10,10)
            self.parameterBounds['D{}_pos_y'.format(i+1)] = (-10,10)
            self.names += ['D{}_phi_{}'.format(i+1,j) for j in range(len(self.data))]
            for j in range(len(self.data)):
                self.parameterBounds['D{}_phi_{}'.format(i+1,j)] = (-np.pi,np.pi)
            self.names +=['D{}_beam_center_x'.format(i+1),
                           'D{}_beam_center_y'.format(i+1),
                           'D{}_beam_sigma'.format(i+1),
                           'D{}_beam_sigma_x'.format(i+1),
                           'D{}_beam_sigma_y'.format(i+1),
                           'D{}_beam_smooth'.format(i+1),
                           'D{}_beam_smooth_x'.format(i+1),
                           'D{}_beam_smooth_y'.format(i+1)]
            self.parameterBounds['D{}_beam_center_x'.format(i+1)] = (-0.1,0.1)
            self.parameterBounds['D{}_beam_center_y'.format(i+1)] = (-0.1,0.1)
            self.parameterBounds['D{}_beam_sigma_x'.format(i+1)] = (0.01,0.5)
            self.parameterBounds['D{}_beam_sigma_y'.format(i+1)] = (0.01,0.5)
            self.parameterBounds['D{}_beam_smooth_x'.format(i+1)] = (0.01,0.5)
            self.parameterBounds['D{}_beam_smooth_y'.format(i+1)] = (0.01,0.5)
                           
        for i in range(len(self.data)):
            for ch in [11,12,13,14,22,23,24,33,34,44,55,56,57,58,66,67,68,77,78,88]:
                self.names.append('CH{}_offset_r{}'.format(ch, i))
                self.names.append('CH{}_offset_i{}'.format(ch, i))
                self.parameterBounds['CH{}_offset_r{}'.format(ch, i)] = (min(self.data[i][ch].real),max(self.data[i][ch].real))
                self.parameterBounds['CH{}_offset_i{}'.format(ch, i)] = (min(self.data[i][ch].imag),max(self.data[i][ch].imag))
            
        self.var = [] #names of independent variables when fitting
        self.channels = [11,12,13,14,22,23,24,33,34,44] #channels for fit
        self.cut = [0,len(self.data[0][11])] #cut of dataset when fitting
        self.mode = '' #mode for fitting ('amp' for Amplitude, 'phase' for Phase)
        self.datNum = list(range(len(self.data))) #datasets actively being used in fitting
        
        self.predictTime = 0 #float for keeping track of time spent running fit
        
        if params != {}: #Give parameters their designated starting value
            self.setParameters(params)
            
    def astTrack(self, astro): #generates track for astronomical objects from RA and DEC
        #astro: 2D numpy float array of shape (number of astroObjs, 2); organized by object number, then by 0:RA, 1:DEC
        if len(astro)>0:
            track = []
            RA, DEC = astro[:,0], astro[:,1]
            for D in self.data:
                t = []
                for ra, dec in zip(RA, DEC):
                    HA = D.ra - ra
                    ALT = np.arcsin(np.sin(dec)*np.sin(D.dec)+np.cos(dec)*np.cos(D.dec)*np.cos(HA))
                    a = (np.sin(dec) - np.sin(ALT)*np.sin(D.dec))/(np.cos(ALT)*np.cos(D.dec))
                    a[a>1] = 1
                    a[a<-1] = -1
                    AZ = np.arccos(a) * (2*(np.sin(HA)<0)-1)
                    t.append(np.moveaxis(np.array([np.cos(ALT)*np.sin(AZ),np.cos(ALT)*np.cos(AZ)]),0,-1))
                track.append(t)
            return np.array(track) #numpy array of shape (len(self.data), number of astroObjs, 2); tracks of the locations of the astroObjs across the sky for each day of data.
        else:
            return np.array([])
    
    def approxPeaks(self, width=50): #approximates amplitudes of radio sources from the magnitude of the data around their largest peak.
        for i,D in enumerate(self.data):
            for n,t in zip(self.satNames,self.satTracks[:,i]):
                cos2 = (t**2).sum(axis=-1)
                peak = (np.arange(len(cos2))[cos2==min(cos2)])[0]
                for ch in range(8):
                    if max(abs(D[11*(ch+1)][max(peak-width,0):min(peak+width,len(D[11])-1)]-self.offsets_r[i][11*(ch+1)])) < 0:
                        amp = 0
                    else:
                        amp = np.sqrt(max(abs(D[11*(ch+1)][max(peak-width,0):min(peak+width,len(D[11])-1)]-self.offsets_r[i][11*(ch+1)])))*1e-7
                    self.setParameters({'A{}_{}_{}'.format(ch+1,n,i):amp})
    
    def allParameters(self): #return all parameter names
        return self.names
    
    def setParameters(self, params): #set parameter values with dicitonary
        #params: dictionary with format {'[Parameter]': value}
        for i,D in enumerate(self.data):
            for j,n in enumerate(self.satNames):
                if "A_{}_{}".format(n,i) in params.keys():
                    for k in range(8):
                        self.satAmps[j,i,k] = abs(params["A_{}_{}".format(n,i)])
                for k in range(8):
                    if "A{}_{}_{}".format(k+1,n,i) in params.keys():
                        self.satAmps[j,i,k] = abs(params["A{}_{}_{}".format(k+1,n,i)])
        if 'freq' in params.keys():
            self.geometry.freq = params['freq']
        if 'airy' in params.keys():
            self.geometry.isAiry = params['airy']
        if 'fix_amplitude' in params.keys():
            self.geometry.fixAmp = params['fix_amplitude']
            
        if 'beam_sigma' in params.keys():
            sig2 = params['beam_sigma']**2
            for i in range(len(self.geometry.ant_beam)):
                self.geometry.ant_beam[i].sigma2 = np.array([sig2,sig2])
        if 'beam_sigma_x' in params.keys():
            for i in range(len(self.geometry.ant_beam)):
                self.geometry.ant_beam[i].sigma2[0] = params['beam_sigma_x']**2
        if 'beam_sigma_y' in params.keys():
            for i in range(len(self.geometry.ant_beam)):
                self.geometry.ant_beam[i].sigma2[1] = params['beam_sigma_y']**2
                
        if 'beam_smooth' in params.keys():
            sm2 = params['beam_smooth']**2
            for i in range(len(self.geometry.ant_beam)):
                self.geometry.ant_beam[i].smooth2 = np.array([sm2,sm2])
        if 'beam_smooth_x' in params.keys():
            for i in range(len(self.geometry.ant_beam)):
                self.geometry.ant_beam[i].smooth2[0] = params['beam_smooth_x']**2
        if 'beam_smooth_y' in params.keys():
            for i in range(len(self.geometry.ant_beam)):
                self.geometry.ant_beam[i].smooth2[1] = params['beam_smooth_y']**2
                
        if 'D_all_dist' in params.keys():
            r = params['D_all_dist']
            self.geometry.ant_pos = np.array([[0,r],[r,0],[0,-r],[-r,0],
                                             [0,r],[r,0],[0,-r],[-r,0]])
        for i in range(len(self.geometry.ant_pos)):
            if 'D{}_pos_x'.format(i+1) in params.keys():
                self.geometry.ant_pos[i][0] = params['D{}_pos_x'.format(i+1)]
            if 'D{}_pos_y'.format(i+1) in params.keys():
                self.geometry.ant_pos[i][1] = params['D{}_pos_y'.format(i+1)]
            for j in range(len(self.data)):
                if 'D{}_phi_{}'.format(i+1,j) in params.keys():
                    self.geometry.phi[j][i] = params['D{}_phi_{}'.format(i+1,j)]
            if 'D{}_beam_center_x'.format(i+1) in params.keys():
                self.geometry.ant_beam[i].center[0] = params['D{}_beam_center_x'.format(i+1)]
            if 'D{}_beam_center_y'.format(i+1) in params.keys():
                self.geometry.ant_beam[i].center[1] = params['D{}_beam_center_y'.format(i+1)]
            if 'D{}_beam_sigma'.format(i+1) in params.keys():
                a = params['D{}_beam_sigma'.format(i+1)]**2
                self.geometry.ant_beam[i].sigma2 = np.array((a,a))
            if 'D{}_beam_sigma_x'.format(i+1) in params.keys():
                self.geometry.ant_beam[i].sigma2[0] = params['D{}_beam_sigma_x'.format(i+1)]**2
            if 'D{}_beam_sigma_y'.format(i+1) in params.keys():
                self.geometry.ant_beam[i].sigma2[1] = params['D{}_beam_sigma_y'.format(i+1)]**2
            if 'D{}_beam_smooth'.format(i+1) in params.keys():
                a = params['D{}_beam_smooth'.format(i+1)]**2
                self.geometry.ant_beam[i].smooth2 = np.array((a,a))
            if 'D{}_beam_smooth_x'.format(i+1) in params.keys():
                self.geometry.ant_beam[i].smooth2[0] = params['D{}_beam_smooth_x'.format(i+1)]**2
            if 'D{}_beam_smooth_y'.format(i+1) in params.keys():
                self.geometry.ant_beam[i].smooth2[1] = params['D{}_beam_smooth_y'.format(i+1)]**2
        for i in range(len(self.data)):
            for ch in [11,12,13,14,22,23,24,33,34,44,55,56,57,58,66,67,68,77,78,88]:
                if 'CH{}_offset_r{}'.format(ch,i) in params.keys():
                    self.offsets_r[i][ch] = params['CH{}_offset_r{}'.format(ch,i)]
                if 'CH{}_offset_i{}'.format(ch,i) in params.keys():
                    self.offsets_i[i][ch] = params['CH{}_offset_i{}'.format(ch,i)]
        return
                
    def readParameters(self): #return all parameters as dictionary
        params = {}
        for i,D in enumerate(self.data):
            for j,n in enumerate(self.satNames):
                params["A_{}_{}".format(n,i)] = np.mean(self.satAmps[j][i])
                for k in range(8):
                    params['A{}_{}_{}'.format(k+1,n,i)] = self.satAmps[j][i][k]
        params['freq'] = self.geometry.freq
        params['airy'] = self.geometry.isAiry
        params['fix_amplitude'] = self.geometry.fixAmp
        for i,ant_pos in enumerate(self.geometry.ant_pos):
            params['D{}_pos_x'.format(i+1)] = ant_pos[0]
            params['D{}_pos_y'.format(i+1)] = ant_pos[1]
            for j in range(len(self.data)):
                params['D{}_phi_{}'.format(i+1,j)] = self.geometry.phi[j][i]
            params['D{}_beam_center_x'.format(i+1)] = self.geometry.ant_beam[i].center[0]
            params['D{}_beam_center_y'.format(i+1)] = self.geometry.ant_beam[i].center[1]
            params['D{}_beam_center_x'.format(i+1)] = self.geometry.ant_beam[i].center[0]
            params['D{}_beam_center_y'.format(i+1)] = self.geometry.ant_beam[i].center[1]
            params['D{}_beam_sigma_x'.format(i+1)] = np.sqrt(self.geometry.ant_beam[i].sigma2[0])
            params['D{}_beam_sigma_y'.format(i+1)] = np.sqrt(self.geometry.ant_beam[i].sigma2[1])
            params['D{}_beam_smooth_x'.format(i+1)] = np.sqrt(self.geometry.ant_beam[i].smooth2[0])
            params['D{}_beam_smooth_y'.format(i+1)] = np.sqrt(self.geometry.ant_beam[i].smooth2[1])
        for i in range(len(self.data)):
            for ch in [11,12,13,14,22,23,24,33,34,44,55,56,57,58,66,67,68,77,78,88]:
                params['CH{}_offset_r{}'.format(ch,i)] = self.offsets_r[i][ch]
                params['CH{}_offset_i{}'.format(ch,i)] = self.offsets_i[i][ch]
        return params #dictionary with format {'[Parameter]': value}
    
    def output(self, channel, datNum, sources = [], amp = True): #return theory predictions
        #channel: int
        #datNum: int; index of data, indicating for which day of data to make predictions
        #sources: list of ints; indices of satellites to be used in making prediction; all satellites will be used if blank
        #amp: bool; Use satAmps to calculate the amplitude, or set all amplitudes to a fixed value
        if sources == []:
            sources = list(range(len(self.satNames)))
        if amp:
            A = (self.satAmps[sources,datNum,channel//10 - 1] * self.satAmps[sources,datNum,channel%10 - 1])[:,None]*1e14
        else:
            A = 1e14
        satOut = self.geometry.point_source(channel, 1, self.satTracks[sources,datNum], datNum) * A
        signal = satOut.sum(axis=0) + self.offsets_r[datNum][channel] + self.offsets_i[datNum][channel]*1j
        return signal #complex numpy array of shape (number of data samples)
        
    def fitFunc(self, params): #parameterized function used in fitting
        #params: 1D float array; incoming parameter values from least_squares function
        t=time.time()
        p = {}
        for i,n in enumerate(self.var): #match params to parameter names and update their values in TheoryPredictor
            p[n] = params[i]
        self.setParameters(p)
        
        out = np.array([]) #1D array of floats
        for ch in self.channels:
            for i in self.datNum:
                prediction = self.output(ch, i)[self.cut[0]:self.cut[1]] #*self.weight[i][ch]
                data = self.data[i][ch][self.cut[0]:self.cut[1]] #*self.weight[i][ch]
                if self.mode == 'amp':
                    out = np.append(out, abs(data)-abs(prediction))
                elif self.mode == 'phase':
                    pphase = prediction * abs(data)/np.maximum(abs(prediction),1e-20)
                    out = np.append(out, data.real-pphase.real)
                    out = np.append(out, data.imag-pphase.imag)
                else:
                    out = np.append(out, data.real-prediction.real)
                    out = np.append(out, data.imag-prediction.imag)
        
        self.predictTime += time.time()-t
        return out #sum(out.real**2+out.imag**2)

    def showFit(self, channels = [], cut = [], mode = '', perSat=False): #display graphs of fitted results
        #channels: list of ints; channels to be displayed; shows all channels if empty
        #cut: list of two ints; start and end indices of data/fit to be displayed
        #mode: string; 'phase' makes predictions match data's amplitude, 'amp' displays only magnitude of data and predictions.  Anything else makes it display normally
        #perSat: boolean; if True, displays each satellite's signal individually
        if channels == []:
            channels = self.channels
        if cut == []:
            cut = self.cut
        if mode == '':
            mode = self.mode
        
        if perSat and mode!='phase': #Plot each satellite beam individually
            sats = []
            for ch in channels:
                for i,D in enumerate(self.data):
                    SatOut = {}
                    for j,n in enumerate(self.satNames):
                        prediction = self.output(ch, i, sources=[j])[cut[0]:cut[1]]
                        if mode == 'amp':
                            SatOut[n] = abs(prediction)
                        else:
                            SatOut[n] = prediction
                    sats.append(SatOut)
        
        Dout = []
        Pout = []
        for ch in channels:
            for i,D in enumerate(self.data):
                prediction = self.output(ch, i)[cut[0]:cut[1]]
                data = D[ch][cut[0]:cut[1]]
                if mode == 'amp':
                    Dout.append(abs(data))
                    Pout.append(abs(prediction))
                elif mode == 'phase':
                    pphase = prediction * abs(data)/np.maximum(abs(prediction),1e-20)
                    Dout.append(data)
                    Pout.append(pphase)
                else:
                    Dout.append(data)
                    Pout.append(prediction)
        dat = np.array(Dout)
        fit = np.array(Pout)

        for i in range(len(dat)):
            fig = plt.figure(figsize = (12,5))
            axes = fig.subplots(ncols=2)
            axes[0].plot(dat[i].real,label='Data')
            axes[0].plot(fit[i].real,label='Fit')
            axes[0].text(0.45,1.05,'Real', transform=axes[0].transAxes)

            axes[1].plot(dat[i].imag,label='Data')
            axes[1].plot(fit[i].imag,label='Fit')
            axes[1].text(0.45,1.05,'Imag', transform=axes[1].transAxes)
            
            if perSat and mode!='phase':
                for n in sats[i].keys():
                    if max(abs(sats[i][n] - self.offsets_r[i%len(self.data)][channels[i//len(self.data)]])) > max(abs(fit[i] - self.offsets_r[i%len(self.data)][channels[i//len(self.data)]]))/50:
                        axes[0].plot(sats[i][n].real,label=n)
                        axes[1].plot(sats[i][n].imag,label=n)
            
            fig.text(0.8,1.1,'CH {} Fit - pas/211110_1900 - [{}:{}]'.format(channels[i//len(self.data)], cut[0],cut[1]), transform=axes[0].transAxes) #, i%len(self.data)
            plt.legend()
            plt.show()
        
        return
    
    def fit(self, names, mode = 'all', channels = [11,12,13,14,22,23,24,33,34,44,55,56,57,58,66,67,68,77,78,88], datNum = [], cut = [0,-1], pprint = False, plot = False, output=True): #Fits named parameters to data
        #names: list of strings; names of parameters to be fit
        #mode: string; 'phase' makes predictions match data magnitude, 'amp' takes magnitude of data and predictions before subtracting, any thing else makes it run normally
        #channels: list of ints; data channels to be fit
        #datNum: list of ints; indices of data sets to be fit
        #cut: list of two ints; start and end indices in time (samples)
        #pprint: boolean; print fit results at the end?
        #plot: boolean; plot fit results at the end?
        #output: boolean; return fit results?
        if cut[1] == -1:
            cut[1] = len(self.data[0][11])
        if datNum == []:
            datNum = list(range(len(self.data)))
        self.var = names
        self.channels = channels
        self.cut = cut
        self.mode = mode
        self.datNum = datNum
        
        params = []
        bounds_low = []
        bounds_high = []
        state = self.readParameters()
        for n in self.var:
            bounds_low.append(self.parameterBounds[n][0])
            bounds_high.append(self.parameterBounds[n][1])
            params.append(min(max(state[n],bounds_low[-1]),bounds_high[-1]))
        params = np.array(params)
        bounds = (bounds_low, bounds_high)
        
        self.fitTime = time.time()
        self.predictTime = 0
        fit = least_squares(self.fitFunc,params, bounds=bounds)
        self.fitTime = time.time()-self.fitTime
        print('{} of {} seconds spent in predictions'.format(self.predictTime, self.fitTime))
        
        params = fit.x
        
        p = {}
        for i,n in enumerate(self.var):
            p[n] = params[i]
        self.setParameters(p)
        
        if pprint:
            print(params)
        if plot:
            self.showFit()
        if output:
            return params
        return
    
    def fit_parallel(self, params):
        #params: tuple of parameters for the fit function above
        print('Start {}, {}, [{}:{}]'.format(params[2],params[3],params[4][0],params[4][1]))
        p = self.fit(params[0],mode=params[1],channels=params[2],datNum=params[3],cut=params[4])
        print('End {}, {}, [{}:{}]'.format(params[2],params[3],params[4][0],params[4][1]))
        return p
    
    def sigSats(self, cut, thresh=0.03): #Gives set of satellites whose cos^2(ALT) passes below threshold
        #cut: list of two ints; indices for start and end of cut
        #thresh: float; satellites are significant if cos(altitude)^2 passes below threshhold
        if type(cut) == int:
            cut = [cut,cut+1]
        if len(cut)==1:
            cut = [cut[0],cut[0]+1]
        sats = set()
        for D in self.data:
            for s,n in zip(D.sat,D.sat_id):
                if n in self.satNames:
                    if min(np.cos(s['alt'][cut[0]:cut[1]])**2)<thresh:
                        sats = sats | {n}
        return sats
    
    def trackPlot(self, cut=[0,-1], sats=[]): #Plots 2D path of satellites
        #cut: list of two ints
        #sats: list of strings: names of satellites to plot
        if sats==[]:
            sats = list(self.satNames)
        for n,track in zip(self.satNames, self.satTracks):
            for i,D in enumerate(self.data):
                if n in sats:
                    plt.figure(figsize=(12,10))
                    plt.plot(track[i,cut[0]:cut[1],0],track[i,cut[0]:cut[1],1])
                    plt.title('{} track : DataSet {}'.format(n,i))
                    plt.show()
    
    def findCuts(self, thresh=0.03): #Find optimal cuts for separately fitting satellite amplitudes
        #thresh: float; a satellite's signal is present if cos(altitude)^2 <= thresh
        Cuts = []
        Sats = []
        for i in range(len(self.data)):
            cos2 = (self.satTracks[:,i]**2).sum(axis=-1)
            inCut = cos2 <= thresh
            anyInCut = inCut.any(axis=0)
            cutStart = np.arange(len(anyInCut))[1:][np.all([np.invert(anyInCut[:-1]), anyInCut[1:]],axis=0)]
            cutEnd = np.arange(len(anyInCut))[1:][np.all([anyInCut[:-1], np.invert(anyInCut[1:])],axis=0)]
            if anyInCut[0] == True:
                cutStart = np.insert(cutStart,0,0)
            if anyInCut[-1] == True:
                cutEnd = np.append(cutEnd,-1)
            cuts = [[s,e] for s,e in zip(cutStart,cutEnd)]
            sats = []
            for j,cut in enumerate(cuts):
                sats.append([])
                for k,n in enumerate(self.satNames):
                    if np.any(inCut[k][cut[0]:cut[1]]):
                        sats[j].append(n)
            Cuts.append(cuts)
            Sats.append(sats)
        return Cuts, Sats