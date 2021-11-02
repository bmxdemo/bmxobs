import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
import fitsio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import copy
import time
from numba import jit
import multiprocessing

class TheoryPredictor:
    def __init__(self, Data, Geometry, astroObj = {}, params = {}, satAmp = 0, satDelay = 0, thresh=0.04, astAmp = 0, zeroSats = []):
        if type(Data) != list:
            Data = [Data]
        self.data = copy.deepcopy(Data)
        self.geometry = copy.deepcopy(Geometry)
        
        self.names = [] #names of parameters for dictionary input
        self.parameterBounds = {} #dictionary of bounds for named parameters
        self.satNames = set()
        self.satAmps = {} #amplitude of signal from satellite
        self.offsets_r = []
        self.offsets_i = []
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
        self.trackOff = {} #spatial offset for satellite tracks
        self.timeOff = {} #index offset for satellite tracks
        self.delay = int(satDelay)
        
        astroNames = list(astroObj.keys()) #Names of astronomical objects included in the fit
        astroTracks = self.astTrack(np.array([astroObj[n] for n in astroNames])) #tracks of astronomical objects for each data set
        
        self.satTracks = []
        self.satNames = []
        self.satAmps = []
        self.trackOff = []
        self.timeOff = []
        for i,D in enumerate(self.data):
            tracks = []
            names = []
            amps = []
            trackOff = []
            timeOff = []
            for n,s in zip(D.sat_id,D.sat):
                if "COS" not in n and n not in zeroSats:
                    if min(np.cos(D.sat[i]['alt'])**2)<thresh:
                        names.append(n)
                        tracks.append(np.array([np.cos(s['alt'])*np.sin(s['az']),np.cos(s['alt'])*np.cos(s['az'])]).T)
                        amps.append(np.zeros(8) + satAmp)
                        trackOff.append([0.,0.])
                        timeOff.append(0)
                        self.names += ['A{}_{}_{}'.format(j+1, n, i) for j in range(8)]
                        for j in range(8):
                            self.parameterBounds['A{}_{}_{}'.format(j+1, n, i)] = (0,np.sqrt(max(D[(j+1)*11]))*1.5)
                        self.names += ['{}_track_offset_x{}'.format(n,i),
                                      '{}_track_offset_y{}'.format(n,i),
                                      '{}_time_offset_{}'.format(n,i)]
                        self.parameterBounds['{}_track_offset_x{}'.format(n,i)] = (-0.1,0.1)
                        self.parameterBounds['{}_track_offset_y{}'.format(n,i)] = (-0.1,0.1)
                        self.parameterBounds['{}_time_offset_{}'.format(n,i)] = (-100,100)
                        
            for j,n in enumerate(astroNames):
                names.append(n)
                tracks.append(astroTracks[i][j])
                amps.append(np.zeros(8) + astAmp)
                trackOff.append([0.,0.])
                timeOff.append(0)
                self.names += ['A{}_{}_{}'.format(j+1, n, i) for j in range(8)]
                for j in range(8):
                    self.parameterBounds['A{}_{}_{}'.format(j+1, n, i)] = (0,np.sqrt(max(D[(j+1)*11]))*1.5)
                self.names += ['{}_track_offset_x{}'.format(n,i),
                              '{}_track_offset_y{}'.format(n,i),
                              '{}_time_offset_{}'.format(n,i)]
                self.parameterBounds['{}_track_offset_x{}'.format(n,i)] = (-0.1,0.1)
                self.parameterBounds['{}_track_offset_y{}'.format(n,i)] = (-0.1,0.1)
                self.parameterBounds['{}_time_offset_{}'.format(n,i)] = (-100,100)
            self.satTracks.append(np.array(tracks))
            self.satNames.append(names)
            self.satAmps.append(np.array(amps))
            self.trackOff.append(np.array(trackOff))
            self.timeOff.append(np.array(timeOff))
        
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
            self.parameterBounds['D{}_beam_sigma_x'.format(i+1)] = (0,0.5)
            self.parameterBounds['D{}_beam_sigma_y'.format(i+1)] = (0,0.5)
            self.parameterBounds['D{}_beam_smooth_x'.format(i+1)] = (0,0.5)
            self.parameterBounds['D{}_beam_smooth_y'.format(i+1)] = (0,0.5)
                           
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
        
        self.predictTime = 0
        
        if params != {}:
            self.setParameters(params)
            
        #Find Noncorrelating Sources
        cor = self.satCorrelation()
        for i,D in enumerate(self.data):
            remove = []
            for j,n in enumerate(self.satNames[i]):
                if cor['{}_{}'.format(n,i)] < max(cor.values())/100:
                    remove.append(j)
            self.satTracks[i] = np.delete(self.satTracks[i],remove,axis=0)
            self.satNames[i] = np.delete(self.satNames[i],remove)
            self.satAmps[i] = np.delete(self.satAmps[i],remove,axis=0)
    
    def astTrack(self, astro): #generates track for astronomical objects from RA and DEC
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
                    a[a<1] = -1
                    AZ = np.arccos(a) + np.pi*(np.sin(HA)<0) #* (2*(np.sin(HA)>0)-1)
                    t.append(np.moveaxis(np.array([np.cos(ALT)*np.cos(AZ),np.cos(ALT)*np.sin(AZ)]),0,-1))
                track.append(t)
            return np.array(track)
        else:
            return np.array([])
    
    def allParameters(self): #return all parameter names
        return self.names
    
    def setParameters(self, params): #set parameter values with dicitonary
        for i,names in enumerate(self.satNames):
            for j,n in enumerate(names):
                if "{}_track_offset_x{}".format(n,i) in params.keys():
                    self.trackOff[i][j][0] = params["{}_track_offset_x{}".format(n,i)]
                if "{}_track_offset_y{}".format(n,i) in params.keys():
                    self.trackOff[i][j][1] = params["{}_track_offset_y{}".format(n,i)]
                if "{}_time_offset_{}".format(n,i) in params.keys():
                    self.timeOff[i][j] = int(params["{}_time_offset_{}".format(n,i)])
                for k in range(8):
                    if "A{}_{}_{}".format(k+1,n,i) in params.keys():
                        self.satAmps[i][j][k] = abs(params["A{}_{}_{}".format(k+1,n,i)])
        if 'freq' in params.keys():
            self.geometry.freq = params['freq']
        if 'airy' in params.keys():
            for i in range(len(self.geometry.ant_beam)):
                self.geometry.ant_beam[i].airy = params['airy']
        if 'fix_amplitude' in params.keys():
            for i in range(len(self.geometry.ant_beam)):
                self.geometry.ant_beam[i].fixAmp = params['fix_amplitude']
        if 'time_offset_all' in params.keys():
            self.delay = int(params['time_offset_all'])
            
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
        for i,names in enumerate(self.satNames):
            for j,n in enumerate(names):
                params["{}_track_offset_x{}".format(n,i)] = self.trackOff[i][j][0]
                params["{}_track_offset_y{}".format(n,i)] = self.trackOff[i][j][1]
                params["{}_time_offset_{}".format(n,i)] = self.timeOff[i][j]
                for k in range(8):
                    params['A{}_{}_{}'.format(k+1,n,i)] = self.satAmps[i][j][k]
        params['freq'] = self.geometry.freq
        params['airy'] = self.geometry.ant_beam[0].airy
        params['fix_amplitude'] = self.geometry.ant_beam[0].fixAmp
        params['time_offset_all'] = self.delay
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
        return params
    
    def output(self, channel, datNum, sources = [], amp = True): #return theory predictions
        if sources == []:
            sources = list(range(len(self.satNames[datNum])))
        if amp:
            A = (self.satAmps[datNum][sources,channel//10 - 1] * self.satAmps[datNum][sources,channel%10 - 1])[:,None]
        else:
            A = 1e14
        satOut = self.geometry.point_source(channel, 1, self.satTracks[datNum][sources], datNum) * A
        signal = satOut.sum(axis=0) + self.offsets_r[datNum][channel] + self.offsets_i[datNum][channel]*1j
        return signal
        
    def fitFunc(self, params): #parameterized function used in fitting
        t=time.time()
        p = {}
        for i,n in enumerate(self.var):
            p[n] = params[i]
        self.setParameters(p)
        #centers = np.array([beam.center for beam in self.geometry.ant_beam])
        #smooth2s = np.array([beam.smooth2 for beam in self.geometry.ant_beam])
        #Data = np.array([[self.data[i][ch] for ch in self.channels] for i in self.datNum])
        #offsets_r = np.array([[self.offsets_r[i][ch] for ch in self.channels] for i in self.datNum])
        #offsets_i = np.array([[self.offsets_i[i][ch] for ch in self.channels] for i in self.datNum])
        
        #out = fitFunc(self.channels, self.datNum, self.mode, self.cut, Data, self.satTracks, self.geometry.ant_pos, centers, smooth2s, self.geometry.phi, self.geometry.freq, offsets_r, offsets_i, self.satAmps)
        
        TASKS = []
        for ch in self.channels:
            for i in self.datNum:
                TASKS.append((ch,i))
                
        out = np.array([])
        for x in TASKS:
            out = np.append(out,self.parallelOutput(x))
        
        #if len(TASKS)>1:
            #with multiprocessing.Pool(len(TASKS)) as pool:
                #imap_it = pool.imap_unordered(self.parallelOutput, TASKS)
                #for i,x in enumerate(imap_it):
                    #out = np.append(out,x)
        #else:
            #out = np.append(out,self.parallelOutput(TASKS[0]))
        
        self.predictTime += time.time()-t
        return out
    
    def parallelOutput(self, args):
        ch,i = args[0],args[1]
        out = np.array([])
        prediction = self.output(ch, i)[self.cut[0]:self.cut[1]]
        data = self.data[i][ch][self.cut[0]:self.cut[1]]
        if self.mode == 'amp':
            out = np.append(out, abs(data)-abs(prediction))
        elif self.mode == 'phase':
            pphase = prediction * abs(data)/np.maximum(abs(prediction),1e-20)
            out = np.append(out, data.real-pphase.real)
            out = np.append(out, data.imag-pphase.imag)
        else:
            out = np.append(out, data.real-prediction.real)
            out = np.append(out, data.imag-prediction.imag)
        return out

    def showFit(self, channels = [], cut = [], mode = '', perSat=False): #display graphs of fitted results
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
                    for j,n in enumerate(self.satNames[i]):
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
            
            fig.text(0.85,1.1,'CH {} Fit - DataSet {} - [{}:{}]'.format(channels[i//len(self.data)], i%len(self.data), cut[0],cut[1]), transform=axes[0].transAxes)
            plt.legend()
            plt.show()
        
        return
    
    def fit(self, names, mode = 'all', channels = [11,12,13,14,22,23,24,33,34,44,55,56,57,58,66,67,68,77,78,88], datNum = [], cut = [0,-1], pprint = False, plot = False, output=True): #Fits named parameters to data
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
            params.append(state[n])
            bounds_low.append(self.parameterBounds[n][0])
            bounds_high.append(self.parameterBounds[n][1])
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
        print('Start {}'.format(params[2]))
        p = self.fit(params[0],mode=params[1],channels=params[2],datNum=params[3])
        print('End {}'.format(params[2]))
        return p
    
    def sigSats(self, cut, thresh=0.04): #Gives set of satellites whose cos^2(ALT) passes below threshold
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
        if sats==[]:
            sats = list(self.satNames)
        for i,D in enumerate(self.data):
            for n,track in zip(self.satNames[i], self.satTracks[i]):
                if n in sats:
                    plt.figure(figsize=(12,10))
                    plt.plot(track[cut[0]:cut[1],0],track[cut[0]:cut[1],1])
                    plt.title('{} track : DataSet {}'.format(n,i))
                    plt.show()
                    
    def satCorrelation(self, sats=[]): #Attempting to measure correlation between satellite's predicted signal and data to know whether to include it or not
        allSats = False
        if sats==[]:
            allSats= True
        Correlations = {}
        for i,D in enumerate(self.data):
            for j,n in enumerate(self.satNames[i]):
                if n in sats or allSats:
                    Correlations['{}_{}'.format(n,i)] = sum([(sum((self.output(ch, i, sources=[j], amp=False)-self.offsets_r[i][ch] - self.offsets_i[i][ch]*1j)*(np.conj(D[ch])-self.offsets_r[i][ch] - self.offsets_i[i][ch]*1j))).real for ch in [12,13,14,23,24,34,56,57,58,67,68,78]])
        return Correlations

@jit(nopython=True)
def ant_beam(track, center, smooth2):
    track = np.atleast_2d(track)
    r = (track-center)
    ra = np.sqrt((r*r/smooth2).sum(axis=-1))
    beam = np.exp(-0.5*((r*r)/smooth2).sum(axis=-1))
    return beam/max(beam)

@jit(nopython=True)
def output(channel, tracks, ant_pos, centers, smooth2s, phi, freq, A):
    signal = np.zeros(len(tracks[0])).astype(np.complex128)
    for i,track in enumerate(tracks):
        ch1 = channel // 10 - 1
        ch2 = channel % 10 - 1
        beams = ant_beam(track, centers[ch1], smooth2s[ch1])*ant_beam(track, centers[ch2], smooth2s[ch2])
        baseline = (ant_pos[ch2]-ant_pos[ch1]) * (freq*1e6 / 3e8) #freq*1e6/3e8 = 1/lambda
        phase = (track*baseline).sum(axis=-1)*2*np.pi+phi[ch1]-phi[ch2]
        fringe = np.exp(1j*phase)
        signal += fringe*beams
    return signal

@jit(nopython=True)
def fitFunc(channels, datNum, mode, cut, Data, satTracks, ant_pos, centers, smooth2s, phi, freq, offsets_r, offsets_i, satAmps):
    out = np.zeros(0)
    for i,D in zip(datNum,Data):
        for j,ch in enumerate(channels):
            A = satAmps[i][:,ch//10 - 1] * satAmps[i][:,ch%10 - 1]
            pred = output(ch, satTracks[i], ant_pos, centers, smooth2s, phi[i], freq, A) + offsets_r[i,j] + offsets_i[i,j]*1j
            prediction = pred[cut[0]:cut[1]]
            data = D[j][cut[0]:cut[1]]
            if mode == 'amp':
                out = np.append(out, np.absolute(data)-np.absolute(prediction))
            elif mode == 'phase':
                pphase = prediction * np.absolute(data)/np.absolute(prediction)
                out = np.append(out, data.real-pphase.real)
                out = np.append(out, data.imag-pphase.imag)
            else:
                out = np.append(out, data.real-prediction.real)
                out = np.append(out, data.imag-prediction.imag)
    return out