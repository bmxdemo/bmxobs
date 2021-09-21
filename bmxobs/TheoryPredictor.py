import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
import fitsio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import copy

class TheoryPredictor:
    def __init__(self, Data, Geometry = SingleFreqGeometry(), params = {}, satAmp = 1, satDelay = 0, thresh=0.01):
        if type(Data) != list:
            Data = [Data]
        self.data = copy.deepcopy(Data)
        self.geometry = copy.deepcopy(Geometry)
        self.names = [] #names of parameters for dictionary input
        self.satNames = set()
        self.satAmps = {} #amplitude of signal from satellite
        self.offsets = np.array([
            sum(self.data[0][11]-abs(self.data[0][12])*abs(self.data[0][13])/abs(self.data[0][23]))/len(self.data[0][11]),
            sum(self.data[0][22]-abs(self.data[0][12])*abs(self.data[0][23])/abs(self.data[0][13]))/len(self.data[0][11]),
            sum(self.data[0][33]-abs(self.data[0][13])*abs(self.data[0][23])/abs(self.data[0][12]))/len(self.data[0][11]),
            sum(self.data[0][44]-abs(self.data[0][24])*abs(self.data[0][34])/abs(self.data[0][23]))/len(self.data[0][11]),
            sum(self.data[0][55]-abs(self.data[0][56])*abs(self.data[0][57])/abs(self.data[0][67]))/len(self.data[0][11]),
            sum(self.data[0][66]-abs(self.data[0][56])*abs(self.data[0][67])/abs(self.data[0][57]))/len(self.data[0][11]),
            sum(self.data[0][77]-abs(self.data[0][57])*abs(self.data[0][67])/abs(self.data[0][56]))/len(self.data[0][11]),
            sum(self.data[0][88]-abs(self.data[0][68])*abs(self.data[0][78])/abs(self.data[0][67]))/len(self.data[0][11])
        ])
        self.trackOff = {} #spatial offset for satellite tracks
        self.timeOff = {} #index offset for satellite tracks
        self.delay = int(satDelay)
        
        self.astroNames = [] #Names of astronomical objects included in the fit
        self.astroTracks = [[] for D in self.data] #tracks of astronomical objects for each data set
        self.astroAmps = {} #amplitude of signal from astronomical object
        
        
        for D in self.data:
            for i,n in enumerate(D.sat_id):
                if "COS" not in n:
                    if min(np.cos(D.sat[i]['alt'])**2)<thresh:
                        self.satNames = self.satNames | {n}
        for n in self.satNames:
            self.satAmps[n] = [satAmp]*8
            self.trackOff[n] = np.array([0.,0.])
            self.timeOff[n] = 0
            self.names.append("A_{}".format(n))
            self.names.append("{}_track_offset_x".format(n))
            self.names.append("{}_track_offset_y".format(n))
            self.names.append("{}_time_offset".format(n))
            for ch in range(8):
                self.names.append("A{}_{}".format(ch+1,n))
        
        self.names += ['freq','time_offset_all','D_all_dist','beam_sigma','beam_sigma_x','beam_sigma_y','beam_smooth','beam_smooth_x','beam_smooth_y']
        
        for i,ant_pos in enumerate(self.geometry.ant_pos):
            self.names += ['D{}_pos_x'.format(i+1),
                           'D{}_pos_y'.format(i+1),
                           'D{}_phi'.format(i+1),
                           'D{}_beam_center_x'.format(i+1),
                           'D{}_beam_center_y'.format(i+1),
                           'D{}_beam_sigma'.format(i+1),
                           'D{}_beam_sigma_x'.format(i+1),
                           'D{}_beam_sigma_y'.format(i+1),
                           'D{}_beam_smooth'.format(i+1),
                           'D{}_beam_smooth_x'.format(i+1),
                           'D{}_beam_smooth_y'.format(i+1),
                           'CH{}_offset'.format((i+1)*11)]
            
        self.var = [] #names of independent variables when fitting
        self.channels = [11,12,13,14,22,23,24,33,34,44] #channels for fit
        self.cut = [0,len(self.data[0][11])] #cut of dataset when fitting
        self.mode = '' #mode for fitting ('amp' for Amplitude, 'phase' for Phase)
        
        
        if params != {}:
            self.setParameters(params)
    
    def allParameters(self): #return all parameter names
        return self.names
    
    def setParameters(self, params): #set parameter values with dicitonary
        for n in self.satAmps.keys():
            if "A_{}".format(n) in params.keys():
                self.satAmps[n] = [params["A_{}".format(n)]]*8
            if "{}_track_offset_x".format(n) in params.keys():
                self.trackOff[n][0] = params["{}_track_offset_x".format(n)]
            if "{}_track_offset_y".format(n) in params.keys():
                self.trackOff[n][1] = params["{}_track_offset_y".format(n)]
            if "{}_time_offset".format(n) in params.keys():
                self.timeOff[n] = int(params["{}_time_offset".format(n)])
            for i in range(8):
                if "A{}_{}".format(i+1,n) in params.keys():
                    self.satAmps[n][i] = params["A{}_{}".format(i+1,n)]
        if 'freq' in params.keys():
            self.geometry.freq = params['freq']
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
            if 'D{}_phi'.format(i+1) in params.keys():
                self.geometry.phi[i] = params['D{}_phi'.format(i+1)]
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
            if 'CH{}_offset'.format((i+1)*11) in params.keys():
                self.offsets[i] = params['CH{}_offset'.format((i+1)*11)]
                
    def readParameters(self): #return all parameters as dictionary
        params = {}
        for n in self.satAmps.keys():
            for i in range(8):
                params['A{}_{}'.format(i+1,n)] = self.satAmps[n][i]
            params["{}_track_offset_x".format(n)] = self.trackOff[n][0]
            params["{}_track_offset_y".format(n)] = self.trackOff[n][1]
            params["{}_time_offset".format(n)] = self.timeOff[n]
        params['freq'] = self.geometry.freq
        params['time_offset_all'] = self.delay
        for i,ant_pos in enumerate(self.geometry.ant_pos):
            params['D{}_pos_x'.format(i+1)] = ant_pos[0]
            params['D{}_pos_y'.format(i+1)] = ant_pos[1]
            params['D{}_phi'.format(i+1)] = self.geometry.phi[i]
            params['D{}_beam_center_x'.format(i+1)] = self.geometry.ant_beam[i].center[0]
            params['D{}_beam_center_y'.format(i+1)] = self.geometry.ant_beam[i].center[1]
            params['D{}_beam_center_x'.format(i+1)] = self.geometry.ant_beam[i].center[0]
            params['D{}_beam_center_y'.format(i+1)] = self.geometry.ant_beam[i].center[1]
            params['D{}_beam_sigma_x'.format(i+1)] = np.sqrt(self.geometry.ant_beam[i].sigma2[0])
            params['D{}_beam_sigma_y'.format(i+1)] = np.sqrt(self.geometry.ant_beam[i].sigma2[1])
            params['D{}_beam_smooth_x'.format(i+1)] = np.sqrt(self.geometry.ant_beam[i].smooth2[0])
            params['D{}_beam_smooth_y'.format(i+1)] = np.sqrt(self.geometry.ant_beam[i].smooth2[1])
            params['CH{}_offset'.format((i+1)*11)] = self.offsets[i]
        return params
    
    def output(self, channel, datNum): #return theory predictions
        D = self.data[datNum]
        signal = np.zeros(len(D.sat[0]))
        for n,s in zip(D.sat_id,D.sat):
            if n in self.satNames:
                A = self.satAmps[n][channel//10-1] * self.satAmps[n][channel%10-1]
                track = np.array([np.cos(s['alt'])*np.sin(s['az']),np.cos(s['alt'])*np.cos(s['az'])]).T + self.trackOff[n]
                satOut = self.geometry.point_source(channel,1,track)
                satShift = np.roll(satOut, self.timeOff[n]+self.delay, axis=0) #Part of signal gets looped around; need to zero that out
                signal = signal + satShift*A
        for n,track in zip(self.astroNames, self.astroTracks[datNum]):
            A = self.astroAmps[n][channel//10-1] * self.astroAmps[n][channel%10-1]
            signal = signal + self.geometry.point_source(channel,A,track)
        if (channel%11 == 0):
            signal = signal + self.offsets[channel//11 - 1]
        return signal
        
    def fitFunc(self, params): #parameterized function used in fitting
        p = {}
        for i,n in enumerate(self.var):
            p[n] = params[i]
        self.setParameters(p)
        out = []
        for ch in self.channels:
            for i,D in enumerate(self.data):
                prediction = self.output(ch, i)[self.cut[0]:self.cut[1]]
                data = D[ch][self.cut[0]:self.cut[1]]
                if self.mode == 'amp':
                    out.append(abs(data)-abs(prediction))
                elif self.mode == 'phase':
                    pphase = prediction * abs(data)/abs(prediction)
                    out.append(data.real-pphase.real)
                    out.append(data.imag-pphase.imag)
                else:
                    out.append(data.real-prediction.real)
                    out.append(data.imag-prediction.imag)
        out = np.array(out).flatten()
        return out/1e14

    def showFit(self, channels = [], cut = [], mode = ''): #display graphs of fitted results
        if channels == []:
            channels = self.channels
        if cut == []:
            cut = self.cut
        if mode == '':
            mode = self.mode
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
                    pphase = prediction * abs(data)/abs(prediction)
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
            
            fig.text(0.9,1.1,'CH {} Fit - DataSet {}'.format(channels[i//len(self.data)], i%len(self.data)), transform=axes[0].transAxes)
            plt.legend()
            plt.show()
        
        return
    
    def fit(self, names, channels = [11,12,13,14,22,23,24,33,34,44], cut = [0,-1], mode = '', pprint = True, plot = True, output=True):
        if cut[1] == -1:
            cut[1] = len(self.data[0][11])
        self.var = names
        self.channels = channels
        self.cut = cut
        self.mode = mode
        
        params = []
        state = self.readParameters()
        for n in self.var:
            params.append(state[n])
        params = np.array(params)
        
        fit = least_squares(self.fitFunc,params)
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