import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
import fitsio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import copy

class TheoryPredictor:
    def __init__(self, Data, Geometry = SingleFreqGeometry(), params = {}, satAmp = 1):
        self.data = copy.deepcopy(Data)
        self.geometry = copy.deepcopy(Geometry)
        self.names = [] #names of parameters for dictionary input
        self.satAmps = {} #amplitude of signal from satellite
        self.offsets = np.zeros(8)
        for i,n in enumerate(self.data.sat_id):
            if "COS" not in n:
                self.satAmps[n] = [satAmp]*8
                self.names.append("A_{}".format(n))
                for ch in range(8):
                    self.names.append("A{}_{}".format(ch+1,n))
        
        self.names += ['freq','D_all_dist','beam_sigma','beam_sigma_x','beam_sigma_y','beam_smooth','beam_smooth_x','beam_smooth_y']
        
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
        self.cut = [0,len(self.data[11])] #cut of dataset when fitting
        self.mode = '' #mode for fitting ('amp' for Amplitude, 'phase' for Phase)
        
        
        if params != {}:
            self.setParameters(params)
    
    def allParameters(self): #return all parameter names
        return self.names
    
    def setParameters(self, params): #set parameter values with dicitonary
        for n in self.satAmps.keys():
            if "A_{}".format(n) in params.keys():
                self.satAmps[n] = [params["A_{}".format(n)]]*8
            for i in range(8):
                if "A{}_{}".format(i+1,n) in params.keys():
                    self.satAmps[n][i] = params["A{}_{}".format(i+1,n)]
        if 'freq' in params.keys():
            self.geometry.freq = params['freq']
            
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
        params['freq'] = self.geometry.freq
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
    
    def output(self, channel): #return theory predictions
        signal = np.zeros(len(self.data.sat[0]))
        for n,s in zip(self.data.sat_id,self.data.sat):
            if "COS" not in n:
                A = self.satAmps[n][channel//10-1] * self.satAmps[n][channel%10-1]
                track = np.array([np.cos(s['alt'])*np.sin(s['az']),np.cos(s['alt'])*np.cos(s['az'])]).T
                satOut = self.geometry.point_source(channel,1,track)
                signal = signal + satOut*A
        if (channel%11 == 0):
            signal = signal + self.offsets[channel//11 - 1]
        return signal
        
    def fitFunc(self, params):
        p = {}
        for i,n in enumerate(self.var):
            p[n] = params[i]
        self.setParameters(p)
        out = []
        for ch in self.channels:
            prediction = self.output(ch)[self.cut[0]:self.cut[1]]
            data = self.data[ch][self.cut[0]:self.cut[1]]
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

    def showFit(self, channels = [], cut = [], mode = ''):
        if channels == []:
            channels = self.channels
        if cut == []:
            cut = self.cut
        if mode == '':
            mode = self.mode
        Dout = []
        Pout = []
        for ch in channels:
            prediction = self.output(ch)[cut[0]:cut[1]]
            data = self.data[ch][cut[0]:cut[1]]
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
        fig = plt.figure(figsize = (12,5*len(channels)))
        axes = fig.subplots(nrows= len(dat), ncols=2)
        if len(dat)==1:
            axes = np.array([axes])

        for i,ch in enumerate(channels):
            axes[i][0].plot(dat[i].real,label='Data')
            axes[i][0].plot(fit[i].real,label='Fit')

            axes[i][1].plot(dat[i].imag,label='Data')
            axes[i][1].plot(fit[i].imag,label='Fit')

        plt.legend()

        plt.tight_layout()
        plt.show()
        
        return
    
    def fit(self, names, channels = [11,12,13,14,22,23,24,33,34,44], cut = [0,-1], mode = '', pprint = True, plot = True, output=True):
        if cut[1] == -1:
            cut[1] = len(self.data[11])
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