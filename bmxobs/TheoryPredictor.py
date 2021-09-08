import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
import fitsio
import numpy as np
import copy

class TheoryPredictor:
    def __init__(self, Data, Geometry = SingleFreqGeometry(), params = {}, satAmp = 1):
        self.data = copy.deepcopy(Data)
        self.geometry = copy.deepcopy(Geometry)
        self.names = [] #names of parameters for dictionary input
        self.satAmps = {} #amplitude of signal from satellite
        self.offsets = np.zeros(8)
        for n in self.data.sat_id:
            if "COS" not in n:
                self.satAmps[n] = satAmp
                self.names.append("A_{}".format(n))
        
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
            
        
        
        if params != {}:
            self.setParameters(params)
    
    def allParameters(self): #return all parameter names
        return self.names
    
    def setParameters(self, params): #set parameter values with dicitonary
        for n in self.satAmps.keys():
            if "A_{}".format(n) in params.keys():
                self.satAmps[n] = params["A_{}".format(n)]
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
        params = self.satAmps.copy()
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
                A = self.satAmps[n]
                track = np.array([np.cos(s['alt'])*np.cos(s['az']),np.cos(s['alt'])*np.sin(s['az'])]).T
                signal = signal + self.geometry.point_source(channel,A,track)
        if (channel%11 == 0):
            signal = signal + self.offsets[channel//11 - 1]
        return signal