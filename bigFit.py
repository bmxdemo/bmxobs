#Format for call:
#python ampFit.py {output filename} {input filename}

import sys
import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
from bmxobs.TheoryPredictor import TheoryPredictor
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import multiprocessing

def approxPeaks(Theory, width=100):
    for i,D in enumerate(Theory.data):
        for n,t in zip(Theory.satNames[i],Theory.satTracks[i]):
            cos2 = (t**2).sum(axis=-1)
            peak = (np.arange(len(cos2))[cos2==min(cos2)])[0]
            for ch in range(8):
                if max(abs(D[11*(ch+1)][max(peak-width,0):min(peak+width,len(D[11])-1)]-Theory.offsets_r[i][11*(ch+1)])) < 0:
                    amp = 0
                else:
                    amp = np.sqrt(max(abs(D[11*(ch+1)][max(peak-width,0):min(peak+width,len(D[11])-1)]-Theory.offsets_r[i][11*(ch+1)])))
                Theory.setParameters({'A{}_{}_{}'.format(ch+1,n,i):amp})
        
needAmp = False
astroObj= {}

if len(sys.argv)>2: # Load starting parameters from file named in 2nd argument
    fileIn = 'bmxobs/fits/' + sys.argv[2]
    f = open(fileIn,'r')
    startData = f.read()
    f.close()
    exec(startData)
    
else: # Default starting parameters without loading
    Data_ids = ["pas/210904_2300"#,
                #"pas/210905_2300",
                #"pas/210906_2300",
                #"pas/210907_2300",
                #"pas/210908_2300",
                #"pas/210909_2300",
                #"pas/210910_2300",
                #"pas/210911_2300",
                #"pas/210912_2300",
                #"pas/210913_2300"
                ]

    startParams = {'D1_pos_x': 0.0, 'D1_pos_y': 4.0, 'D1_phi_0': 0.0, 'D1_beam_center_x': 0.015372021366992238, 'D1_beam_center_y': -0.017957237148543616, 'D1_beam_sigma_x': 0.05, 'D1_beam_sigma_y': 0.05, 'D1_beam_smooth_x': 0.0502983033608641, 'D1_beam_smooth_y': 0.045469094777206, 'D2_pos_x': 4.518530355262371, 'D2_pos_y': -0.5675777194917917, 'D2_phi_0': -0.23167257957037482, 'D2_beam_center_x': 0.014062073064744095, 'D2_beam_center_y': -0.019532178552770497, 'D2_beam_sigma_x': 0.05, 'D2_beam_sigma_y': 0.05, 'D2_beam_smooth_x': 0.0457799537590874, 'D2_beam_smooth_y': 0.042871828405632716, 'D3_pos_x': 0.25402982959302256, 'D3_pos_y': -4.930860516543797, 'D3_phi_0': 2.3655494305695326, 'D3_beam_center_x': -0.00301813644224245, 'D3_beam_center_y': -0.011432868771212813, 'D3_beam_sigma_x': 0.05, 'D3_beam_sigma_y': 0.05, 'D3_beam_smooth_x': 0.043689282102453715, 'D3_beam_smooth_y': 0.04892401963994162, 'D4_pos_x': -4.234598505256559, 'D4_pos_y': -0.3220397958043332, 'D4_phi_0': -0.3554964349766069, 'D4_beam_center_x': -0.05014578043509247, 'D4_beam_center_y': 0.0158685658665472, 'D4_beam_sigma_x': 0.05, 'D4_beam_sigma_y': 0.05, 'D4_beam_smooth_x': 0.04617671645088604, 'D4_beam_smooth_y': 0.03972326453912008, 'D5_pos_x': 0.0, 'D5_pos_y': 4.0, 'D5_phi_0': 0.0, 'D5_beam_center_x': 0.011186000546988486, 'D5_beam_center_y': -0.009949767188537356, 'D5_beam_sigma_x': 0.05, 'D5_beam_sigma_y': 0.05, 'D5_beam_smooth_x': 0.04173487380458516, 'D5_beam_smooth_y': 0.050091346988889174, 'D6_pos_x': 4.36406787331792, 'D6_pos_y': -0.3227264803938347, 'D6_phi_0': 2.8279775427597182, 'D6_beam_center_x': 0.007218040484671332, 'D6_beam_center_y': -0.02619531022472418, 'D6_beam_sigma_x': 0.05, 'D6_beam_sigma_y': 0.05, 'D6_beam_smooth_x': 0.05568767672734272, 'D6_beam_smooth_y': 0.04363819149960366, 'D7_pos_x': -0.10178021281120257, 'D7_pos_y': -4.824719841087661, 'D7_phi_0': -1.4553120453471653, 'D7_beam_center_x': -0.0003604184757539212, 'D7_beam_center_y': -0.019700232212781076, 'D7_beam_sigma_x': 0.05, 'D7_beam_sigma_y': 0.05, 'D7_beam_smooth_x': 0.044903203983911524, 'D7_beam_smooth_y': 0.0457220199819722, 'D8_pos_x': -5.0480482568704925, 'D8_pos_y': -0.8630018437361706, 'D8_phi_0': 2.953870119979808, 'D8_beam_center_x': -0.04039205428636955, 'D8_beam_center_y': 0.016668129375729614, 'D8_beam_sigma_x': 0.05, 'D8_beam_sigma_y': 0.05, 'D8_beam_smooth_x': 0.04698547422390188, 'D8_beam_smooth_y': 0.04345569776138584}

    bins = (280,300) #Frequency 1178 MHz
    zeroSats = []
    needAmp = True

Data = [bmxobs.BMXSingleFreqObs(ids, freq_bins=bins) for ids in Data_ids]

Theory = TheoryPredictor(Data, Geometry = SingleFreqGeometry(len(Data), freq=Data[0].freq, airy=False), params = startParams, zeroSats=zeroSats, astroObj=astroObj)

if needAmp:
    print('Preliminary Amp Fit') #Approximate Satellite Amplitudes
    approxPeaks(Theory)
    print()

print('Begin Big Fit') #Fit Amplitudes
names = []
detectors = [1,2,3,4,5,6,7,8]
offsetReal = [11,12,13,14,22,23,24,33,34,44,55,56,57,58,66,67,68,77,78,88]
offsetImag = [12,13,14,23,24,34,56,57,58,67,68,78]
for d in detectors:
    names += ['D{}_beam_center_x'.format(d),
            'D{}_beam_center_y'.format(d),
            'D{}_beam_smooth_x'.format(d),
            'D{}_beam_smooth_y'.format(d)]
    if d!=1 and d!=5:
        names += ['D{}_pos_x'.format(d),
                  'D{}_pos_y'.format(d)]
        names += ['D{}_phi_{}'.format(d,j) for j in range(len(Theory.data))]
    for j in range(len(Theory.data)):
        names += ["A{}_{}_{}".format(d,n,j) for n in Theory.satNames[j]]
        #names += ['CH{}_offset_r{}'.format(off,j) for off in offsetReal]
        #names += ['CH{}_offset_i{}'.format(off,j) for off in offsetImag]
Theory.fit(names)
    
if len(sys.argv)>1: # Save end parameters to file named in 1st argument
    fileOut = 'bmxobs/fits/' + sys.argv[1]
    f = open(fileOut,'w')
    f.write('Data_ids = {}\n\nstartParams = {}\n\nbins = {}\n\nzeroSats = {}\n\nastroObj={}\n'.format(Data_ids, Theory.readParameters(), bins, zeroSats, astroObj))
    f.close()
    