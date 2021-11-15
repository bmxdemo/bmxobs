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

def getNoAmpFit(Theory):
    detectorSet = [[1,2,3,4],[5,6,7,8]]
    channelSet = [[11,12,13,14,22,23,24,33,34,44],
                  [55,56,57,58,66,67,68,77,78,88]]
    
    paramsOut = {}
    
    TASKS = []
    for detectors,ch in zip(detectorSet,channelSet):
        names = []
        for d in detectors:
            names += ['D{}_beam_center_x'.format(d),
                    'D{}_beam_center_y'.format(d),
                    'D{}_beam_smooth_x'.format(d),
                    'D{}_beam_smooth_y'.format(d)]
            if d!= 1 and d!= 5:
                names += ['D{}_pos_x'.format(d),
                          'D{}_pos_y'.format(d)]
                names += ['D{}_phi_{}'.format(d,j) for j in range(len(Theory.data))]
        TASKS.append((names, 'all', ch, list(range(len(Theory.data))), [0,-1]))
        
            
    with multiprocessing.Pool(len(TASKS)) as pool:
        imap_it = pool.imap(Theory.fit_parallel, TASKS)
        paramsOut = {}
        
        print('Ordered results using pool.imap():')
        for i,x in enumerate(imap_it):
            for n,p in zip(TASKS[i][0],x):
                paramsOut[n] = p
                
    Theory.setParameters(paramsOut)
    
    return paramsOut

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
    Data_ids = ['pas/211025_2000']

    startParams = {'D1_pos_x': 0.0, 'D1_pos_y': 4.0, 'D1_phi_0': 0.0, 'D1_beam_center_x': 0.015372021366992238, 'D1_beam_center_y': -0.017957237148543616, 'D1_beam_sigma_x': 0.05, 'D1_beam_sigma_y': 0.05, 'D1_beam_smooth_x': 0.0502983033608641, 'D1_beam_smooth_y': 0.045469094777206, 'D2_pos_x': 4.518530355262371, 'D2_pos_y': -0.5675777194917917, 'D2_phi_0': -0.23167257957037482, 'D2_beam_center_x': 0.014062073064744095, 'D2_beam_center_y': -0.019532178552770497, 'D2_beam_sigma_x': 0.05, 'D2_beam_sigma_y': 0.05, 'D2_beam_smooth_x': 0.0457799537590874, 'D2_beam_smooth_y': 0.042871828405632716, 'D3_pos_x': 0.25402982959302256, 'D3_pos_y': -4.930860516543797, 'D3_phi_0': 2.3655494305695326, 'D3_beam_center_x': -0.00301813644224245, 'D3_beam_center_y': -0.011432868771212813, 'D3_beam_sigma_x': 0.05, 'D3_beam_sigma_y': 0.05, 'D3_beam_smooth_x': 0.043689282102453715, 'D3_beam_smooth_y': 0.04892401963994162, 'D4_pos_x': -4.234598505256559, 'D4_pos_y': -0.3220397958043332, 'D4_phi_0': -0.3554964349766069, 'D4_beam_center_x': -0.05014578043509247, 'D4_beam_center_y': 0.0158685658665472, 'D4_beam_sigma_x': 0.05, 'D4_beam_sigma_y': 0.05, 'D4_beam_smooth_x': 0.04617671645088604, 'D4_beam_smooth_y': 0.03972326453912008, 'D5_pos_x': 0.0, 'D5_pos_y': 4.0, 'D5_phi_0': 0.0, 'D5_beam_center_x': 0.011186000546988486, 'D5_beam_center_y': -0.009949767188537356, 'D5_beam_sigma_x': 0.05, 'D5_beam_sigma_y': 0.05, 'D5_beam_smooth_x': 0.04173487380458516, 'D5_beam_smooth_y': 0.050091346988889174, 'D6_pos_x': 4.36406787331792, 'D6_pos_y': -0.3227264803938347, 'D6_phi_0': 2.8279775427597182, 'D6_beam_center_x': 0.007218040484671332, 'D6_beam_center_y': -0.02619531022472418, 'D6_beam_sigma_x': 0.05, 'D6_beam_sigma_y': 0.05, 'D6_beam_smooth_x': 0.05568767672734272, 'D6_beam_smooth_y': 0.04363819149960366, 'D7_pos_x': -0.10178021281120257, 'D7_pos_y': -4.824719841087661, 'D7_phi_0': -1.4553120453471653, 'D7_beam_center_x': -0.0003604184757539212, 'D7_beam_center_y': -0.019700232212781076, 'D7_beam_sigma_x': 0.05, 'D7_beam_sigma_y': 0.05, 'D7_beam_smooth_x': 0.044903203983911524, 'D7_beam_smooth_y': 0.0457220199819722, 'D8_pos_x': -5.0480482568704925, 'D8_pos_y': -0.8630018437361706, 'D8_phi_0': 2.953870119979808, 'D8_beam_center_x': -0.04039205428636955, 'D8_beam_center_y': 0.016668129375729614, 'D8_beam_sigma_x': 0.05, 'D8_beam_sigma_y': 0.05, 'D8_beam_smooth_x': 0.04698547422390188, 'D8_beam_smooth_y': 0.04345569776138584}

    bins = (580, 600) #Frequency 1258 MHz
    
    zeroSats = ['BEIDOU-3_M1_(C19)', 'GPS_BIIF-9__(PRN26)', 'GSAT0213_(PRN_E04)', 'GSAT0208_(PRN_E08)', 'GSAT0221_(PRN_E15)', 'BEDIOU-3_M11_(C25)', 'GPS_BIII-4__(PRN_14)', 'GSAT0216_(PRN_E25)', 'GPS_BIII-2__(PRN_18)', 'BEIDOU-3_M3_(C27)', 'GPS_BIIRM-1_(PRN_17)', 'GSAT0204_(PRN_E22)', 'GPS_BIIR-11_(PRN_19)', 'GPS_BIIF-6__(PRN_06)', 'BEIDOU-3_M8_(C30)', 'GPS_BIIR-13_(PRN_02)', 'GPS_BIIR-4__(PRN_20)', 'BEIDOU-3_M7_(C29)', 'BEIDOU_15_(C14)', 'BEIDOU-3_M14_(C33)', 'GPS_BIIRM-4_(PRN_15)', 'GSAT0221_(PRN_E15)', 'GSAT0103_(PRN_E19)', 'GPS_BIII-3__(PRN_23)', 'BEIDOU-3_M5_(C21)', 'GPS-BIIF-11_(PRN_10)', 'GPS-BIIRM-6_(PRN_07)', 'BEIDOU-3_M9_(C23)', 'GPS_BIIF-12_(PRN_32)', 'GSAT0218_(PRN_E31)', 'GPS_BIIRM-2_(PRN_31)', 'BEIDOU-3_M18_(C37)', 'BEIDOU-3_M11_(C25)', 'GPS_BIIF-9__(PRN_26)', 'GSAT0205_(PRN_E24)', 'GPS_BIIR-8__(PRN_16)']
    
    astroObj={'Cygnus_A': [5.233686582997465, 0.7109409436737796]}
    
    needAmp = True

Data = [bmxobs.BMXSingleFreqObs(ids, freq_bins=bins) for ids in Data_ids]

Theory = TheoryPredictor(Data, Geometry = SingleFreqGeometry(len(Data), freq=Data[0].freq, airy=False), params = startParams, zeroSats=zeroSats, astroObj=astroObj)

if needAmp:
    print('Preliminary Amp Fit') #Approximate Satellite Amplitudes
    approxPeaks(Theory)
    print()

print('Begin NoAmp Fit') #Fit Amplitudes
print(getNoAmpFit(Theory))
    
if len(sys.argv)>1: # Save end parameters to file named in 1st argument
    fileOut = 'bmxobs/fits/' + sys.argv[1]
    f = open(fileOut,'w')
    f.write('Data_ids = {}\n\nstartParams = {}\n\nbins = {}\n\nzeroSats = {}\n\nastroObj={}\n'.format(Data_ids, Theory.readParameters(), bins, zeroSats, astroObj))
    f.close()
    