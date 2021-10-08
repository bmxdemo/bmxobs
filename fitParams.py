# Arguments to change between runs
Data_ids = [["pas/210904_2300",
            "pas/210905_2300"],
            ["pas/210906_2300",
            "pas/210907_2300"],
            ["pas/210908_2300",
            "pas/210909_2300"],
            ["pas/210910_2300",
            "pas/210911_2300"],
            ["pas/210912_2300",
            "pas/210913_2300"]
           ]

startParams = [
    {'D1_pos_x': 0.0, 'D1_pos_y': 4.0, 'D1_phi': 0.0, 'D1_beam_center_x': 0.01749635983762062, 'D1_beam_center_y': -0.012373568675064233, 'D1_beam_sigma_x': 0.046229835063531566, 'D1_beam_sigma_y': 2.5855930335831183, 'D1_beam_smooth_x': 0.05108013644194418, 'D1_beam_smooth_y': 0.05447353098406783, 'D2_pos_x': 4.496720655923967, 'D2_pos_y': -0.6287600000433483, 'D2_phi': -0.2162769382554462, 'D2_beam_center_x': 0.015057139633920634, 'D2_beam_center_y': -0.017828822755548254, 'D2_beam_sigma_x': 603.9867845730674, 'D2_beam_sigma_y': 260.41041041197315, 'D2_beam_smooth_x': 0.04665942661591509, 'D2_beam_smooth_y': 0.0481010865926077, 'D3_pos_x': 0.3001220480458651, 'D3_pos_y': -5.027992542650962, 'D3_phi': 2.4221178826730054, 'D3_beam_center_x': 0.007639744030000907, 'D3_beam_center_y': 0.007512873713030796, 'D3_beam_sigma_x': 0.020441968538857172, 'D3_beam_sigma_y': 0.2460829848790449, 'D3_beam_smooth_x': 11.512069894113507, 'D3_beam_smooth_y': 0.07429916889840309, 'D4_pos_x': -4.205341249150452, 'D4_pos_y': -0.8634242563413563, 'D4_phi': -0.360279571400098, 'D4_beam_center_x': -0.047881632878266994, 'D4_beam_center_y': 0.02106248487059875, 'D4_beam_sigma_x': 0.034698065651445106, 'D4_beam_sigma_y': 0.023887398983064773, 'D4_beam_smooth_x': 940717.7106833917, 'D4_beam_smooth_y': 1684706.2811076841, 'D5_pos_x': 0.0, 'D5_pos_y': 4.0, 'D5_phi': 0.0, 'D5_beam_center_x': 0.012062337891249978, 'D5_beam_center_y': -0.013355149346370547, 'D5_beam_sigma_x': 0.024316967864202205, 'D5_beam_sigma_y': 0.025010283828254078, 'D5_beam_smooth_x': 1077641.8742468602, 'D5_beam_smooth_y': 753785.546358706, 'D6_pos_x': 4.302920183195884, 'D6_pos_y': -0.30173473558525316, 'D6_phi': 2.7941999397976307, 'D6_beam_center_x': 0.005324033941432069, 'D6_beam_center_y': -0.026549849868700653, 'D6_beam_sigma_x': 0.03436222352551767, 'D6_beam_sigma_y': 3756.4365347514167, 'D6_beam_smooth_x': 6626.1727797933645, 'D6_beam_smooth_y': 0.04750305112583341, 'D7_pos_x': -0.11910451531087028, 'D7_pos_y': -4.367339677477533, 'D7_phi': -1.749506910188624, 'D7_beam_center_x': 0.005457892120875577, 'D7_beam_center_y': -0.021818798787514016, 'D7_beam_sigma_x': 0.023258983449163714, 'D7_beam_sigma_y': 0.026284832944583614, 'D7_beam_smooth_x': 0.10591867687099832, 'D7_beam_smooth_y': 16.327186433744693, 'D8_pos_x': -5.115882137283665, 'D8_pos_y': -1.1103425854390399, 'D8_phi': 2.8637630024124805, 'D8_beam_center_x': -0.03601356017291175, 'D8_beam_center_y': 0.02082528825637791, 'D8_beam_sigma_x': 0.042588702081136826, 'D8_beam_sigma_y': 0.02491853592947365, 'D8_beam_smooth_x': 545398.8945094788, 'D8_beam_smooth_y': 1172898.3444874086}
]

#bins = (280,300) #Frequency 1178 MHz
bins = (1760,1780) #Frequency 1575 MHz (GPS L5)

#zeroSats = ['GPS_BIIF-11_(PRN_10)','BEIDOU-3_M17_(C36)','GPS_BIIRM-6_(PRN_07)'] #Satellites set to 0 amplitude
zeroSats = []

#Code below ideally stays the same

import sys
import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
from bmxobs.TheoryPredictor import TheoryPredictor
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

def getPhaseFit(Theory):
    detectors = [2,3,4,6,7,8]
    channelSet = [[12],[13,23],[14,24,34],
                  [56],[57,67],[58,68,78]]

    paramsOut = {}
        
    for i,d in enumerate(detectors):
        ch = channelSet[i]
        names = ['D{}_pos_x'.format(d),
                 'D{}_pos_y'.format(d),
                 'D{}_phi'.format(d)
                ]

        params = Theory.fit(names, mode = 'phase', channels = ch, plot=False, pprint=False)
        for n,p in zip(names,params):
            paramsOut[n] = p
        print('Detector {} fit done'.format(d))
    
    return paramsOut

def getBeamFit(Theory):
    detectors = [1,2,3,4,5,6,7,8]
    channelSet = [[11],[12,22],[13,23,33],[14,24,34,44],
                  [55],[56,66],[57,67,77],[58,68,78,88]]
    
    paramsOut = {}

    for i,d in enumerate(detectors):
        ch = channelSet[i]
        names = ['D{}_beam_center_x'.format(d),
                'D{}_beam_center_y'.format(d),
                'D{}_beam_sigma_x'.format(d),
                'D{}_beam_sigma_y'.format(d),
                'D{}_beam_smooth_x'.format(d),
                'D{}_beam_smooth_y'.format(d)
                ]


        params = Theory.fit(names, mode = 'amp', channels = ch, plot=False, pprint=False)
        for n,p in zip(names,params):
            paramsOut[n] = p
        print('Detector {} fit done'.format(d))
    
    return paramsOut

def getAmpFit(Theory, mode='amp'):
    detectors = [1,2,3,4,5,6,7,8]
    channelSet = [[11],[12,22],[13,23,33],[14,24,34,44],
                  [55],[56,66],[57,67,77],[58,68,78,88]]
    
    paramsOut = {}
    
    for i,d in enumerate(detectors):
        for j in range(len(Theory.data)):
            ch = channelSet[i]
            names = ["A{}_{}_{}".format(d,n,j) for n in (Theory.satNames - set(zeroSats))]
            #names += ["CH{}_offset_r".format(channel) for channel in ch]
            #for channel in ch:
                #if channel%11 != 0:
                    #names += ["CH{}_offset_i".format(channel)]
            if len(names)>0:
                params = Theory.fit(names, mode = mode, channels = ch, datNum = [j], plot=False, pprint=False)
                for n,p in zip(names,params):
                    paramsOut[n] = p
            print('Detector {} Data {} fit done'.format(d,j))
                
    return paramsOut

print('Begin Fit')

if len(sys.argv)>1:
    DatNum = [int(a) for a in sys.argv[1:]]
else:
    DatNum = list(range(len(Data_ids)))

DataSets = [[bmxobs.BMXSingleFreqObs(ids, freq_bins=bins) for ids in Data_ids[i]] for i in DatNum]

if len(startParams)<len(DataSets):
    startParams = [startParams[0]]*len(DataSets)

Theories = [TheoryPredictor(DataSet, Geometry = SingleFreqGeometry(freq=DataSet[0].freq), params = params, satAmp=np.sqrt(max(DataSet[0][11]))) for DataSet,params in zip(DataSets,startParams)]

for i in range(len(Theories)): #zero out satellite signals
    p = {}
    for z in zeroSats:
        p['A_{}'.format(z)] = 0
    Theories[i].setParameters(p)

for i in range(len(Theories)): #Fit amplitude
    print('Preliminary Amp Fit Theory {}'.format(i))
    getAmpFit(Theories[i])

for i in range(len(Theories)): #Fit beams
    print('Theory {} Beam Fit'.format(i))
    print(getBeamFit(Theories[i]))
    print()

for i in range(len(Theories)): #Fit detectors
    print('Theory {} Phase Fit'.format(i))
    print(getPhaseFit(Theories[i]))
    print()
    
for i in range(len(Theories)): #Refit amplitude
    print('Theory {} Amp Fit'.format(i))
    print(getAmpFit(Theories[i], mode='all'))
    print()