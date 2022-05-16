#!/usr/bin/env /cvmfs/astro.sdcc.bnl.gov/SL73/packages/bacon/2021.12/bin/python
#Used with condorfile.job. 15 good days
#Cut is [].
#Format for call:
#python bigFit.py {input filename}

import bmxobs
from bmxobs.SingleFreqGeometry import SingleFreqGeometry
from bmxobs.TheoryPredictor import TheoryPredictor

import os, sys, argparse
import pickle
import multiprocessing
    

def getBigFit(Theory, channels='all', cuts=None):
    detectorSet = [[1,2,3,4],[5,6,7,8]]

    if channels.lower() == 'all':
        channelSet = [[11,12,13,14,22,23,24,33,34,44],
                      [55,56,57,58,66,67,68,77,78,88]]
    elif channels.lower() == 'auto':
        channelSet = [[11,22,33,44],
                      [55,66,77,88]]
    elif channels.lower() == 'cross':
        channelSet = [[12,13,14,23,24,34],
                      [56,57,58,67,68,78]]


    # TODO: implement cuts
    if cuts is None:
        cuts = [0,-1]
    
    paramsOut = {}
    
    TASKS = []
    for detectors,ch in zip(detectorSet,channelSet):
        names = []
        for d in detectors:
            names += ['D{}_beam_center_x'.format(d), #Varies beam center x and y
                      'D{}_beam_center_y'.format(d),
                      'D{}_beam_sigma_x'.format(d),
                      'D{}_beam_sigma_y'.format(d),
                      #'D{}_beam_smooth_x'.format(d),
                      #'D{}_beam_smooth_y'.format(d)
                     ]
            
            for i,D in enumerate(Theory.data):
                names += ["A{}_{}_{}".format(d,n,i) for n in Theory.satNames]
        for c in ch:
            if c%11 == 0:
                names += ['CH{}_offset_r{}'.format(c,i) for i in range(len(Theory.data))]
        TASKS.append((names, 'all', ch, list(range(len(Theory.data))), cuts))
        
            
    with multiprocessing.Pool(len(TASKS)) as pool:
        imap_it = pool.imap(Theory.fit_parallel, TASKS)
        paramsOut = {}
        
        print('Ordered results using pool.imap():')
        for i,x in enumerate(imap_it):
            for n,p in zip(TASKS[i][0],x):
                paramsOut[n] = p
                
    Theory.setParameters(paramsOut)
    
    return paramsOut


def main(args):

    if os.path.isfile(args.dataset): # Load starting parameters from file named in 1st argument
        print('...Loading data from txt file {}...'.format(args.dataset))
        fileIn = args.dataset
        f = open(fileIn, 'r')
        startData = f.read()
        f.close()
        exec(startData)
        print('...Done loading data from txt file {}...'.format(args.dataset))
        
    elif len(args.dataset.split('_')[0]) == 6  and len(args.dataset.split('_')[1]) == 4: # Default starting parameters without loading
        print('...Loading data directly from /gpfs02/astro/workarea/bmxdata/reduced folder...')
        Data_ids = ['pas/'+ args.dataset]
        startParams = {}
        bins = (args.bin_freq_min, args.bin_freq_max) #Frequency 1258 MHz
        zeroSats = []
        astroObj = {'Cygnus_A': [5.233686582997465, 0.7109409436737796]}
        print('...Done loading data directly from /gpfs02/astro/workarea/bmxdata/reduced folder...')
    else:
        errstr = "Dataset to analyze not found. Please check the name of the dataset."
        raise ValueError(errstr)
    
    # Create BMX observation objects
    print('...Creating BMX observation objects...')
    Data = [bmxobs.BMXSingleFreqObs(ids, freq_bins=bins) for ids in Data_ids]
    print('...Done creating BMX observation objects...')

    # Initalize theory predictor
    print('...Initializing theory predictor...')
    Theory = TheoryPredictor(Data,
                             Geometry=SingleFreqGeometry(len(Data), 
                                                         freq=Data[0].freq), 
                                                         params=startParams, 
                                                         zeroSats=zeroSats,
                                                         astroObj=astroObj)
    print('...Done initializing theory predictor...')

    # Fit
    print('...Begin fitting...')
    paramsOut = getBigFit(Theory, channels=args.channels, cuts=args.cuts)
    print(paramsOut)
    print('...Done fitting...')
    
    # Save parameters to pickle file
    print('...Saving parameters to pickle file...')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        pickle.dump(paramsOut, open(args.out_dir+'fit_results_{}.pkl'.format(Data_ids[0].split['/'][1]), 'wb'))
    print('...Done saving parameters to pickle file...')

    # Make plots
    if args.save_plots:
        print('...Making plots...')
        plotdir = os.path.join(args.out_dir,'plots')
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        Theory.showFit(savedir=plotdir, )
        Theory.showFit(savedir=plotdir, perSat=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BMX Beam fitter')
    parser.add_argument('-dataset', dest='dataset', help='text file or BMX obs id', type=str)
    parser.add_argument('-channels', dest='channels', help='which combinations of channels to analyze, auto/cross/all', type=str, default='all')
    parser.add_argument('-cuts', dest='cuts', help='data splits', type=str, default=None)
    parser.add_argument('-bin_freq_min', dest='bin_freq_min', default=580, help='lower frequency bin', type=int)
    parser.add_argument('-bin_freq_max', dest='bin_freq_max', default=600, help='upper frequency bin', type=int)
    parser.add_argument('-out_dir', dest='out_dir', default='.', help='output directory', type=str)
    parser.add_argument('-save_plots', dest='save_plots', action='store_true', help='save diagnostic plots')

    args = parser.parse_args()

    main(args)
