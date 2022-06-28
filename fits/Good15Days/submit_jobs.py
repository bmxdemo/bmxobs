import sys,os

def write_condor_job(fn, ds, out_dir, channels, 
                     beam_type='gaussian', cuts=None, thresh=0.03, start_params=None, fix_beam_params=False):
    if fix_beam_params:
        fix_beam_params = '-fix_beam_params'
    else:
        fix_beam_params = ''
        
    f=open(fn,'w')
    f.write("""
Universe        = vanilla
Executable      = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/run_fitter.py
Arguments       = -dataset {} -out_dir {} -channels {} -save_plots -beam_type {} -cuts {} -thresh {} -start_params {} {}
Requirements    = (CPU_Speed >= 1)
Rank            = CPU_Speed
request_memory  = 120000M
request_cpus    = 2
Priority        = 4
GetEnv          = True
Initialdir      = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/
Input           = /dev/null 
Output          = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testout/{}_{}_.out
Error           = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testerr/{}_{}_.err
Log             = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testlog/{}_{}_.log
""".format(ds, out_dir, channels, beam_type, cuts, thresh, start_params, fix_beam_params, fn, channels, fn, channels, fn, channels))
    f.write ('Queue \n')
    f.close()
    
    
data_ids = ['211215_1700',
            '211213_1700',
            '220101_1600',
            '211225_1600',
            '211229_1600',
            '211228_1600',
            '220103_1500',
            '211222_1600',
            '211227_1600',
            '211220_1600',
            '211130_1800',
            '211113_1900',
            '211128_1800',
            '211129_1800',
            '211121_1800'
            ]

    
data_ids_good = ['211213_1700',
            '220101_1600',
            '211225_1600',
            '211228_1600',
            '220103_1500',
            '211222_1600',
            '211227_1600',
            '211220_1600',
            '211130_1800',
            '211113_1900',
            '211128_1800',
            '211129_1800',
            '211121_1800'
            ]


channels_combo = ['auto']#,'all']
beams = ['gaussian']#,'airy']
fit_routines = ['scipy_LS']
cuts = [None]#['sun_up','sun_down','moon_up','moon_down']
threshs = [0.03]#[0.003,0.03,0.3]

'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Single day runs with all params free ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for fit_routine in fit_routines:
    for beam in beams:
        for channels in channels_combo:
            for cut in cuts:
                for thresh in threshs:
                    out_dir = '/gpfs02/astro/workarea/fbianchin/bmx/geometry_fits/results_%s_%s_%s_cuts_%s_thresh_%s_amp'%(channels,beam,fit_routine,cut,str(thresh).replace('.','p'))

                    for data_id in data_ids:
                        fname = '%s_%s_%s_%s_%s_amp_%s.job' %(channels,beam,fit_routine,cut,str(thresh).replace('.','p'),data_id)
                        write_condor_job(fname, data_id, out_dir, channels, beam_type=beam, cuts=cut, thresh=thresh)
                        os.system('condor_submit '+fname)
'''
'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Single day runs with fixed beam params ~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


for fit_routine in fit_routines:
    for beam in beams:
        for channels in channels_combo:
            for cut in cuts:
                for thresh in threshs:
                    out_dir_main = '/gpfs02/astro/workarea/fbianchin/bmx/geometry_fits'
                    out_dir = out_dir_main +'/results_%s_%s_%s_cuts_%s_thresh_%s_amp_fixed_params'%(channels,beam,fit_routine,cut,str(thresh).replace('.','p'))

                    start_params = '/astro/u/fbianchin/Research/bmxobs/fits/Good15Days/means_13_days_%s_%s_%s_cuts_%s_thresh_%s_amp.pkl'%(channels,beam,fit_routine,cut,str(thresh).replace('.','p'))

                    for data_id in data_ids:
                        fname = '%s_%s_%s_%s_%s_amp_%s_fixed_params.job' %(channels,beam,fit_routine,cut,str(thresh).replace('.','p'),data_id)
                        write_condor_job(fname, data_id, out_dir, channels, beam_type=beam, cuts=cut, thresh=thresh, start_params=start_params)
                        os.system('condor_submit '+fname)
'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For multi-day run ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for thresh in threshs:
    out_dir = '/gpfs02/astro/workarea/fbianchin/bmx/geometry_fits/results_auto_gaussian_scipy_LS_cuts_None_thresh_%s_amp_15days'%(str(thresh).replace('.','p'))
    fname = 'auto_gaussian_scipy_LS_None_thresh_%s_amp_15days.job'%(str(thresh).replace('.','p'))
    start_params = '/astro/u/fbianchin/Research/bmxobs/fits/Good15Days/means_13_days_auto_gaussian_scipy_LS_cuts_None_thresh_%s_amp.pkl'%(str(thresh).replace('.','p'))
    write_condor_job(fname, " ".join(str(x) for x in data_ids), out_dir, 'auto', beam_type='gaussian', cuts=None, thresh=thresh, start_params=start_params)
    '''
    out_dir = '/gpfs02/astro/workarea/fbianchin/bmx/geometry_fits/results_auto_gaussian_scipy_LS_cuts_None_thresh_%s_amp_10days'%(str(thresh).replace('.','p'))
    fname = 'auto_gaussian_scipy_LS_None_thresh_%s_amp_10days.job'%(str(thresh).replace('.','p'))
    start_params = '/astro/u/fbianchin/Research/bmxobs/fits/Good15Days/means_13_days_auto_gaussian_scipy_LS_cuts_None_thresh_%s_amp.pkl'%(str(thresh).replace('.','p'))
    write_condor_job(fname, " ".join(str(x) for x in data_ids[:10]), out_dir, 'auto', beam_type='gaussian', cuts=None, thresh=thresh, start_params=start_params)
    '''
    os.system('condor_submit '+fname)

