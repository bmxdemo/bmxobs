import sys,os

# def write_condor_job_multi(fn):
#     f=open(fn,'w')
#     f.write("""
# Universe        = vanilla
# Executable      = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/run_fitter.py
# Arguments       = -dataset $(words) -save_plots
# Requirements    = (CPU_Speed >= 1)
# Rank            = CPU_Speed
# request_memory  = 12000M
# request_cpus    = 2
# Priority        = 4
# GetEnv          = True
# Initialdir      = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/
# Input           = /dev/null 
# Output          = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testout/test.$(Cluster)_$(Process).out
# Error           = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testerr/test.$(Cluster)_$(Process).err
# Log             = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testlog/test.$(Cluster)_$(Process).log
#     """)
#     f.write ('Queue words from Good15DaysNovJan2022_federico.txt \n')
#     f.close()

def write_condor_job(fn, ds, out_dir, channels, beam_type='gaussian'):
    f=open(fn,'w')
    f.write("""
Universe        = vanilla
Executable      = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/run_fitter.py
Arguments       = -dataset %s -out_dir %s -channels %s -save_plots -beam_type %s
Requirements    = (CPU_Speed >= 1)
Rank            = CPU_Speed
request_memory  = 12000M
request_cpus    = 2
Priority        = 4
GetEnv          = True
Initialdir      = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/
Input           = /dev/null 
Output          = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testout/test.%s_%s.out
Error           = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testerr/test.%s_%s.err
Log             = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testlog/test.%s_%s.log
"""%(ds, out_dir, channels, beam_type, ds, channels, ds, channels, ds, channels))
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

channels_combo = ['all','auto']
beams = ['gaussian','airy']
fit_routines = ['scipy_LS']
cuts = ['sun_up','sun_down','moon_up','moon_down']#[None]

for fit_routine in fit_routines:
    for beam in beams:
        for channels in channels_combo:
            for cut in cuts:
                out_dir = '/astro/u/fbianchin/Research/bmxobs/fits/Good15Days/results_%s_%s_%s_cuts_%s_amp'%(channels,beam,fit_routine,cut)
                for data_id in data_ids:
                    fname = '%s_%s_%s_%s_amp_%s.job' %(channels,beam,fit_routine,cut,data_id)
                    write_condor_job(fname, data_id, out_dir, channels, beam_type=beam)
                    os.system('condor_submit '+fname)
