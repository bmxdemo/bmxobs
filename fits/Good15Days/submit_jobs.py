import sys,os

def write_condor_job(fn):
    f=open(fn,'w')
    f.write("""
    Universe        = vanilla
    Executable      = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/run_fitter.py
    Arguments       = -dataset $(words) -save_plots
    Requirements    = (CPU_Speed >= 1)
    Rank            = CPU_Speed
    request_memory  = 12000M
    request_cpus    = 2
    Priority        = 4
    GetEnv          = True
    Initialdir      = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/
    Input           = /dev/null 
    Output          = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testout/test.$(Cluster)_$(Process).out
    Error           = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testerr/test.$(Cluster)_$(Process).err
    Log             = /astro/u/fbianchin/Research/bmxobs/fits/Good15Days/Condor/testlog/test.$(Cluster)_$(Process).log
    """)
    f.write ('Queue words from Good15DaysNovJan2022_federico.txt \n')
    f.close()
    
fname = '15gooddays.job'
write_condor_job(fname)
os.system('condor_submit '+fname)