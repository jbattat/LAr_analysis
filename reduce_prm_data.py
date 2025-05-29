# Goal: take as input cathode&anode waveforms from the Purity Monitor
# as well as a run "metadata" file (containing run params like Vc, Vag, Va, XeF reprate, etc.)
# and extract physical quantities (Qa and Qc) and log that information
# to an "output metadata" file
# That output file can then be used to compute impurity level / lifetime

import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

import LAr_PrM as pm

# Read in the run filenames and run parameters (Vc, Vag, Va, XeF reprate, etc.)
meta = pm.get_run_metadata(debug=False)

# Specify files to be analyzed
fnames = pm.get_files_to_analyze()   # fully qualified path
basenames = [os.path.basename(ff) for ff in fnames] # e.g. 20250523T093233.csv 

# Local directory where PrM data is stored
diagnostic_dir = pm.diagnostic_output_dir(verbose=True)

# Load waveforms into a list of DataFrames
dfs = []
ndata = len(fnames)
for ii, fname in enumerate(fnames):
    print(f"{ii:05d}/{ndata}: {fname}")
    dfs.append(pm.read_csv(fname, ch3='anode', ch4='cathode', tunit='us', vunit='mV'))

# remove mean of baseline for all waveforms
pm.subtract_baselines(dfs, chans=['cathode', 'anode'])


# Define fitting functions for cathode and anode signals
# FIXME: this should go in the LAr_PrM.py library
# FIXME: anode function could also include a "bump" at the beginning...
def anode_fxn(t, t0, td, tau, amp, Cf, k):
    exp_rise  = amp * (1-np.exp(-(t-t0)/tau)) * pm.sigmoid(t, t0, k) * (1-pm.sigmoid(t, t0+td, k))
    exp_fall  = amp * (np.exp(td/tau)-1)*np.exp(-(t-t0)/tau) * pm.sigmoid(t, t0+td, k)
    return exp_rise + exp_fall

# FIXME: this should go in the LAr_PrM.py library
def cathode_fxn(t, t0, td, tau, amp, Cf, k):
    exp_rise  = -amp * (1-np.exp(-(t-t0)/tau)) * pm.sigmoid(t, t0, k) * (1-pm.sigmoid(t, t0+td, k))
    exp_fall  = -amp * (np.exp(td/tau)-1)*np.exp(-(t-t0)/tau) * pm.sigmoid(t, t0+td, k)
    return exp_rise + exp_fall

# FIXME: should add a .guess() method for both to guess initial parameter values
# See e.g.: https://github.com/lmfit/lmfit-py/blob/master/lmfit/models.py
###    def guess(self, data, x, **kwargs):
###        """Estimate initial model parameter values from data."""
###        try:
###            sval, oval = np.polyfit(x, np.log(abs(data)+1.e-15), 1)
###        except TypeError:
###            sval, oval = 1., np.log(abs(max(data)+1.e-9))
###        pars = self.make_params(amplitude=np.exp(oval), decay=-1.0/sval)
###        return update_param_vals(pars, self.prefix, **kwargs)




# FIXME: preamp decay time is hard-coded at 140us...
# FIXME: preamp Cf is hardcoded at 1.4pF
# td is the time from the start to peak of the signal (~ risetime)

# Units:
# Q0    = pC
# t, td = us
# Cf    = pF

TAU_CATHODE = 135.0
TAU_ANODE = 133.0

def nominal_params_cathode():
    p = pm.Struct()
    p.t0 = 10.0 # us
    p.td = 10.0 # us
    p.tau = TAU_CATHODE # us
    p.amp = 200.0 # mV
    p.Cf = 1.4 # pF
    p.k = 1.0 # 1/us
    return p

def nominal_params_anode():
    p = pm.Struct()
    p.t0 = 50.0 # us
    p.td = 10.0 # us
    p.tau = TAU_ANODE # us
    p.amp = 10.0 # mV
    p.Cf = 1.4 # pF
    p.k = 1.0 # 1/us
    return p

def nominal_params(src):
    # src: waveform source ('cathode' or 'anode')
    #      case insensitive 'cathode' or 'CAthODe' are equivalent
    try:
        src = src.upper()
    except:
        print("nominal_params: invalid parameter src:")
        print(src)
        return None
    
    src = src.upper()
    match src:
        case 'CATHODE':
            p = nominal_params_cathode()
        case 'ANODE':
            p = nominal_params_anode()
        case _:
            p = None
            
    return p
        



# Cathode
model_cathode = Model(cathode_fxn, independent_vars=['t'])
model_anode = Model(anode_fxn, independent_vars=['t'])
# FIXME: implement "guess" as a method of the Model() (and move all to LAr_PrM)
guess_cathode = nominal_params(src='cathode')
guess_anode = nominal_params(src='anode')

# set up guess parameters
dt = dfs[0]['time'][1]-dfs[0]['time'][0]
kk = 5/dt
guess_cathode.k = kk
guess_anode.k = kk

params_cathode = model_cathode.make_params(t0=dict(value=guess_cathode.t0, min=0), # us
                                           td=dict(value=guess_cathode.td, min=0), # us
                                           tau=dict(value=guess_cathode.tau, min=130.0, max=145.0, vary=False), # us
                                           amp=dict(value=guess_cathode.amp, min=0), # mV
                                           Cf=dict(value=guess_cathode.Cf, vary=False), # pF
                                           k=dict(value=guess_cathode.k, vary=False), # 1/us (sigmoid transition rate)
                                           )
params_cathode.add('Q0', expr='amp*Cf*0.5*td/tau') # pC

params_anode = model_anode.make_params(t0=dict(value=guess_anode.t0, min=0),  # us
                                       td=dict(value=guess_anode.td, min=0),  # us
                                       tau=dict(value=guess_anode.tau, min=130.0, max=145.0, vary=False), # us
                                       amp=dict(value=guess_anode.amp, min=0), # mV
                                       Cf=dict(value=guess_anode.Cf, vary=False), # pF
                                       k=dict(value=guess_anode.k, vary=False), # 1/us (sigmoid transition rate)
                                       )
params_anode.add('Q0', expr='amp*Cf*0.5*td/tau') # pC


# FIXME: move this to LAr_PrM.py
# list of column names (these must match the parameter names in the Model()
CATHODE_PARAMS_TO_LOG = ['t0', 'td', 'tau', 'amp', 'Cf']  # Q0?
ANODE_PARAMS_TO_LOG = CATHODE_PARAMS_TO_LOG[:] # in general these lists can differ...
CATHODE_PREFIX = 'Cathode_'
ANODE_PREFIX = 'Anode_'

def get_wf_params(rc, ra, header=False):
    # get best-fit parameter values for a single waveform (cathode and anode)
    # can also get a header string

    # rc: ModelResult (e.g. result_cathode)
    # ra: ModelResult (e.g. result_anode)
    # header: if True, return just the header strings (as a list)
    #            used to make the header/columns for a DataFrame
    #         if False, then get the actual data values
    #            used to make the row entries in a DataFrame

    if header:
        # Need unique names for the output DataFrame
        # FIXME: could use "prefix" option in Model()...
        cp = [f'{CATHODE_PREFIX}{p}' for p in CATHODE_PARAMS_TO_LOG]
        ap = [f'{ANODE_PREFIX}{p}' for p in ANODE_PARAMS_TO_LOG]
        # List of "derived" quantities
        der = ['Qc', 'Qa']
        out = ['Filename']+cp+ap+der
    else:
        out = []
        # cathode values
        for p in CATHODE_PARAMS_TO_LOG:
            out.append(rc.params[p].value)
        # anode values
        for p in ANODE_PARAMS_TO_LOG:
            out.append(ra.params[p].value)
        # derived values (bespoke)
        out.append(rc.params['Q0'].value) # Qc
        out.append(ra.params['Q0'].value) # Qa
        
    return out  # list
    

# Make a list of lists (eventually convert to DataFrame)
# + first entry is the header (columns) for the DataFrame
# + each subsequent entry is a row of the output DataFrame
reduced_data = []
reduced_data.append(get_wf_params(None, None, header=True))

#for ii in range(2):  # just do 2 waveforms for now
for ii in range(len(dfs)):
    #print(f'\n ii={ii}')
    result_anode = model_anode.fit(dfs[ii]['anode'], params_anode, t=dfs[ii]['time'])
    #print("======== Anode =========")
    #print(result_anode.fit_report())

    result_cathode = model_cathode.fit(dfs[ii]['cathode'], params_cathode, t=dfs[ii]['time'])
    #print("\n======== Cathode =========")
    #print(result_cathode.fit_report())
    
    plt.plot(dfs[ii]['time'], dfs[ii]['anode'], linewidth=0.5, color=pm.acol)
    plt.plot(dfs[ii]['time'], dfs[ii]['cathode'], linewidth=0.5, color=pm.ccol)
    
    plt.plot(dfs[ii]['time'], result_anode.best_fit, linewidth=0.5, color='green', linestyle='--')
    plt.plot(dfs[ii]['time'], result_cathode.best_fit, linewidth=0.5, color='black', linestyle='--')

    dely = result_anode.eval_uncertainty(sigma=5)
    plt.fill_between(dfs[ii]['time'], result_anode.best_fit-dely, result_anode.best_fit+dely,
                     color="#ABABAB", label=r'5-$\sigma$ uncertainty band')
    
    fname = meta["Filename"][ii] # e.g. 20250523T093233.csv 
    froot = os.path.splitext(fname)[0]   # e.g. 20250523T093233 (splitext gives: ['20250523T093233', '.csv'])
    plt.title(fname)
    plt.savefig(os.path.join(diagnostic_dir, f'{froot}_wf_fit.pdf'))
    plt.clf()

    reduced_data.append([])
    reduced_data[-1] += [meta['Filename'][ii]]
    reduced_data[-1] += get_wf_params(result_cathode, result_anode, header=False)

# Convert list to a dataframe and store it
dout = pd.DataFrame(reduced_data[1:], columns=reduced_data[0])
print(dout)

# FIXME: save output to disk
    
