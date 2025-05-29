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

import LAr_PrM as pm

# Read in the run filenames and run parameters (Vc, Vag, Va, XeF reprate, etc.)
meta = pm.get_run_metadata(debug=False)

# Specify files to be analyzed
fnames = pm.get_files_to_analyze()   # fully qualified path
basenames = [os.path.basename(ff) for ff in fnames] # e.g. 20250523T093233.csv 

# Local directory where PrM data is stored
diagnostic_dir = pm.diagnostic_output_dir(verbose=True)

# Load waveforms into a list of DataFrames
dfs = pm.get_waveform_data(fnames) 

# remove mean of baseline for all waveforms
pm.subtract_baselines(dfs, chans=['cathode', 'anode'])

######################################################################
# Get fitting functions for cathode and anode signals
######################################################################
model_cathode, model_anode = pm.get_fitting_models()
guess_cathode, guess_anode = pm.get_nominal_params()

# Tweak some guess parameters
# FIXME: maybe the sigmoid rise rate should not be a model parameter?
# or should be changed to an independent variable?
dt = dfs[0]['time'][1]-dfs[0]['time'][0]
kk = 5/dt
guess_cathode.k = kk
guess_anode.k = kk

# Initialize the parameters
# FIXME: this should be done dynamically for each dataset (based on the data)
# via the "guess" method of the LMFIT model...
params_cathode = pm.make_params(model_cathode, guess_cathode, src='CATHODE')
params_anode = pm.make_params(model_anode, guess_anode, src='ANODE')


# Make a list of lists (eventually convert to DataFrame)
# + first entry is the header (columns) for the DataFrame
# + each subsequent entry is a row of the output DataFrame
reduced_data = []
reduced_data.append(pm.get_wf_params(None, None, header=True))

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
    reduced_data[-1] += pm.get_wf_params(result_cathode, result_anode, header=False)

# Convert list to a dataframe and store it
dout = pd.DataFrame(reduced_data[1:], columns=reduced_data[0])
print(dout)

# FIXME: save output to disk
    
