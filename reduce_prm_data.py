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

######################################################################
# Housekeeping: get data filenames, metadata, setup output dirs
######################################################################

# Which waveform data files to analyze?
fnames = pm.get_files_to_analyze()   # fully qualified path
basenames = [os.path.basename(ff) for ff in fnames] # e.g. 20250523T093233.csv 

# Read in the run parameters (Vc, Vag, Va, XeF reprate, etc.)
# WARNING: during analysis need to ensure that you pull the metadata for the correct run
# no guarantee that the order of basenames matches the order in the metadata
# (or even that there are the same number of entries... could have all data in the metadata but
# only choose to analyze a few files....)
meta = pm.get_run_metadata(debug=False)

# If any files are missing meta data then drop them
# FIXME: this should go into LAr_PrM.py
# FIXME: in fact, pm.get_files_to_analyze() should do this check
#        and only return analyzable files (i.e. files with metadata)...

# Find the bad files (missing metadata) ...
badfiles = []
for fname in basenames:
    if not meta[pm.COL_FILENAME].eq(fname).any():
        badfiles.append(fname)
# ... and remove from list of files to analyze
basenames = [ff for ff in basenames if ff not in badfiles]
fnames = [ff for ff in fnames if os.path.basename(ff) not in badfiles]

# Output dir for plots of waveforms and fits
diagnostic_dir = pm.diagnostic_output_dir(verbose=True)

######################################################################
# Read and pre-process the data
######################################################################

# Load waveforms into a list of DataFrames
dfs = pm.get_waveform_data(fnames) 
# remove mean of baseline for all waveforms
pm.subtract_baselines(dfs, chans=['cathode', 'anode'])

######################################################################
# Setup the models for fitting to the cathode and anode signals
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


######################################################################
# Do the analysis (waveform fitting)
######################################################################
# FIXME: should also save the parameter uncertainties

# Fit results stored in a list of lists (eventually convert to DataFrame)
# + first entry is the header (columns) for the DataFrame
# + each subsequent entry is a row of the output DataFrame
reduced_data = []

# Populate the header information
reduced_data.append(pm.get_reduced_header())
#print(reduced_data[0])

for ii in range(len(dfs)):
#for ii in range(3):
    result_anode = model_anode.fit(dfs[ii]['anode'], params_anode, t=dfs[ii]['time'])
    result_cathode = model_cathode.fit(dfs[ii]['cathode'], params_cathode, t=dfs[ii]['time'])
    
    plt.plot(dfs[ii]['time'], dfs[ii]['anode'], linewidth=0.5, color=pm.ACOL)
    plt.plot(dfs[ii]['time'], dfs[ii]['cathode'], linewidth=0.5, color=pm.CCOL)
    
    plt.plot(dfs[ii]['time'], result_anode.best_fit, linewidth=0.5, color='green', linestyle='--')
    plt.plot(dfs[ii]['time'], result_cathode.best_fit, linewidth=0.5, color='black', linestyle='--')

    dely = result_anode.eval_uncertainty(sigma=5)
    plt.fill_between(dfs[ii]['time'], result_anode.best_fit-dely, result_anode.best_fit+dely,
                     color="#ABABAB", label=r'5-$\sigma$ uncertainty band')
    
    fname = basenames[ii] # e.g. 20250523T093233.csv 
    froot = os.path.splitext(fname)[0]   # e.g. 20250523T093233 (splitext gives: ['20250523T093233', '.csv'])
    plt.title(fname)
    plt.savefig(os.path.join(diagnostic_dir, f'{froot}_wf_fit.pdf'))
    plt.clf()
    
    reduced_data.append([])
    reduced_data[-1] += [fname]
    reduced_data[-1] += pm.get_meta_params(fname, meta)
    reduced_data[-1] += pm.get_wf_params(result_cathode, result_anode)

# Convert list to a dataframe and store it
dout = pd.DataFrame(reduced_data[1:], columns=reduced_data[0])
print(dout)

# FIXME: save output to disk
pm.save_reduced(dout)    
