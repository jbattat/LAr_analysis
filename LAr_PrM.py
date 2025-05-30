# Code for analyzing purity monitor data
# Wellesley College purity monitor (PrM)

import numpy as np
import pandas as pd
import os
from glob import glob

from lmfit import Model

ACOL = 'red'  # color for anode waveforms
CCOL = 'blue' # color for cathode waveforms

# Decay time of cathode and anode preamps
TAU_CATHODE = 135.0
TAU_ANODE = 133.0

# Column names for metadata
META_COL_FILE = 'Filename' # Filename
META_COL_VC = 'Vc [V]'     # Cathode HV
META_COL_VAG = 'Vag [V]'   # Anode grid HV
META_COL_VA = 'Va [V]'     # Anode HV

# Default name of reduced data file        
REDUCED_DATA_OUTPUT_FILENAME = 'PrM_Reduced.h5'
REDUCED_HDF5_KEY = 'reduced'

# List of column names (these must match the parameter names in the Model()
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
    

def get_waveform_data(fnames):
    # return a list of Pandas DataFrames, each containing time, anode and cathode waveform data
    # fnames is a list of filenames to read from (list of .csv file names) -- full path
    dfs = []
    ndata = len(fnames)
    for ii, fname in enumerate(fnames):
        print(f"{ii+1:05d}/{ndata}: {fname}")
        dfs.append(read_csv(fname, ch3='anode', ch4='cathode', tunit='us', vunit='mV'))
    return dfs

########################################################
# Fitting functions / models
########################################################
def cathode_fxn(t, t0, td, tau, amp, Cf, k):
    exp_rise  = -amp * (1-np.exp(-(t-t0)/tau)) * sigmoid(t, t0, k) * (1-sigmoid(t, t0+td, k))
    exp_fall  = -amp * (np.exp(td/tau)-1)*np.exp(-(t-t0)/tau) * sigmoid(t, t0+td, k)
    return exp_rise + exp_fall

def get_cathode_model():
    return Model(cathode_fxn, independent_vars=['t'])

# FIXME: anode function could also include a "bump" at the beginning...
def anode_fxn(t, t0, td, tau, amp, Cf, k):
    exp_rise  = amp * (1-np.exp(-(t-t0)/tau)) * sigmoid(t, t0, k) * (1-sigmoid(t, t0+td, k))
    exp_fall  = amp * (np.exp(td/tau)-1)*np.exp(-(t-t0)/tau) * sigmoid(t, t0+td, k)
    return exp_rise + exp_fall

def sigmoid(t, t0, k):
    # https://en.wikipedia.org/wiki/Logistic_function
    # function that transitions smoothly from zero to 1
    # t0: location of the transition (value of sigmoid is 0.5 there)
    # k:  "growth rate" (width of transition region from 0 to 1)
    #     (can think of k as 1/"sigma" where sigma is the width of the transition)
    return 1/(1+np.exp(-k*(t-t0)))

def get_anode_model():
    return Model(anode_fxn, independent_vars=['t'])

def get_fitting_models():
    return [get_cathode_model(), get_anode_model()]

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

# td is the time from the start to peak of the signal (~ risetime)

# Units:
# Q0    = pC
# t, td = us
# Cf    = pF

def nominal_params_cathode():
    p = Struct()
    p.t0 = 10.0 # us
    p.td = 10.0 # us
    p.tau = TAU_CATHODE # us
    p.amp = 200.0 # mV
    p.Cf = 1.4 # pF
    p.k = 1.0 # 1/us
    return p

def nominal_params_anode():
    p = Struct()
    p.t0 = 50.0 # us
    p.td = 10.0 # us
    p.tau = TAU_ANODE # us
    p.amp = 10.0 # mV
    p.Cf = 1.4 # pF
    p.k = 1.0 # 1/us
    return p

def get_nominal_params(src=None):
    # src: waveform source ('cathode' or 'anode')
    #      case insensitive 'cathode' or 'CAthODe' are equivalent
    if src is None:
        return [nominal_params_cathode(), nominal_params_anode()]
        
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
        
def make_params(model, guess, src=None):
    src = src.upper()
    match src:
        case 'CATHODE':
            pp = model.make_params(t0=dict(value=guess.t0, min=0), # us
                                   td=dict(value=guess.td, min=0), # us
                                   tau=dict(value=guess.tau, min=130.0, max=145.0, vary=False), # us
                                   amp=dict(value=guess.amp, min=0), # mV
                                   Cf=dict(value=guess.Cf, vary=False), # pF
                                   k=dict(value=guess.k, vary=False), # 1/us (sigmoid transition rate)
                                   )
            pp.add('Q0', expr='amp*Cf*0.5*td/tau') # pC
        case 'ANODE':  # in principle the models can have different sets of parameters
            pp = model.make_params(t0=dict(value=guess.t0, min=0), # us
                                   td=dict(value=guess.td, min=0), # us
                                   tau=dict(value=guess.tau, min=130.0, max=145.0, vary=False), # us
                                   amp=dict(value=guess.amp, min=0), # mV
                                   Cf=dict(value=guess.Cf, vary=False), # pF
                                   k=dict(value=guess.k, vary=False), # 1/us (sigmoid transition rate)
                                   )
            pp.add('Q0', expr='amp*Cf*0.5*td/tau') # pC
        case _:
            pp = None

    return pp


def data_dir(verbose=False):
    # Get the location of PrM data (set by environment variable)
    # User should either set this environment variable in their .bashrc file
    # (or similar), or set the env variable at the command line prior to running the analysis
    # e.g. on mac:
    #  $ export PRM_DATA_DIR='/Users/jbattat/research/qpix/purity_monitor_data'
    # if the environment variable cannot be found by ptyhon then the current directory is used
    
    try:
        dataDir = os.environ['PRM_DATA_DIR']
    except: # use the current directory if the environment variable is not set
        dataDir = os.path.join(os.getcwd(), 'data')

    if verbose:
        print(f"Data will be read from: {dataDir}")
    return dataDir

def diagnostic_output_dir(verbose=True):
    try:
        dataDir = os.environ['PRM_DIAGNOSTIC_DIR']
    except: # use the current directory if the environment variable is not set
        dataDir = os.path.join(os.getcwd(), 'diagnostic_output')

    if verbose:
        print(f"Diagnostic outputs will be saved to: {dataDir}")

    return dataDir


def get_files_to_analyze():
    ddir = data_dir(verbose=True)
    glob_str = os.path.join(ddir, '2025*.csv')
    fnames = glob(glob_str)
    fnames = sorted(fnames)
    return fnames


def get_run_metadata(fname='LAr_Purity_Monitor_Runs_All.csv', debug=True):
    # Read in table of run metadata (containing Vc, Vag, Va etc.)
    # 
    # fname should be the name of the .csv file generated from:
    # https://docs.google.com/spreadsheets/d/1Kji3os3iBWxYyT-fSda9Bmrm8J0q-8kCbXKnLPBlu7A/edit?gid=0#gid=0
    # debug:
    #
    # return: dataframe containing information in the .csv file
    
    metadata_file = os.path.join(data_dir(), fname)
    df = pd.read_csv(metadata_file, comment='#')
    if debug:
        print(df.head)
        print(df.columns)
    return df

def get_hv_of_fname(fname, df, source='all'):
    # return the PrM HV value(s) for a given dataset
    # source can be C, AG, A or ALL (Cathode, Anode Grid, Anode, or all 3 returned as a list [Vc, Vag, Va])

    cond = (df[META_COL_FILE] == fname)
    # FIXME: handle case where the requested file is not present in the dataframe
    
    source = source.upper()
    if source not in ['C', 'AG', 'A', 'ALL']:
        print("Error: invalid HV source specified")
        return
    
    HV_COLS = {'C':META_COL_VC,
               'AG':META_COL_VAG,
               'A':META_COL_VA}
    if source == 'ALL':
        out = [df[cond][META_COL_VC].values[0],
               df[cond][META_COL_VAG].values[0],
               df[cond][META_COL_VA].values[0]]
    else:
        col = HV_COLS[source]
        out = df[cond][col].values[0]

    return out

########## originally these functions were in RigolTools.py
# but that's a bad name -- they are not tools for interacting with
# the rigol scope... they are tools for analyzing waveforms...


def find_baseline(series, npts=50):
    return series[0:npts].mean()
    
def subtract_baseline(df, chans=['anode','cathode']):
    # subtract the baselines for a single dataframe (a single PrM datafile)
    for chan in chans:
        df[chan] -= find_baseline(df[chan]) 

def subtract_baselines(dfs, chans=['anode','cathode']):
    # subtract the baselines for df in a list of dataframes (several PrM data files)
    for df in dfs:
        subtract_baseline(df, chans=chans)

def df_from_csv(fname):
    df = pd.read_csv(fname)
    return df

def read_csv(fname, ch3=None, ch4=None, ch1=None, ch2=None, vunit='mV', tunit='us'):
    # FIXME: add check that file exists

    #e.g. rigol.readCsv(filename, ch3='cathode', ch4='anode', vunit='mV', tunit='us')
    #returns
    #      time   cathode     anode
    #0   -337.0  64.73467  75.52400
    #1   -336.0  64.40639  75.42400
    
    # if doing this manually...
    # first line is a header
    # e.g.
    #    Time(s),CH3V,CH4V
    #    -3.370000e-04,6.473467e-02,7.552400e-02
    #    -3.360000e-04,6.440639e-02,7.542400e-02
    # expect to have time and then channel data
    #with open(fname) as fh:
    #    header = fh.readline()
    #    print(header)

    # or just use pandas...
    df = df_from_csv(fname)

    # rename the time column
    df.rename(columns={"Time(s)":"time"}, inplace=True)
    voltage_labels = [] # track which channels are "in play"
    if ch1:
        df.rename(columns={"CH1V":ch1}, inplace=True)
        voltage_labels.append(ch1)
    if ch2:
        df.rename(columns={"CH2V":ch2}, inplace=True)
        voltage_labels.append(ch2)
    if ch3:
        df.rename(columns={"CH3V":ch3}, inplace=True)
        voltage_labels.append(ch3)
    if ch4:
        df.rename(columns={"CH4V":ch4}, inplace=True)
        voltage_labels.append(ch4)

    # If requested, then:
    # scale time from seconds to microseconds
    if tunit == 'us':
        df['time'] *= 1e6
    # and voltages to mV
    if vunit == 'mV':
        for vl in voltage_labels:
            df[vl] *= 1e3

    return df

# Helper class to hold key/val pairs with dot access
class Struct(dict):
    # Usage:
    #    obj = Struct()
    #    obj.somefield = "somevalue"
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def reduced_dir(verbose=False):
    try:
        rDir = os.environ['PRM_REDUCED_DIR']
    except: # use the current directory if the environment variable is not set
        rDir = os.path.join(os.getcwd(), 'analysis_output')

    if verbose:
        print(f"Reduced data stored in {rDir}")
        
    return rDir


def save_reduced(df, fname=REDUCED_DATA_OUTPUT_FILENAME):
    df.to_hdf(os.path.join(reduced_dir(), REDUCED_DATA_OUTPUT_FILENAME), key=REDUCED_HDF5_KEY)

def read_reduced(fname=REDUCED_DATA_OUTPUT_FILENAME):
    return pd.read_hdf(os.path.join(reduced_dir(), fname), key=REDUCED_HDF5_KEY)

