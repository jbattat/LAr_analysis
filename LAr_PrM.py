# Code for analyzing purity monitor data
# Wellesley College purity monitor (PrM)

import numpy as np
import pandas as pd
import os

acol = 'red'  # color for anode waveforms
ccol = 'blue' # color for cathode waveforms

# Column names for metadata
META_COL_FILE = 'Filename' # Filename
META_COL_VC = 'Vc [V]'     # Cathode HV
META_COL_VAG = 'Vag [V]'   # Anode grid HV
META_COL_VA = 'Va [V]'     # Anode HV

def sigmoid(t, t0, k):
    # https://en.wikipedia.org/wiki/Logistic_function
    # function that transitions smoothly from zero to 1
    # t0: location of the transition (value of sigmoid is 0.5 there)
    # k:  "growth rate" (width of transition region from 0 to 1)
    #     (can think of k as 1/"sigma" where sigma is the width of the transition)
    return 1/(1+np.exp(-k*(t-t0)))

def data_dir():
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

    return dataDir

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

