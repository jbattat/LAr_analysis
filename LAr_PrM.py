# Code for analyzing purity monitor data
# Wellesley College purity monitor (PrM)

import os

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
        dataDir = os.getcwd() 

    return dataDir
    
