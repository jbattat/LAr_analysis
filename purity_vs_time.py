import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import LAr_PrM as pm

# Goal: generate plot of purity vs. time for a set of runs
#       provided by the user
# Assumes that the waveform analysis has already been done
# so that there is a "reduced" HDF5 file


# FIXME: this can be an argument on the command line
# Name of file containing run names to be included in analysis
list_name = '7thRun_20250522_purity_vs_time.list'

# FIXME: this file list reader should go in the LAr_PrM library
flist = np.genfromtxt(list_name, dtype='str', comments='#', unpack=True)
print("Analyzing the following runs:")
print(flist)

# Open reduced data file
df_all = pm.read_reduced()

# Extract only the rows corresponding to the requested runs
df = df_all[ df_all[pm.COL_FILENAME].isin(flist)]

#print(f"len(df_all)) = {len(df_all)}")
#print(f"len(df)) = {len(df)}")
#print(f"len(flist) = {len(flist)}")
#print(df.columns)


# FIXME: need a more accurate lifetime estimator!
# As a crude estimator of lifetime, just take the ratio of the best-fit QA and QC...
tlife = pm.calc_lifetime(df) # us
times = pm.get_datetimes(df)

dts = (times - times[0]) 
dts_hr = dts.astype('timedelta64[s]').astype("int64")/3600 # hours since start

# PLOT RESULTS
#plt.plot(times, tlife, 'ko')
plt.plot(dts_hr, tlife, 'ko')
plt.xlabel("Time since start [hours]")
plt.ylabel("Lifetime [us]")
plt.title("VERY PRELIMINARY, VALUES NOT TRUSTWORTHY")
plt.ylim(ymin=0)
plt.savefig('junk.pdf')


