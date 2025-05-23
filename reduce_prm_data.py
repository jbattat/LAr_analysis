import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ExpressionModel

import LAr_PrM as pm

# FIXME: this should go in the LAr_PrM.py library
vmod = ExpressionModel(" (-2*(1e3*Q0*140)/(td*1.4)) * ( (1-exp(-x/140))*(x<td) + (exp(td/140)-1)*(exp(-x/140))*(x>=td) )")

# Local directory where PrM data is stored
data_dir = pm.data_dir()

meta = pm.get_run_metadata()

# Specify files to be analyzed
basenames = ['20250522T131342.csv',
             '20250522T145313.csv']
fnames = [os.path.join(data_dir, base) for base in basenames]
print(basenames)
print(fnames)

dfs = []
for ii, fname in enumerate(fnames):
    print(f"ii, fname = {ii}, {fname}")
    dfs.append(pm.read_csv(fname, ch3='anode', ch4='cathode', tunit='us', vunit='mV'))
    #basename = os.path.basename(fname)
    #dfs[-1].attrs['vcath'] = pm.get_hv_of_fname(basename, meta, source='C')

# remove mean of baseline
pm.subtract_baselines(dfs, chans=['cathode', 'anode'])

#ii = 1
#plt.plot(dfs[ii]['time'], dfs[ii]['cathode'], linewidth=0.5, color=pm.ccol)
#plt.plot(dfs[ii]['time'], dfs[ii]['anode'], linewidth=0.5, color=pm.acol)
#plt.savefig('junk.pdf')


sys.exit()


    
# just do 3 waveforms for now
for ii in range(3):#len(dfs)):
    td = 10.0 # us
    Cf = 1.4
    vgain = 2.0
    # FIXME: model assumes peak starts at first entry, but not true for data!
    Q0 = np.max(np.abs(dfs[ii]['cathode'].values))*Cf*td/vgain
    result = vmod.fit(dfs[ii]['cathode'], x=dfs[ii]['time'], Q0=Q0, td=td)
    print(result.fit_report())
    plt.plot(dfs[ii]['time'], dfs[ii]['cathode'], linewidth=0.5, color='black')
    plt.plot(dfs[ii]['time'], result.best_fit, linewidth=0.5, color='red', linestyle='--')
    plt.savefig(f'junk_{ii}.pdf')
    plt.clf()
