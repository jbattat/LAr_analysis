# test functions in the LAr.py library

import LAr
import matplotlib.pyplot as plt
import numpy as np

EE = np.logspace(0, 4, 100)  # V/cm
TT = 87.0 # K
mu = LAr.mobility(TT, EE) # cm^2/V/s

plt.loglog(EE*1e-3, mu) 
plt.xlabel("E [kV/cm]")
plt.ylabel("mu [cm^2/V/s]")
plt.savefig('lar_test_mobility.pdf')
plt.clf()

EE = np.linspace(10, 3000, 100) # V/cm
print(LAr.mobility(87.3, 160))
print(LAr.mobility(89.3, 160))
print(LAr.mobility(91.3, 160))
print("Electron drift speed")
print(LAr.electron_drift_speed(LAr.mobility(87.3, 160), 160))

v873 = LAr.electron_drift_speed(LAr.mobility(87.3, EE), EE)
v893 = LAr.electron_drift_speed(LAr.mobility(89.3, EE), EE)
v913 = LAr.electron_drift_speed(LAr.mobility(91.3, EE), EE)
plt.plot(EE*1e-3, v873, 'b-', label='87.3K')
plt.plot(EE*1e-3, v893, 'g-', label='89.3K')
plt.plot(EE*1e-3, v913, 'r-', label='91.3K')
plt.xlabel("Electric Field [kV/cm]")
plt.ylabel("Electron drift speed [cm/us]")
plt.savefig('lar_test_electron_drift_speed.pdf')


