import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
print(sys.path)
from PyBoltz import PyBoltz
import numpy as np

CF4 = 1
He4 = 3
He3 = 4
Ne  = 5
Kr  = 6
Xe  = 7
CH4 = 8
ETHANE = 9
PROPANE = 10
ISOBUTANE = 11
CO2 = 12
H2O = 14
O2 = 15
N2 = 16
H2 = 21
DEUTERIUM = 22
DME = 24

GASES = [np.nan, 'CF4', 'ARGON', 'HELIUM4', 'HELIUM3', 'NEON', 'KRYPTON', 'XENON', 'CH4', 'ETHANE', 'PROPANE'
         , 'ISOBUTANE', 'CO2', np.nan, 'H2O', 'OXYGEN', 'NITROGEN', np.nan, np.nan, np.nan, np.nan
         , 'HYDROGEN', 'DEUTERIUM', np.nan, np.nan, 'DME']

Object = PyBoltz()

import time
t1 =time.time()

Args = sys.argv

efield     = int(Args[1])
OUT        = str(Args[2])
Pack       = int(Args[3])



# Set the number of gases
Object.NumberOfGases =1
# Set the number of collisons 
Object.MaxNumberOfCollisions =15*40000000.0
# Set penning
Object.EnablePenning = 0
# Calculate the electron energy
Object.EnableThermalMotion=1
Object.FinalElectronEnergy = 0.0
# Set the gas's with there given number
Object.GasIDs=[7,0,0,0,0,0]

# Set the gas fractions
Object.GasFractions=[100,0,0,0,0,0]
# Set the tempature
Object.TemperatureCentigrade = float(23)
# Set the pressure
Object.PressureTorr = 750.062
# Set the eletric field
Object.EField = efield
# Set the magnetic field and angle
Object.BFieldMag =0
Object.BFieldAngle =0
Object.ConsoleOutputFlag = 1
Object.WhichAngularModel = 2

if Pack == 1:
    # editing the gases.npy with the new cross sections
    gd = np.load('gases.npy').item()
    # The Y section
    gd['gas7/YMOM'] = [1.76000000e+02, 1.68000000e+02, 1.39000000e+02, 1.27000000e+02,
       1.16000000e+02, 1.02325000e+02, 8.86500000e+01, 7.49750000e+01,
       6.13000000e+01, 5.04000000e+01, 3.95000000e+01, 3.48666667e+01,
       3.02333333e+01, 2.56000000e+01, 2.04000000e+01, 1.76000000e+01,
       1.48000000e+01, 1.11000000e+01, 8.40000000e+00, 5.90000000e+00,
       4.86000000e+00, 3.30000000e+00, 2.96000000e+00, 2.45000000e+00,
       2.11000000e+00, 1.60000000e+00, 1.38600000e+00, 1.17200000e+00,
       9.58000000e-01, 7.44000000e-01, 5.30000000e-01, 5.15000000e-01,
       5.00000000e-01, 4.85000000e-01, 4.70000000e-01, 4.55000000e-01,
       4.40000000e-01, 4.25000000e-01, 4.10000000e-01, 3.95000000e-01,
       3.80000000e-01, 3.80000000e-01, 3.80000000e-01, 3.80000000e-01,
       3.80000000e-01, 3.80000000e-01, 3.80000000e-01, 3.80000000e-01,
       3.80000000e-01, 3.80000000e-01, 3.80000000e-01, 4.05500000e-01,
       4.31000000e-01, 4.65000000e-01, 4.99000000e-01, 5.50000000e-01,
       6.55000000e-01, 7.25000000e-01, 7.95000000e-01, 9.00000000e-01,
       1.15000000e+00, 1.45000000e+00, 1.67500000e+00, 1.90000000e+00,
       2.43333333e+00, 2.96666667e+00, 3.50000000e+00, 5.10000000e+00,
       7.50000000e+00, 1.15000000e+01, 1.65000000e+01, 2.05000000e+01,
       2.45000000e+01, 2.62500000e+01, 3.07000000e+01, 3.15000000e+01,
       3.23000000e+01, 3.16000000e+01, 3.10000000e+01, 2.78000000e+01,
       2.35000000e+01, 1.98000000e+01, 1.50000000e+01, 1.09000000e+01,
       8.40000000e+00, 7.25000000e+00, 5.65000000e+00, 5.00000000e+00,
       4.50000000e+00, 3.10000000e+00, 2.42000000e+00, 2.17000000e+00,
       2.00000000e+00, 1.89000000e+00, 1.80000000e+00, 1.73000000e+00,
       1.65000000e+00, 1.50000000e+00, 1.39000000e+00, 1.26000000e+00,
       1.09000000e+00, 9.40000000e-01, 8.40000000e-01, 7.50000000e-01,
       6.80000000e-01, 5.60000000e-01, 3.80000000e-01, 2.60000000e-01,
       1.55000000e-01, 1.05000000e-01, 7.60000000e-02, 5.90000000e-02,
       3.80000000e-02, 2.70000000e-02, 1.48000000e-02, 9.40000000e-03,
       5.00000000e-03, 3.10000000e-03, 2.20000000e-03, 1.63000000e-03,
       1.02400000e-03, 7.14000000e-04, 4.98000000e-04, 3.72000000e-04,
       2.91000000e-04, 2.36000000e-04, 1.66000000e-04, 1.25000000e-04,
       9.90000000e-05, 8.08000000e-05, 6.76000000e-05, 5.77000000e-05,
       4.38000000e-05, 3.48000000e-05, 2.85000000e-05, 2.39000000e-05,
       2.04000000e-05, 1.43000000e-05, 1.08000000e-05, 8.52000000e-06,
       6.91000000e-06, 4.85000000e-06, 3.62000000e-06, 2.81000000e-06,
       2.25000000e-06, 1.85000000e-06, 1.55000000e-06, 1.13000000e-06,
       8.67000000e-07, 6.86000000e-07, 5.58000000e-07, 4.63000000e-07,
       3.10000000e-07, 2.23000000e-07, 1.68000000e-07, 1.31000000e-07,
       8.64000000e-08, 6.11000000e-08, 4.54000000e-08, 3.51000000e-08,
       2.78000000e-08, 2.26000000e-08, 1.57000000e-08, 1.15000000e-08,
       8.79000000e-09, 6.93000000e-09, 5.60000000e-09, 3.57000000e-09,
       2.47000000e-09, 1.81000000e-09, 1.38000000e-09, 8.82000000e-10,
       6.11000000e-10, 4.48000000e-10, 3.43000000e-10, 2.71000000e-10,
       2.19000000e-10, 1.52000000e-10, 1.12000000e-10, 8.55000000e-11,
       6.75000000e-11, 5.47000000e-11]
    np.save("gases", gd)


Object.Start()

INFO = [
Object.GasFractions[1],
Object.PressureTorr,
Object.EField,
Object.VelocityZ,
Object.VelocityErrorZ,
Object.TransverseDiffusion,
Object.TransverseDiffusionError,
Object.LongitudinalDiffusion, 
Object.LongitudinalDiffusionError,
Object.TransverseDiffusion1,
Object.TransverseDiffusion1Error,
Object.LongitudinalDiffusion1,
Object.LongitudinalDiffusion1Error,
Object.MeanElectronEnergy,
Object.MeanElectronEnergyError]


INFO = np.array(INFO)
np.save(OUT,INFO)


t2 =time.time()

print("************************************************")
print("************************************************")
print("*****         Here are the outputs         *****")
print("************************************************ \n")
print("run time [s]= ",round(t2-t1,3))

for I in range(Object.NumberOfGases):
    if Object.GasIDs[I]<=25:
        print("Percentage of "+GASES[int(Object.GasIDs[I])]+" = "+  str(Object.GasFractions[I]))

print("Tempature [C]         = ", Object.TemperatureCentigrade)
print("Pressure [torr]       = ", Object.PressureTorr)
print("Eletric field [V/cm]  = ", Object.EField)
print("----------------------------------------------------")
print("Drift velocity [mm/mus]              = ", round(Object.VelocityZ,3))
print("----------------------------------------------------")
print("Drift velocity error [%]             = ", round(Object.VelocityErrorZ,3))
print("----------------------------------------------------")
print("Transverse diffusion [cm**2/s]       = ", round(Object.TransverseDiffusion,3))
print("----------------------------------------------------")
print("Transverse diffusion error [%]       = ", round(Object.TransverseDiffusionError,3))
print("----------------------------------------------------")
print("Longitudinal diffusion [cm**2/s]     = ", round(Object.LongitudinalDiffusion,3))
print("----------------------------------------------------")
print("Longitudinal diffusion error [%]     = ", round(Object.LongitudinalDiffusionError,3))
print("----------------------------------------------------")
print("Transverse diffusion [mum/cm**0.5]   = ", round(Object.TransverseDiffusion1,3))
print("----------------------------------------------------")
print("Transverse diffusion error [%]       = ", round(Object.TransverseDiffusion1Error,3))
print("----------------------------------------------------")
print("Longitudinal diffusion [mum/cm**0.5] = ", round(Object.LongitudinalDiffusion1,3))
print("----------------------------------------------------")
print("Longitudinal diffusion error [%]     = ", round(Object.LongitudinalDiffusion1Error,3))
print("----------------------------------------------------")
print("Diffusion X [cm**2/s]       = ", round(Object.DiffusionX,3))
print("----------------------------------------------------")
print("Diffusion X error [%]     = ", round(Object.ErrorDiffusionX,3))
print("----------------------------------------------------")
print("Diffusion Y [cm**2/s]       = ", round(Object.DiffusionY,3))
print("----------------------------------------------------")
print("Diffusion Y error [%]     = ", round(Object.ErrorDiffusionY,3))
print("----------------------------------------------------")
print("Diffusion Z [cm**2/s]       = ", round(Object.DiffusionZ,3))
print("----------------------------------------------------")
print("Diffusion Z error [%]     = ", round(Object.ErrorDiffusionZ,3))
print("----------------------------------------------------")
print("Diffusion YZ [cm**2/s]       = ", round(Object.DiffusionYZ,3))
print("----------------------------------------------------")
print("Diffusion YZ error [%]     = ", round(Object.ErrorDiffusionYZ,3))
print("----------------------------------------------------")
print("Diffusion XY [cm**2/s]       = ", round(Object.DiffusionXY,3))
print("----------------------------------------------------")
print("Diffusion XY error [%]     = ", round(Object.ErrorDiffusionXY,3))
print("----------------------------------------------------")
print("Diffusion XZ [cm**2/s]       = ", round(Object.DiffusionXZ,3))
print("----------------------------------------------------")
print("Diffusion XZ error [%]     = ", round(Object.ErrorDiffusionXZ,3))
print("----------------------------------------------------")
print("Mean electron energy [eV]            = ", round(Object.MeanElectronEnergy,3))
print("----------------------------------------------------")
print("Mean electron energy error [%]       = ", round(Object.MeanElectronEnergyError,3))
print("----------------------------------------------------")
print("Mean Collision Time [PicoSeconds]            = ", round(Object.MeanCollisionTime,3))
print("----------------------------------------------------")
print("************************************************")
print("************************************************")

