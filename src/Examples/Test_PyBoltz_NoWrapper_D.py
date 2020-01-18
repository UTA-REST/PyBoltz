import sys
import warnings
from pathlib import Path
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
print(sys.path)
from PyBoltz import PyBoltz
import numpy as np

exec(Path("../Cython/Setup_npy.py").read_text())                                # Setup gases

CF4       = 1
Ar        = 2
He4       = 3
He3       = 4
Ne        = 5
Kr        = 6
Xe        = 7
CH4       = 8
ETHANE    = 9
PROPANE   = 10
ISOBUTANE = 11
CO2       = 12
H2O       = 14
O2        = 15
N2        = 16
H2        = 21
DEUTERIUM = 22
DME       = 24

GASES = [np.nan, 'CF4', 'ARGON', 'HELIUM4', 'HELIUM3', 'NEON', 'KRYPTON', 'XENON', 'CH4', 'ETHANE', 'PROPANE'
         , 'ISOBUTANE', 'CO2', np.nan, 'H2O', 'OXYGEN', 'NITROGEN', np.nan, np.nan, np.nan, np.nan
         , 'HYDROGEN', 'DEUTERIUM', np.nan, np.nan, 'DME']

Object = PyBoltz()

t1 = time.time()

Object.NumberOfGases         = 2                                                # Set the number of gases
Object.MaxNumberOfCollisions = 1*40000000.0                                     # Set the number of collisions 
Object.EnablePenning         = 0                                                # Set Penning effect
Object.EnableThermalMotion   = 1                                                # Enabling thermal motion
Object.FinalElectronEnergy   = 1.4                                              # Calculate the electron energy
Object.GasIDs                = [2, 8, 0, 0, 0, 0]                               # Set the gases with their given number
Object.GasFractions          = [90, 10, 0, 0, 0, 0]                             # Set the gas fractions
Object.TemperatureCentigrade = float(23)                                        # Set the temperature
Object.PressureTorr          = 750.062                                          # Set the pressure
Object.EField                = 80                                               # Set the electric field
Object.BFieldMag             = 0                                                # Set the magnetic field
Object.BFieldAngle           = 0                                                # Set angle between magnetic and electric field
Object.WhichAngularModel     = 2                                                # Set angular model
Object.ConsoleOutputFlag     = 1

Object.Start()

t2 = time.time()

print("************************************************"   )
print("************************************************"   )
print("*****         Here are the outputs         *****"   )
print("************************************************ \n")
print("run time [s]= ",round(t2-t1,3))

for I in range(Object.NumberOfGases):
    if Object.GasIDs[I]<=25:
        print("Percentage of " + GASES[int(Object.GasIDs[I])] + " = " + str(Object.GasFractions[I]))

print("Temperature [C]        = ", Object.TemperatureCentigrade)
print("Pressure [torr]        = ", Object.PressureTorr)
print("Electric field [V/cm]  = ", Object.EField)
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
print(" ")
print("Diego printing Python code for the first time!!")
print(" ")

