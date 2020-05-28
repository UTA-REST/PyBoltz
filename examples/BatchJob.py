import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
print(sys.path)
from PyBoltz.Boltz import Boltz
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

Object = Boltz()

import time
t1 =time.time()

Args = sys.argv

this_Gas   = int(Args[1])
this_Gas_frac   = float(Args[2])
that_Gas   = int(Args[3])
that_Gas_frac = float(Args[4])
efield     = int(Args[5])

a = str(that_Gas)+"-"
b = '{0:.7f}'.format(that_Gas_frac)+"-"
c = str(efield)+".npy"
F_NAME = a+b+c
F_PATH = str(Args[6])


# Set the number of gases
Object.NumberOfGases =2
# Set the number of collisons 
Object.MaxNumberOfCollisions =10*40000000.0
# Set penning
Object.Enable_Penning = 0
# Calculate the electron energy
Object.Enable_Thermal_Motion=1
Object.Max_Electron_Energy = 0.0
# Set the gas's with there given number
Object.GasIDs=[this_Gas,that_Gas,0,0,0,0]

# Set the gas fractions
Object.GasFractions=[this_Gas_frac,that_Gas_frac,0,0,0,0]
# Set the tempature
Object.TemperatureCentigrade = float(23)
# Set the pressure
Object.Pressure_Torr = 750.062
# Set the eletric field
Object.EField = efield
# Set the magnetic field and angle
Object.BField_Mag =0
Object.BField_Angle =0
Object.Console_Output_Flag = 1

Object.Which_Angular_Model = 2

Object.Start()

INFO = [
Object.GasFractions[1],
Object.Pressure_Torr,
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
Object.MeanElectronEnergyError,
# Ionzitation rate                                                                                                                                                                  
Object.IonisationRate,
# Attachment rate                                                                                                                                                                   
Object.AttachmentRate,
# steady state outputs                                                                                                                                                              
Object.ALPHA,
Object.ALPER,
Object.ATT,
Object.ATTER,
# Pulsed townsend                                                                                                                                                                   
Object.ALPTEST
]


INFO = np.array(INFO)
np.save(F_PATH+F_NAME,INFO)


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
print("Pressure [torr]       = ", Object.Pressure_Torr)
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

