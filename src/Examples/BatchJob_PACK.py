import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
print(sys.path)
from PyBoltz import PyBoltz
import numpy as np


GASES = [np.nan, 'CF4', 'ARGON', 'HELIUM4', 'HELIUM3', 'NEON', 'KRYPTON', 'XENON', 'CH4', 'ETHANE', 'PROPANE'
         , 'ISOBUTANE', 'CO2', np.nan, 'H2O', 'OXYGEN', 'NITROGEN', np.nan, np.nan, np.nan, np.nan
         , 'HYDROGEN', 'DEUTERIUM', np.nan, np.nan, 'DME']

Object = PyBoltz()

import time
t1 =time.time()

Args = sys.argv

Scattering = int(Args[1])
efield     = float(Args[2])

a = str(Scattering)+"-"
b = str(efield)+".npy"

F_NAME = a+b
F_PATH = "/n/holystore01/LABS/guenette_lab/Users/amcdonald/Xenon_Dev/Outputs_PACK/"

# Pack     = 107
# mert     = 61
# pressure = 95

# Set the number of gases                                                                                                                                                           
Object.NumberOfGases =1
# Set the number of collisons                                                                                                                                                       
Object.MaxNumberOfCollisions = 1e10
# Set penning                                                                                                                                                                       
Object.Enable_Penning = 0
# Calculate the electron energy                                                                                                                                                     
Object.Enable_Thermal_Motion=1
Object.Max_Electron_Energy = 0.0
# Set the gas's with there given number                                                                                                                                             
Object.GasIDs=[107,0,0,0,0,0]
# Set the gas fractions                                                                                                                                                             
Object.GasFractions=[100,0,0,0,0,0]
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
Object.Steady_State_Threshold = 40
Object.Random_Seed = 54217137
Object.Swarm = 0

Object.Which_Angular_Model = Scattering


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
#Object.IonisationRate,
# Attachment rate                                                                                                                                                                   
#Object.AttachmentRate,
# steady state outputs                                                                                                                                                              
#Object.ALPHA,
#Object.ALPER,
#Object.ATT,
#Object.ATTER,
# Pulsed townsend                                                                                                                                                                   
#Object.ALPTEST
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
print("Mean electron energy [eV]            = ", round(Object.MeanElectronEnergy,3))
print("----------------------------------------------------")
print("Mean electron energy error [%]       = ", round(Object.MeanElectronEnergyError,3))
print("----------------------------------------------------")
print("Mean Collision Time [PicoSeconds]            = ", round(Object.MeanCollisionTime,3))
print("----------------------------------------------------")
print("************************************************")
print("************************************************")

