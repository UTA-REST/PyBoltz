import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from Magboltz import Magboltz
import numpy as np

Object = Magboltz()

import time
t1 =time.time()

# Set the number of gases
Object.NGAS =1
# Set the number of collisons 
Object.NMAX =4
# Set penning
Object.IPEN = 0
# Calculate the electron energy
Object.ITHRM=1
Object.EFINAL = 0.0
# Set the gas's with there given number
Object.NGASN=[7, 0, 0, 0, 0, 0]
# Set the gas fractions
Object.FRAC=[100, 0, 0, 0, 0, 0]
# Set the tempature
Object.TEMPC = float(23)
# Set the pressure
Object.TORR = 750.062
# Set the eletric field
Object.EFIELD =200
# Set the magnetic field and angle 
Object.BMAG = 0
Object.BTHETA =0

Object.Start()

t2 =time.time()

print("************************************************")
print("************************************************")
print("*****         Here are the outputs         *****")
print("************************************************ \n")
print("run time [s]= ",round(t2-t1,3))

print("Gas1")
print(str((Object.FRAC[0]))) #Gas1 Percentage
print("Gas2")    
print(str((Object.FRAC[1]))) #Gas2 Percentage

print("Tempature [C]         = ", Object.TEMPC)
print("Pressure [torr]       = ", Object.TORR)
print("Eletric field [V/cm]  = ", Object.EFIELD)
print("----------------------------------------------------")
print("Drift velocity [mm/mus]              = ", round((Object.WZ*1e-5),3))
print("----------------------------------------------------")
print("Drift velocity error [%]             = ", round(Object.DWZ,3))
print("----------------------------------------------------")
print("Transverse diffusion [cm**2/s]       = ", round(Object.DIFTR,3))
print("----------------------------------------------------")
print("Transverse diffusion error [%]       = ", round(Object.DFTER,3))
print("----------------------------------------------------")
print("Longitudinal diffusion [cm**2/s]     = ", round(Object.DIFLN,3))
print("----------------------------------------------------")
print("Longitudinal diffusion error [%]     = ", round(Object.DFLER,3))
print("----------------------------------------------------")
print("Transverse diffusion [mum/cm**0.5]   = ", round(Object.DTMN,3))
print("----------------------------------------------------")
print("Transverse diffusion error [%]       = ", round(Object.DFTER1,3))
print("----------------------------------------------------")
print("Longitudinal diffusion [mum/cm**0.5] = ", round(Object.DLMN,3))
print("----------------------------------------------------")
print("Longitudinal diffusion error [%]     = ", round(Object.DFLER1,3))
print("----------------------------------------------------")
print("Mean electron energy [eV]            = ", round(Object.AVE,3))
print("----------------------------------------------------")
print("Mean electron energy error [%]       = ", round(Object.DEN,3))
print("----------------------------------------------------")
print("************************************************")
print("************************************************")
