import sys
import warnings
from pathlib import Path
import time
from PyBoltzTestData import TestData
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# print(sys.path)
from PyBoltz import PyBoltz
import numpy as np
from tables import *


def test_type_two():

    D = TestData.getData()
    for ITest in range(1,D['NTests']+1):
        TN = 'T'+str(ITest)
        if D[TN + '/type'] == 2:

            Object = PyBoltz()
            # Set the number of gases
            Object.NumberOfGases = D[TN +'/Input/NumberOfGases']
            # Set the number of collisons
            Object.MaxNumberOfCollisions = D[TN +'/Input/MaxNumberOfCollisions']
            # Set penning
            Object.Enable_Penning =  D[TN +'/Input/Enable_Penning']
            # Calculate the electron energy
            Object.Enable_Thermal_Motion = D[TN +'/Input/Enable_Thermal_Motion']
            Object.Max_Electron_Energy =  D[TN +'/Input/Max_Electron_Energy']
            # Set the gas's with there given number
            Object.GasIDs =  D[TN +'/Input/GasIDs']
            # Set the gas fractions
            Object.GasFractions = D[TN +'/Input/GasFractions']
            # Set the tempature
            Object.TemperatureCentigrade = D[TN +'/Input/TemperatureCentigrade']
            # Set the pressure
            Object.Pressure_Torr = D[TN +'/Input/Pressure_Torr']
            # Set the eletric field
            Object.EField =  D[TN +'/Input/EField']
            # Set the magnetic field and angle
            Object.BField_Mag = 0.0 # first run...
            Object.BField_Angle = 0
            Object.Console_Output_Flag = D[TN +'/Input/Console_Output_Flag']
            Object.Steady_State_Threshold = D[TN +'/Input/Steady_State_Threshold']
            Object.Which_Angular_Model = D[TN +'/Input/Which_Angular_Model']
            Object.Start()

            Trans1 = Object.TransverseDiffusion

            Object = PyBoltz()
            # Set the number of gases
            Object.NumberOfGases = D[TN + '/Input/NumberOfGases']
            # Set the number of collisons
            Object.MaxNumberOfCollisions = D[TN + '/Input/MaxNumberOfCollisions']
            # Set penning
            Object.Enable_Penning = D[TN + '/Input/Enable_Penning']
            # Calculate the electron energy
            Object.Enable_Thermal_Motion = D[TN + '/Input/Enable_Thermal_Motion']
            Object.Max_Electron_Energy = D[TN + '/Input/Max_Electron_Energy']
            # Set the gas's with there given number
            Object.GasIDs = D[TN + '/Input/GasIDs']
            # Set the gas fractions
            Object.GasFractions = D[TN + '/Input/GasFractions']
            # Set the tempature
            Object.TemperatureCentigrade = D[TN + '/Input/TemperatureCentigrade']
            # Set the pressure
            Object.Pressure_Torr = D[TN + '/Input/Pressure_Torr']
            # Set the eletric field
            Object.EField = D[TN + '/Input/EField']
            # Set the magnetic field and angle
            Object.BField_Mag = D[TN + '/Input/BField_Mag']
            Object.BField_Angle = D[TN + '/Input/BField_Angle']
            Object.Console_Output_Flag = D[TN + '/Input/Console_Output_Flag']
            Object.Steady_State_Threshold = D[TN + '/Input/Steady_State_Threshold']
            Object.Which_Angular_Model = D[TN + '/Input/Which_Angular_Model']


            Object.Start()

            Trans = Object.TransverseDiffusion

            if D[TN + '/Comparisons'] == 1 or D[TN + '/Comparisons'] == 3:
                TransM = D[TN + '/Output/MBdtr']
                TransEM = D[TN + '/Output/MBdtrE']

                assert Trans1/Trans >= TransM - TransEM , "Test #: {}".format(ITest)
                assert Trans1/Trans <= TransM + TransEM, "Test #: {}".format(ITest)

            if D[TN + '/Comparisons'] == 2 or D[TN + '/Comparisons'] == 3:
                TransD = D[TN + '/Output/dtr']
                TransDE = D[TN + '/Output/dtrE']


                assert Trans1/Trans >= TransD - TransDE, "Test #: {}".format(ITest)
                assert Trans1/Trans <= TransD + TransDE, "Test #: {}".format(ITest)