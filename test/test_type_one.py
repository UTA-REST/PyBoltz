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


def test_type_one():

    D = TestData.getData()
    for ITest in range(1,D['NTests']+1):
        TN = 'T'+str(ITest)
        if D[TN + '/type'] == 1:

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
            Object.BField_Mag = D[TN +'/Input/BField_Mag']
            Object.BField_Angle = D[TN +'/Input/BField_Angle']
            Object.Console_Output_Flag = D[TN +'/Input/Console_Output_Flag']
            Object.Steady_State_Threshold = D[TN +'/Input/Steady_State_Threshold']
            Object.Which_Angular_Model = D[TN +'/Input/Which_Angular_Model']

            Object.Start()

            if D[TN + '/Comparisons'] == 1 or D[TN + '/Comparisons'] == 3:
                DriftVelocityM = D[TN + '/Output/MBvel']
                DriftVelocityErrM = D[TN + '/Output/MBvelE']

                LongM = D[TN + '/Output/MBdl']
                LongEM =  D[TN + '/Output/MBdlE'] 

                TransM = D[TN + '/Output/MBdt']
                TransEM = D[TN + '/Output/MBdtE']

                assert Object.VelocityZ >= DriftVelocityM - DriftVelocityErrM, "Test #: {}".format(ITest)
                assert Object.VelocityZ <= DriftVelocityM + DriftVelocityErrM, "Test #: {}".format(ITest)

                assert Object.LongitudinalDiffusion1 >= LongM - LongEM, "Test #: {}".format(ITest)
                assert Object.LongitudinalDiffusion1 <= LongM + LongEM, "Test #: {}".format(ITest)

                assert Object.TransverseDiffusion1 >= TransM - TransEM, "Test #: {}".format(ITest)
                assert Object.TransverseDiffusion1 <= TransM + TransEM, "Test #: {}".format(ITest)

            if D[TN + '/Comparisons'] == 2 or D[TN + '/Comparisons'] == 3:
                DriftVelocity = D[TN + '/Output/vel']
                DriftVelocityErr = D[TN + '/Output/velE']

                Long = D[TN + '/Output/dl']
                LongE =  D[TN + '/Output/dlE']

                Trans = D[TN + '/Output/dt']
                TransE = D[TN + '/Output/dtE']

                assert Object.VelocityZ >= DriftVelocity - DriftVelocityErr, "Test #: {}".format(ITest)
                assert Object.VelocityZ <= DriftVelocity + DriftVelocityErr, "Test #: {}".format(ITest)

                assert Object.LongitudinalDiffusion1 >= Long - LongE, "Test #: {}".format(ITest)
                assert Object.LongitudinalDiffusion1 <= Long + LongE, "Test #: {}".format(ITest)

                assert Object.TransverseDiffusion1 >= Trans - TransE, "Test #: {}".format(ITest)
                assert Object.TransverseDiffusion1 <= Trans + TransE, "Test #: {}".format(ITest)