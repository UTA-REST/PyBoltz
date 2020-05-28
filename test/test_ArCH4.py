import sys
import warnings
from pathlib import Path
import time
from ArXePlusAnything import *
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# print(sys.path)
from PyBoltz.Boltz import Boltz
import numpy as np
from tables import *


def test_ArCH4_90_10():

    ap = APA.getAPA()

    DriftVelocity = ap.root.CH4["10.0"].Drift_Velocity[5]
    DriftVelocityErr = ap.root.CH4["10.0"].Drift_Velocity_Error[5] * DriftVelocity

    Long = ap.root.CH4["10.0"].Long[5]
    LongE = ap.root.CH4["10.0"].LongE[5] * Long

    Trans = ap.root.CH4["10.0"].Trans[5]
    TransE = ap.root.CH4["10.0"].TransE[5] * Trans
    Object = Boltz()
    # Set the number of gases
    Object.NumberOfGases = 2
    # Set the number of collisons
    Object.MaxNumberOfCollisions = 1 * 40000000.0
    # Set penning
    Object.Enable_Penning = 0
    # Calculate the electron energy
    Object.Enable_Thermal_Motion = 1
    Object.Max_Electron_Energy = 0.0
    # Set the gas's with there given number
    Object.GasIDs = [2, 8, 0, 0, 0, 0]
    # Set the gas fractions
    Object.GasFractions = [90, 10, 0, 0, 0, 0]
    # Set the tempature
    Object.TemperatureCentigrade = float(23)
    # Set the pressure
    Object.Pressure_Torr = 750.062
    # Set the eletric field
    Object.EField = ap.root.CH4["10.0"].Reduced_Field[5]
    # Set the magnetic field and angle
    Object.BField_Mag = 0
    Object.BField_Angle = 0
    Object.Console_Output_Flag = 0
    Object.Steady_State_Threshold = 40
    Object.Which_Angular_Model = 2

    Object.Start()

    assert Object.VelocityZ >= DriftVelocity - DriftVelocityErr
    assert Object.VelocityZ <= DriftVelocity + DriftVelocityErr

    assert Object.LongitudinalDiffusion1 >= Long - LongE
    assert Object.LongitudinalDiffusion1 <= Long + LongE

    assert Object.TransverseDiffusion1 >= Trans - TransE
    assert Object.TransverseDiffusion1 <= Trans + TransE
