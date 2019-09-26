import sys
import warnings
import time
import os
print(os.environ['PYTHONPATH'])

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
print(sys.path)
from PyBoltz import PyBoltz
import numpy as np
from ctypes import *
import os

import ctypes

'''
double PElectronEnergyStep,double PMaxCollisionFreqTotal,double PEField, double PCONST1,double PCONST2,double PCONST3
, double Ppi,double PISIZE,double PNumMomCrossSectionPoints,double PMaxCollisionFreq, double * PVTMB, double PAngleFromZ, double PAngleFromX,
double PInitialElectronEnergy, double** PCollisionFrequency, double *PTotalCollisionFrequency, double ** PRGAS, double ** PEnergyLevels,
double ** PAngleCut,double ** PScatteringParameter, double * PINDEX, double * PIPN'''


def get_MonteTGpu():
    func = CDLL(os.path.abspath("./MonteTGpu.so"),mode=ctypes.RTLD_LOCAL).MonteTGpu
    func.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double,
                     POINTER(c_double), c_double, c_double, c_double, POINTER(POINTER(c_double)),
                     POINTER(c_double), POINTER(POINTER(c_double)), POINTER(POINTER(c_double)),
                     POINTER(POINTER(c_double)), POINTER(POINTER(c_double)), POINTER(c_double), POINTER(c_double)]

    return func


_MonteTGPU = get_MonteTGpu()


def MonteTGpu(PElectronEnergyStep, PMaxCollisionFreqTotal, PEField, PCONST1, PCONST2, PCONST3
              , Ppi, PISIZE, PNumMomCrossSectionPoints, PMaxCollisionFreq, PVTMB, PAngleFromZ, PAngleFromX,
              PInitialElectronEnergy, PCollisionFrequency, PTotalCollisionFrequency, PRGAS, PEnergyLevels,
              PAngleCut, PScatteringParameter, PINDEX, PIPN):
    PElectronEnergyStep_P = ctypes.c_double(PElectronEnergyStep)

    PMaxCollisionFreqTotal_P =  ctypes.c_double(PMaxCollisionFreqTotal)
    PEField_P = ctypes.c_double(PEField)
    PCONST1_P = ctypes.c_double(PCONST1)
    PCONST2_P = ctypes.c_double(PCONST2)
    PCONST3_P = ctypes.c_double(PCONST3)
    Ppi_P = ctypes.c_double(Ppi)
    PISIZE_P = ctypes.c_double(PISIZE)
    PNumMomCrossSectionPoints_P = ctypes.c_double(PNumMomCrossSectionPoints)
    PMaxCollisionFreq_P = ctypes.c_double(PMaxCollisionFreq)
    PVTMB[0]=9
    PVTMB_P = (ctypes.c_double * len(PVTMB))(*PVTMB)
    PVTMB_P =ctypes.cast(PVTMB_P, ctypes.POINTER(c_double))

    print(PVTMB_P)
    print(PVTMB_P[0])
    PAngleFromZ_P = ctypes.c_double(PAngleFromZ)
    PAngleFromX_P = ctypes.c_double(PAngleFromX)
    PInitialElectronEnergy_P = ctypes.c_double(PInitialElectronEnergy)
    PCollisionFrequency_P = (POINTER(ctypes.c_double)*len(PCollisionFrequency))()
    for i in range(len(PCollisionFrequency)):
        PCollisionFrequency_P[i] = (ctypes.c_double * len(PCollisionFrequency[0]))()
        for j in range(len(PCollisionFrequency[0])):
            PCollisionFrequency_P[i][j] = PCollisionFrequency[i][j]
    PCollisionFrequency_P =ctypes.cast(PCollisionFrequency_P, POINTER(POINTER(c_double)))


    PTotalCollisionFrequency_P = (ctypes.c_double * len(PTotalCollisionFrequency))()
    for i in range(len(PTotalCollisionFrequency)):
        PTotalCollisionFrequency_P[i] = PTotalCollisionFrequency[i]
    PTotalCollisionFrequency_P = ctypes.cast(PTotalCollisionFrequency_P, ctypes.POINTER(c_double))

    PRGAS[5][289] = 656
    PRGAS_P = (POINTER(ctypes.c_double)*len(PRGAS))()
    for i in range(len(PRGAS)):
        PRGAS_P[i] = (ctypes.c_double * len(PRGAS[0]))()
        for j in range(len(PRGAS[0])):
            PRGAS_P[i][j] = PRGAS[i][j]
    PRGAS_P = ctypes.cast(PRGAS_P, POINTER(POINTER(c_double)))


    PEnergyLevels_P = (POINTER(ctypes.c_double)*len(PEnergyLevels))()
    for i in range(len(PEnergyLevels)):
        PEnergyLevels_P[i] = (ctypes.c_double * len(PEnergyLevels[0]))()
        for j in range(len(PEnergyLevels)):
            PEnergyLevels_P[i][j] = PEnergyLevels[i][j]
    PEnergyLevels_P = ctypes.cast(PEnergyLevels_P, POINTER(POINTER(c_double)))

    PAngleCut_P = (POINTER(ctypes.c_double)*len(PAngleCut))()
    for i in range(len(PAngleCut)):
        PAngleCut_P[i] = (ctypes.c_double * len(PAngleCut[0]))()
        for j in range(len(PAngleCut[0])):
            PAngleCut_P[i][j] = PAngleCut[i][j]
    PAngleCut_P=ctypes.cast(PAngleCut_P, POINTER(POINTER(c_double)))

    PScatteringParameter_P = (POINTER(ctypes.c_double)*len(PScatteringParameter))()
    for i in range(len(PScatteringParameter)):
        PScatteringParameter_P[i] = (ctypes.c_double * len(PScatteringParameter[0]))()
        for j in range(len(PScatteringParameter[0])):
            PScatteringParameter_P[i][j] = PScatteringParameter[i][j]
    PScatteringParameter_P =ctypes.cast(PScatteringParameter_P, POINTER(POINTER(c_double)))

    PINDEX_P = (ctypes.c_double * len(PINDEX))()
    for i in range(len(PINDEX)):
        PINDEX_P[i] = PINDEX[i]
    ctypes.cast(PINDEX_P, ctypes.POINTER(c_double))

    PIPN_P = (ctypes.c_double * len(PIPN))()
    for i in range(len(PIPN)):
        PIPN_P[i] = PIPN[i]
    ctypes.cast(PIPN_P, ctypes.POINTER(c_double))

    _MonteTGPU(PElectronEnergyStep_P, PMaxCollisionFreqTotal_P, PEField_P, PCONST1_P, PCONST2_P, PCONST3_P
              , Ppi_P, PISIZE_P, PNumMomCrossSectionPoints_P, PMaxCollisionFreq_P, PVTMB_P, PAngleFromZ_P, PAngleFromX_P,
              PInitialElectronEnergy_P, PCollisionFrequency_P, PTotalCollisionFrequency_P, PRGAS_P, PEnergyLevels_P,
              PAngleCut_P, PScatteringParameter_P, PINDEX_P, PIPN_P)

CF4 = 1
He4 = 3
He3 = 4
Ne = 5
Kr = 6
Xe = 7
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

Object.__init__()

import time

t1 = time.time()

# Set the number of gases
Object.NumberOfGases = 1
# Set the number of collisons
Object.MaxNumberOfCollisions = 40000000.0
# Set penning
Object.EnablePenning = 0
# Calculate the electron energy
Object.EnableThermalMotion = 1
Object.FinalElectronEnergy = 20.0
# Set the gas's with there given number
Object.GasIDs = [7, 0, 0, 0, 0, 0]

# Set the gas fractions
Object.GasFractions = [100, 0, 0, 0, 0, 0]
# Set the tempature
Object.TemperatureCentigrade = float(23)
# Set the pressure
Object.PressureTorr = 750.062
# Set the eletric field
Object.EField = 1000
# Set the magnetic field and angle
Object.BFieldMag = 0
Object.BFieldAngle = 0
Object.ConsoleOutputFlag = 1
Object.RandomSeed = 54217137
Object.WhichAngularModel = 2
Object.Start_No_MONTE()

print(Object.RGAS[0][0])
MonteTGpu(Object.ElectronEnergyStep, Object.MaxCollisionFreqTotal, Object.EField, Object.CONST1, Object.CONST2, Object.CONST3
              , np.pi, Object.ISIZE[0], Object.NumMomCrossSectionPoints[0], Object.MaxCollisionFreq[0], Object.VTMB, Object.AngleFromZ, Object.AngleFromX,
              Object.InitialElectronEnergy, Object.CollisionFrequency[0], Object.TotalCollisionFrequency[0], Object.RGAS, Object.EnergyLevels,
              Object.AngleCut[0], Object.ScatteringParameter[0], Object.INDEX[0], Object.IPN[0])
print(Object.VelocityZ)
t2 = time.time()

print("************************************************")
print("************************************************")
print("*****         Here are the outputs         *****")
print("************************************************ \n")
print("run time [s]= ", round(t2 - t1, 3))

for I in range(Object.NumberOfGases):
    if Object.GasIDs[I] <= 25:
        print("Percentage of " + GASES[int(Object.GasIDs[I])] + " = " + str(Object.GasFractions[I]))

print("Tempature [C]         = ", Object.TemperatureCentigrade)
print("Pressure [torr]       = ", Object.PressureTorr)
print("Eletric field [V/cm]  = ", Object.EField)
print("----------------------------------------------------")
print("Drift velocity [mm/mus]              = ", round(Object.VelocityZ, 3))
print("----------------------------------------------------")
print("Drift velocity error [%]             = ", round(Object.VelocityErrorZ, 3))
print("----------------------------------------------------")
print("Transverse diffusion [cm**2/s]       = ", round(Object.TransverseDiffusion, 3))
print("DIFXX [cm**2/s]       = ", round(Object.DiffusionX, 3))
print("DIFYY [cm**2/s]       = ", round(Object.DiffusionY, 3))
print("DIFZZ [cm**2/s]       = ", round(Object.DiffusionZ, 3))
print("DIFYZ [cm**2/s]       = ", round(Object.DiffusionYZ, 3))
print("DIFXY [cm**2/s]       = ", round(Object.DiffusionXY, 3))
print("DIFXZ [cm**2/s]       = ", round(Object.DiffusionXZ, 3))

print("----------------------------------------------------")
print("Transverse diffusion error [%]       = ", round(Object.TransverseDiffusionError, 3))
print("----------------------------------------------------")
print("Longitudinal diffusion [cm**2/s]     = ", round(Object.LongitudinalDiffusion, 3))
print("----------------------------------------------------")
print("Longitudinal diffusion error [%]     = ", round(Object.LongitudinalDiffusionError, 3))
print("----------------------------------------------------")
print("Transverse diffusion [mum/cm**0.5]   = ", round(Object.TransverseDiffusion1, 3))
print("----------------------------------------------------")
print("Transverse diffusion error [%]       = ", round(Object.TransverseDiffusion1Error, 3))
print("----------------------------------------------------")
print("Longitudinal diffusion [mum/cm**0.5] = ", round(Object.LongitudinalDiffusion1, 3))
print("----------------------------------------------------")
print("Longitudinal diffusion error [%]     = ", round(Object.LongitudinalDiffusion1Error, 3))
print("----------------------------------------------------")
print("Mean electron energy [eV]            = ", round(Object.MeanElectronEnergy, 3))
print("----------------------------------------------------")
print("Mean electron energy error [%]       = ", round(Object.MeanElectronEnergyError, 3))
print("----------------------------------------------------")
print("************************************************")
print("************************************************")
print(Object.MeanCollisionTime)
