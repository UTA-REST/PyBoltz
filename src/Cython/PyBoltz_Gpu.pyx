from PyBoltz cimport PyBoltz
import numpy as np
cimport  numpy as np
cdef extern from "MonteGpu.hh":
    cdef cppclass C_MonteGpu "MonteGpu":
        void MonteTGpu()
        C_MonteGpu()
        double ElectronEnergyStep
        double MaxCollisionFreqTotal
        double EField
        double CONST1
        double CONST2
        double CONST3
        double pi
        double *ISIZE
        double *NumMomCrossSectionPoints
        double *MaxCollisionFreq
        double * VTMB
        double AngleFromZ
        double AngleFromX
        double InitialElectronEnergy
        double * CollisionFrequency
        double * TotalCollisionFrequency
        double * RGAS
        double * EnergyLevels
        double * AngleCut
        double * ScatteringParameter
        double * INDEX
        double * IPN
        double * output

'''MonteTGpu(Object.ElectronEnergyStep, Object.MaxCollisionFreqTotal, Object.EField, Object.CONST1, Object.CONST2, Object.CONST3
              , np.pi, Object.ISIZE[0], Object.NumMomCrossSectionPoints[0], Object.MaxCollisionFreq[0], Object.VTMB, Object.AngleFromZ, Object.AngleFromX,
              Object.InitialElectronEnergy, Object.CollisionFrequency[0], Object.TotalCollisionFrequency[0], Object.RGAS, Object.EnergyLevels,
              Object.AngleCut[0], Object.ScatteringParameter[0], Object.INDEX[0], Object.IPN[0])'''

cdef class PyBoltz_Gpu(PyBoltz):
    cdef int numElectrons
    cdef int NumColls
    cdef C_MonteGpu* MonteGpuObject
    def __cinit__(self):
        self.MonteGpuObject= new C_MonteGpu()

    def Run(self):
        self.Start_No_MONTE()
        cdef double output[400000]
        self.MonteGpuObject.ElectronEnergyStep = self.ElectronEnergyStep
        self.MonteGpuObject.MaxCollisionFreqTotal = self.MaxCollisionFreqTotal
        self.MonteGpuObject.EField = self.EField
        self.MonteGpuObject.CONST1 = self.CONST1
        self.MonteGpuObject.CONST2 = self.CONST2
        self.MonteGpuObject.CONST3 = self.CONST3
        self.MonteGpuObject.pi = np.pi
        self.MonteGpuObject.ISIZE = self.ISIZE #here
        self.MonteGpuObject.NumMomCrossSectionPoints = self.NumMomCrossSectionPoints #here
        self.MonteGpuObject.MaxCollisionFreq = self.MaxCollisionFreq
        self.MonteGpuObject.VTMB = self.VTMB
        self.MonteGpuObject.AngleFromZ = self.AngleFromZ
        self.MonteGpuObject.AngleFromX = self.AngleFromX
        self.MonteGpuObject.InitialElectronEnergy = self.InitialElectronEnergy
        self.MonteGpuObject.CollisionFrequency = <double *>self.CollisionFrequency
        self.MonteGpuObject.TotalCollisionFrequency = <double *>self.TotalCollisionFrequency # here
        self.MonteGpuObject.RGAS = <double *>self.RGAS #here
        self.RGAS[0][10]=200
        self.MonteGpuObject.EnergyLevels = <double *>self.EnergyLevels #here
        self.MonteGpuObject.AngleCut = <double *>self.AngleCut # here
        self.MonteGpuObject.ScatteringParameter = <double *>self.ScatteringParameter # here
        self.MonteGpuObject.INDEX = <double *>self.INDEX # here
        self.MonteGpuObject.IPN = <double *>self.IPN #here
        self.MonteGpuObject.output = <double *>output

        self.MonteGpuObject.MonteTGpu()