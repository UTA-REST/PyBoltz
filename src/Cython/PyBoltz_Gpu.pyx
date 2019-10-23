from PyBoltz cimport PyBoltz
import numpy as np
cimport  numpy as np
cdef extern from "MonteGpu.hh":
    cdef cppclass C_MonteGpu "MonteGpu":
        void MonteTGpu()
        C_MonteGpu()
        double PElectronEnergyStep
        double PMaxCollisionFreqTotal
        double PEField
        double PCONST1
        double PCONST2
        double PCONST3
        double Ppi
        double *PISIZE
        double PNumMomCrossSectionPoints
        double PMaxCollisionFreq
        double * PVTMB
        double PAngleFromZ
        double PAngleFromX
        double PInitialElectronEnergy
        double** PCollisionFrequency
        double *PTotalCollisionFrequency
        void ** PRGAS
        double ** PEnergyLevels
        double ** PAngleCut
        double ** PScatteringParameter
        double * PINDEX
        double * PIPN
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
        self.MonteGpuObject.PElectronEnergyStep = self.ElectronEnergyStep
        self.MonteGpuObject.PMaxCollisionFreqTotal = self.MaxCollisionFreqTotal
        self.MonteGpuObject.PEField = self.EField
        self.MonteGpuObject.PCONST1 = self.CONST1
        self.MonteGpuObject.PCONST2 = self.CONST2
        self.MonteGpuObject.PCONST3 = self.CONST3
        self.MonteGpuObject.Ppi = np.pi
        self.MonteGpuObject.PISIZE = self.ISIZE #here
        self.MonteGpuObject.PNumMomCrossSectionPoints = self.NumMomCrossSectionPoints #here
        self.MonteGpuObject.PMaxCollisionFreq = self.MaxCollisionFreq
        self.MonteGpuObject.PVTMB = self.VTMB
        self.MonteGpuObject.PAngleFromZ = self.AngleFromZ
        self.MonteGpuObject.PAngleFromX = self.AngleFromX
        self.MonteGpuObject.PInitialElectronEnergy = self.InitialElectronEnergy
        self.MonteGpuObject.PCollisionFrequency = self.CollisionFrequency
        self.MonteGpuObject.PTotalCollisionFrequency = self.TotalCollisionFrequency # here
        self.MonteGpuObject.PRGAS = self.RGAS #here
        self.MonteGpuObject.PEnergyLevels = self.EnergyLevels #here
        self.MonteGpuObject.PAngleCut = self.AngleCut # here
        self.MonteGpuObject.PScatteringParameter = self.ScatteringParameter # here
        self.MonteGpuObject.PINDEX = self.INDEX # here
        self.MonteGpuObject.PIPN = self.IPN #here

        self.MonteGpuObject.MonteTGpu()