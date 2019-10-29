from PyBoltz cimport PyBoltz
import numpy as np
cimport  numpy as np
cdef extern from "MonteGpu.hh":
    cdef cppclass C_MonteGpu "MonteGpu":
        void MonteTGpu()
        void Setup()
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
        long long * SeedsGpu
        long long numElectrons
        long long NumColls
        int threads
        int blocks

cdef class PyBoltz_Gpu(PyBoltz):

    cdef public long long numElectrons
    cdef public long long NumColls
    cdef C_MonteGpu* MonteGpuObject


    def __cinit__(self):
        self.MonteGpuObject= new C_MonteGpu()

    def RunAndSetup(self):
        # Find collision frequencies and/or final electron energy
        self.Start_No_MONTE()

        # Current Output array.
        cdef double output[400000]
        cdef long long seeds[1000]

        # Setup variables
        self.MonteGpuObject.ElectronEnergyStep = self.ElectronEnergyStep
        self.MonteGpuObject.MaxCollisionFreqTotal = self.MaxCollisionFreqTotal
        self.MonteGpuObject.EField = self.EField
        self.MonteGpuObject.CONST1 = self.CONST1
        self.MonteGpuObject.CONST2 = self.CONST2
        self.MonteGpuObject.CONST3 = self.CONST3
        self.MonteGpuObject.pi = np.pi
        self.MonteGpuObject.ISIZE = self.ISIZE
        self.MonteGpuObject.NumMomCrossSectionPoints = self.NumMomCrossSectionPoints
        self.MonteGpuObject.MaxCollisionFreq = self.MaxCollisionFreq
        self.MonteGpuObject.VTMB = self.VTMB
        self.MonteGpuObject.AngleFromZ = self.AngleFromZ
        self.MonteGpuObject.AngleFromX = self.AngleFromX
        self.MonteGpuObject.InitialElectronEnergy = self.InitialElectronEnergy
        self.MonteGpuObject.CollisionFrequency = <double *>self.CollisionFrequency
        self.MonteGpuObject.TotalCollisionFrequency = <double *>self.TotalCollisionFrequency
        self.MonteGpuObject.RGAS = <double *>self.RGAS
        self.MonteGpuObject.EnergyLevels = <double *>self.EnergyLevels
        self.MonteGpuObject.AngleCut = <double *>self.AngleCut
        self.MonteGpuObject.ScatteringParameter = <double *>self.ScatteringParameter
        self.MonteGpuObject.INDEX = <double *>self.INDEX
        self.MonteGpuObject.IPN = <double *>self.IPN
        self.MonteGpuObject.output = <double *>output
        self.MonteGpuObject.SeedsGpu = <long long*>seeds
        self.MonteGpuObject.numElectrons = self.numElectrons
        self.MonteGpuObject.NumColls = self.NumColls
        self.MonteGpuObject.threads = 25
        self.MonteGpuObject.blocks = 40

        print ("Setting up...")
        self.MonteGpuObject.Setup()
        print ("Running...")
        # Run the Gpu code.
        self.MonteGpuObject.MonteTGpu()

        print("here",self.MonteGpuObject.RGAS[10])