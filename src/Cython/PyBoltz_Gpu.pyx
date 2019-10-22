from PyBoltz cimport PyBoltz

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
        double PISIZE
        double PNumMomCrossSectionPoints
        double PMaxCollisionFreq
        double * PVTMB
        double PAngleFromZ
        double PAngleFromX
        double PInitialElectronEnergy
        double** PCollisionFrequency
        double *PTotalCollisionFrequency
        double ** PRGAS
        double ** PEnergyLevels
        double ** PAngleCut
        double ** PScatteringParameter
        double * PINDEX
        double * PIPN
        double * output



cdef class PyBoltz_Gpu(PyBoltz):
    cdef int numElectrons
    cdef int NumColls
    cdef C_MonteGpu* MonteGpuObject
    def __cinit__(self):
        self.MonteGpuObject= new C_MonteGpu()

    def Run(self):

        #self.Start_No_MONTE()
        self.MonteGpuObject.PElectronEnergyStep = 1000
        self.MonteGpuObject.MonteTGpu()