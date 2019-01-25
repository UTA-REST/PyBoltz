import numpy as np
cimport numpy as np

cdef class Gas:
    def __init__(self):
        self.NGS = 0
        self.Q = np.zeros(shape=(6,4000))
        self.QIN = np.zeros(shape=(250,4000))
        self.NIN =0
        self.E = np.zeros(6)
        self.EI = np.zeros(250)
        self.KIN = np.zeros(250)
        self.QION = np.zeros(shape=(30,4000))
        self.PEQION = np.zeros(shape=(30,4000))
        self.EION = np.zeros(30)
        self.EB = np.zeros(30)
        self.PEQEL = np.zeros(shape=(6,4000))
        self.PEQIN = np.zeros(shape=(250,4000))
        self.KEL = np.zeros(6)
        self.PENFRA = np.zeros(shape=(3,290))
        self.NC0 = np.zeros(30)
        self.EC0 = np.zeros(30)
        self.WK = np.zeros(30)
        self.EFL = np.zeros(30)
        self.NG1 = np.zeros(30)
        self.EG1 = np.zeros(30)
        self.NG2 = np.zeros(30)
        self.EG2 = np.zeros(30)
        self.QATT = np.zeros(shape=(8,4000))
        self.QNULL = np.zeros(shape=(10,4000))
        self.SCLN = np.zeros(10)
        self.EG =np.zeros(shape=(4000))
        self.EROOT =np.zeros(shape=(4000))
        self.QT1 =np.zeros(shape=(4000))
        self.QT2 =np.zeros(shape=(4000))
        self.QT3 =np.zeros(shape=(4000))
        self.QT4 =np.zeros(shape=(4000))
        self.DEN =np.zeros(shape=(4000))
