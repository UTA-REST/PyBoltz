cimport numpy as np
import numpy as np

np.import_array()

cdef class Gas:
    cdef public:
        int NGS, NIN, NSTEP, NANISO, NATT, NNULL, NION, IPEN, NGAS
        double TORR, TEMPC, ARY, ESTEP, AKT, EFINAL, DENS,
        np.ndarray QIN, QT1, QT2, QT3, QT4, DEN, EROOT, EG, SCLN, QNULL, QATT, NC0, EC0, WK, EFL, NG1, EG1, NG2, EG2
        np.ndarray PENFRA, KEL, PEQIN, PEQEL, EB, EION, PEQION, QION, KIN, EI, E, Q
    cdef inline int getNGS(self):
        return self.NGS
