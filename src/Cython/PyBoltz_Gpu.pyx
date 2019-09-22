from PyBoltz cimport PyBoltz
import math

from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
from libc.string cimport memset
import Setups
import Mixers
import EnergyLimits
import Monte
from Monte import *
from Gasmix cimport Gasmix
cdef extern from "C/RM48.h":
    double DRAND48(double dummy)
    void RM48(double lenv)
    void RM48IN(int IJL, int NTOT, intNTOT2)


cdef double drand48(double dummy):
    return DRAND48(dummy)

cdef void setSeed(int seed):
    RM48IN(seed,0,0)
    return
cdef class PyBoltz_Gpu(PyBoltz):
    def __init__(self):
        PyBoltz.__init__(self)

