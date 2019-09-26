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
from PyBoltz cimport PyBoltz
cdef class PyBoltz_Gpu(PyBoltz):
    def __init__(self):
        PyBoltz.__init__(self)
