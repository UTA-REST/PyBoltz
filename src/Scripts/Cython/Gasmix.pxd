from Gas cimport Gas
cimport numpy as np
import numpy as np

cdef class Gasmix:
    cdef public:
         Gas Gases[6]
