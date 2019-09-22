from PyBoltz cimport PyBoltz
cdef class PyBoltz_Gpu(PyBoltz):
    cdef public:
        int numElectrons

