from PyBoltz cimport PyBoltz
cdef class PyBoltz_mert(PyBoltz):
    cdef public:
        double A,D,F,A1,Lambda,EV0