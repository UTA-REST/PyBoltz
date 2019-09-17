from Gasmix cimport Gasmix
cdef class Gasmix_mert(Gasmix):
    cdef public:
        double A, D, F, Lambda,A1,EV0