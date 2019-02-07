import Magboltz
from libc.math cimport sin, cos, log, sqrt
cimport cython

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform():
    cdef double r = drand48()
    return r
cdef void GERJAN(double RDUM, double API,double *RNMX):
    cdef double RAN1, RAN2, TWOPI
    srand48(int(RDUM*1000))
    for J in range(0, 5, 2):
        RAN1 = random_uniform()
        RAN2 = random_uniform()
        TWOPI = 2.0 * API
        RNMX[J] = sqrt(-1*log(RAN1)) * cos(RAN2 * TWOPI)
        RNMX[J + 1] = sqrt(-1*log(RAN1)) * sin(RAN2 * TWOPI)
