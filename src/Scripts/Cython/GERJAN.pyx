import Magboltz
from libc.math cimport sin, cos, log, sqrt
cimport cython

cdef extern from "C/RM48.h":
    double DRAND48(double dummy)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = DRAND48(dummy)
    return r
cdef void GERJAN(double RDUM, double API,double *RNMX):
    cdef double RAN1, RAN2, TWOPI
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RDUM)
        RAN2 = random_uniform(RDUM)
        TWOPI = 2.0 * API
        RNMX[J] = sqrt(-1*log(RAN1)) * cos(RAN2 * TWOPI)
        RNMX[J + 1] = sqrt(-1*log(RAN1)) * sin(RAN2 * TWOPI)

