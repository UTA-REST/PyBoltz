import Magboltz
from libc.stdlib cimport rand, RAND_MAX,srand
from libc.math cimport sin, cos, log, sqrt
cimport cython
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX

cdef void GERJAN(double RDUM, double API,double *RNMX):
    cdef double RAN1, RAN2, TWOPI
    srand(int(RDUM*1000))
    for J in range(0, 5, 2):
        RAN1 = random_uniform()
        RAN2 = random_uniform()
        TWOPI = 2.0 * API
        RNMX[J] = sqrt(-1*log(RAN1)) * cos(RAN2 * TWOPI)
        RNMX[J + 1] = sqrt(-1*log(RAN1)) * sin(RAN2 * TWOPI)
