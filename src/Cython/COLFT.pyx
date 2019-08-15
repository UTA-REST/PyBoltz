
from Magboltz cimport Magboltz


cimport cython
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* COLFT(Magboltz object):
    cdef double NINEL, NELA,NATT,NION
    cdef int J
    NINEL=0
    NELA = 0
    NATT = 0
    NION = 0
    for J in range(object.NGAS):
        NINEL+=object.ICOLL[J][3]+object.ICOLL[J][4]
        NELA+=object.ICOLL[J][0]
        NATT+=object.ICOLL[J][2]
        NION+=object.ICOLL[J][1]
    NTOTAL=NELA+NATT+NION+NINEL
    if object.TTOTS == 0:
        NREAL = NTOTAL
        object.TTOTS = object.ST
    else:
        NREAL = NTOTAL

    cdef double DUM[6]
    DUM[5] = NTOTAL
    DUM[0] = <double>(NREAL) / object.TTOTS
    DUM[4] = <double>(NINEL) / object.TTOTS
    DUM[1] = <double>(NELA) / object.TTOTS
    DUM[2] = <double>(NION) / object.TTOTS
    DUM[3] = <double>(NATT) / object.TTOTS
    return DUM