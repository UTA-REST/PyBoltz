from Magboltz cimport Magboltz
from libc.math cimport log


cimport cython
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef PT(Magboltz object,int JPRT):
    cdef double ANTPL[8]
    cdef int I
    ANTPL[0] = <double>(object.NETPL[0])
    object.RI[0] = (log(ANTPL[0]) - log((object.IPRIM))) / object.TSTEP
    object.RI[0] -= object.FAKEI
    object.EPT[0] = object.ETPL[0] / ANTPL[0]
    object.TTEST[0]=object.TTPL[0]/ANTPL[0]
    object.VZPT[0]=1e9*object.VZTPL[0]/ANTPL[0]
    for I in range(1,<int>(object.ITFINAL)):
        if object.NETPL[I]==0:
            object.ITFINAL=I-1
            break
        ANTPL[I]=float(object.NETPL[I])
        object.RI[I]= (log(ANTPL[I]) - log(ANTPL[I-1])) / object.TSTEP
        object.RI[I] -= object.FAKEI
        object.EPT[I]=object.ETPL[I]/ANTPL[I]
        object.TTEST[I]=object.TTPL[I]/ANTPL[I]
        object.VZPT[I]=1e9*object.VZTPL[I]/ANTPL[I]
    # print pulsed townsend results
