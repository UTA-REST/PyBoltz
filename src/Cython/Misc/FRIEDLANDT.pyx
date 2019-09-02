from PyBoltz cimport PyBoltz
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef FRIEDLANDT(PyBoltz object):
    cdef double FR[4000],ALFBAR,ATTBAR,EBAR,FSUM,TCFSUM
    cdef int I
    ALFBAR = 0.0
    ATTBAR = 0.0
    EBAR = 0.0
    FSUM = 0.0
    for I in range(4000):
        TCFSUM = 0.0
        for GasIndex in range(object.NumberOfGases):
            TCFSUM += object.TCF[GasIndex][I]
        FR[I] = object.SPEC[I] / TCFSUM
        EBAR += object.E[I] * object.SPEC[I] / TCFSUM
        ALFBAR += object.FCION[I] * object.SPEC[I] / TCFSUM
        ATTBAR += object.FCATT[I] * object.SPEC[I] / TCFSUM
        FSUM+=FR[I]
    for I in range(4000):
        FR[I]=FR[I]/FSUM
    EBAR = EBAR/object.TTOTS
    ALFBAR/=object.TTOTS
    ATTBAR/=object.TTOTS
    # ESTIMATION USING FRIEDLAND
    return  object
