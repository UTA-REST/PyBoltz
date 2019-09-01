from CollisionFreqs cimport CollisionFreq, CollisionFreqT


from Magboltz cimport Magboltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow,tan,atan


cimport cython
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef SST(Magboltz object):
    cdef double ESST[8],VDSST[8],WSSST[8],DXSST[8],DYSST[8],WTEMP[8],DRSST[8],ALFNE[8],ALFNJ[8],ALFN[8],ZSST[8],DLSST[8]
    cdef double DRSS1[8],DRSS2[8],DRSS3[8],ALFEX1[8],NEPL[8],DUM[6],FRION,FRATT,ATTOINT,CORERR,CORF
    cdef int JPRINT,K,I,JPRT
    object.VDOUT = 0.0
    object.VDERR = 0.0
    object.WSOUT = 0.0
    object.WSERR = 0.0
    object.DLOUT = 0.0
    object.DLERR = 0.0
    object.DTOUT = 0.0
    object.DTERR = 0.0
    object.ALPHSST = 0.0
    object.ALPHERR = 0.0
    object.ATTSST = 0.0
    object.ATTERR = 0.0
    JPRINT = <int>(object.IZFINAL)
    NEPL[0] = object.IPRIM + object.NESST[0]
    for K in range(1, JPRINT):
        NEPL[K] = NEPL[K - 1] + object.NESST[K]
    for K in range(JPRINT):
        object.NESST[K] = NEPL[K]

    for I in range(JPRINT):
        if object.NESST[I] == 0:
            JPRINT = I - 1
            break
    ESST[0] = object.ESPL[0] / object.TSSUM[0]
    ZSST[0] = object.ZSPL[0] / object.TSSUM[0]
    VDSST[0] = object.VZSPL[0] / object.TSSUM[0]
    WTEMP[0] = object.ZSTEP * object.TSSUM[0] / object.TMSPL[0]
    WSSST[0] = WTEMP[0]
    DXSST[0] = ((object.XXSPL[0] / object.TSSUM[0]) - (object.XSPL[0] / object.TSSUM[0]) ** 2) * WSSST[0] / (
            2 * object.ZSTEP)
    DYSST[0] = ((object.YYSPL[0] / object.TSSUM[0]) - (object.YSPL[0] / object.TSSUM[0]) ** 2) * WSSST[0] / (
            2 * object.ZSTEP)
    DLSST[0] = ((object.TTMSPL[0] / object.TSSUM[0]) - (object.TMSPL[0] / object.TSSUM[0]) ** 2) * WSSST[
        0] ** 3 / (
                       2 * object.ZSTEP)
    if object.EnableThermalMotion == 1:
        DUM = CollisionFreqT(object)
    if object.EnableThermalMotion == 0:
        DUM = CollisionFreq(object)
    FRION = DUM[2]
    FRATT = DUM[3]
    ATTOINT = FRATT / FRION
    CORERR = abs((object.FAKEI + FRION - FRATT) / (FRION - FRATT))
    CORF = (FRION - FRATT) / (object.FAKEI + FRION - FRATT)
    if object.NESST[0] != 0:
        ALFNE[0] = CORF * (log(float(object.NESST[0])) - log(float(object.IPRIM))) / object.ZSTEP
    ALFNJ[0] = 0.0
    ALFN[0] = 0.0
    if JPRINT == 8:
        JPRT = JPRINT
    else:
        JPRT = JPRINT + 1
    for I in range(1, JPRT):
        ESST[I] = object.ESPL[I] / object.TSSUM[I]
        ZSST[I] = object.ZSPL[I] / object.TSSUM[I]
        VDSST[I] = object.VZSPL[I] / object.TSSUM[I]
        WTEMP[I] = object.ZSTEP * float(I + 1) * object.TSSUM[I] / object.TMSPL[I]
        WSSST[I] = (WTEMP[I] * WTEMP[I - 1]) / ((I + 1) * WTEMP[I - 1] - I * WTEMP[I])
        DXSST[I] = ((object.XXSPL[I] / object.TSSUM[I]) - (object.XSPL[I] / object.TSSUM[I]) ** 2 - (
                object.XXSPL[I - 1] / object.TSSUM[I - 1]) + (
                            object.XSPL[I - 1] / object.TSSUM[I - 1]) ** 2) * WSSST[I] / (2 * object.ZSTEP)
        DYSST[I] = ((object.YYSPL[I] / object.TSSUM[I]) - (object.YSPL[I] / object.TSSUM[I]) ** 2 - (
                object.YYSPL[I - 1] / object.TSSUM[I - 1]) + (
                            object.YSPL[I - 1] / object.TSSUM[I - 1]) ** 2) * WSSST[I] / (2 * object.ZSTEP)

        DLSST[I] = ((object.TTMSPL[I] / object.TSSUM[I]) - (object.TMSPL[I] / object.TSSUM[I]) ** 2 - (
                object.TTMSPL[I - 1] / object.TSSUM[I - 1]) + (
                            object.TMSPL[I - 1] / object.TSSUM[I - 1]) ** 2) * WSSST[I] ** 3 / (2 * object.ZSTEP)

        ALFN[I] = (log(object.TSSUM[I]) - log(object.TSSUM[I - 1])) / object.ZSTEP
        ALFNJ[I] = (log(object.TSSUM[I] * VDSST[I]) - log(
            object.TSSUM[I - 1] * VDSST[I - 1])) / object.ZSTEP
        ALFNE[I] = 0.0

        # weird if statement

        ALFNE[I] = (log(float(object.NESST[I])) - log(float(object.NESST[I - 1]))) / object.ZSTEP
        ALFN[I] = CORF * ALFN[I]
        ALFNE[I] *= CORF
        ALFNJ[I] *= CORF
    cdef double DXFIN,DYFIN,DLFIN,ALNGTH,ALFIN,ANST2,ANST3,ANST4,ANST5,ANST6,ANST7,ANST8,ATMP,ATMP2,ATER
    if JPRINT == 8:
        JPRINT -= 1

    DXFIN = ((object.XXSPL[JPRINT] / object.TSSUM[JPRINT]) - (
            object.XSPL[JPRINT] / object.TSSUM[JPRINT]) ** 2) * WSSST[JPRINT] / (
                    (JPRINT + 1) * 2 * object.ZSTEP)
    DXFIN *= 1e16

    DYFIN = ((object.YYSPL[JPRINT] / object.TSSUM[JPRINT]) - (
            object.YSPL[JPRINT] / object.TSSUM[JPRINT]) ** 2) * WSSST[JPRINT] / (
                    (JPRINT + 1) * 2 * object.ZSTEP)
    DYFIN *= 1e16

    DLFIN = ((object.TTMSPL[JPRINT] / object.TSSUM[JPRINT]) - (
            object.TMSPL[JPRINT] / object.TSSUM[JPRINT]) ** 2) * WSSST[JPRINT] ** 3 / (
                    (JPRINT + 1) * 2 * object.ZSTEP)
    DLFIN *= 1e16

    ALNGTH = object.ZSTEP * float(JPRINT + 1)
    ALFIN = log(float(object.NESST[JPRINT]) / float(object.IPRIM)) / ALNGTH
    ALFIN *= 0.01
    for J in range(JPRT):
        VDSST[J] *= 1e9
        WSSST[J] *= 1e9
        DXSST[J] *= 1e16
        DYSST[J] *= 1e16
        DLSST[J] *= 1e16
        ALFN[J] *= 0.01
        ALFNJ[J] *= 0.01
        ALFNE[J] *= 0.01
    for IPL in range(JPRT):
        DRSST[IPL] = (DXSST[IPL] + DYSST[IPL]) / 2

    if object.NESST[0] > object.NESST[4]:
        object.VDOUT = VDSST[1]
        object.VDERR = 100 * abs((WSSST[1] - WSSST[2]) / (2 * WSSST[1]))
        object.WSOUT = WSSST[1]
        object.DLOUT = DLSST[1]
        object.DLERR = 100 * abs((DLSST[1] - DLSST[2]) / (2 * DLSST[1]))
        object.DTOUT = DRSST[1]
        object.DTERR = 100 * abs((DRSST[1] - DRSST[2]) / (2 * DRSST[1]))
        if ATTOINT == -1:
            object.ALPHSST = 0.0
            object.ALPHERR = 0.0
            ANST2 = float(object.NESST[1])
            ANST3 = float(object.NESST[2])
            ANST4 = ANST3 - sqrt(ANST3)
            ANST5 = log(ANST2 / ANST3)
            ANST6 = log(ANST2 / ANST4)
            ANST7 = ANST6 / ANST5
            ANST8 = ANST7 - 1.0
            object.ATTSST = -1 * (ALFN[1] + ALFNJ[1] + ALFNE[1]) / 3
            object.ATTERR = 100 * sqrt(ANST8 ** 2 + object.ATTATER ** 2)
        else:
            ANST2 = float(object.NESST[1])
            ANST3 = float(object.NESST[2])
            ANST4 = ANST3 - sqrt(ANST3)
            ANST5 = log(ANST2 / ANST3)
            ANST6 = log(ANST2 / ANST4)
            ANST7 = ANST6 / ANST5
            ANST8 = ANST7 - 1.0
            ATMP = (ALFN[1] + ALFNJ[1] + ALFNE[1]) / 3
            object.ALPHSST = ATMP / (1 - ATTOINT)
            object.ALPHERR = 100 * sqrt(ANST8 ** 2 + object.ATTIOER ** 2)
            object.ATTSST = ATTOINT * ATMP / (1 - ATTOINT)
            object.ATTERR = 100 * sqrt(ANST8 ** 2 + object.ATTATER ** 2)
    else:
        object.VDOUT = VDSST[7]
        object.VDERR = 100 * abs((VDSST[7] - VDSST[6]) / VDSST[7])
        object.WSOUT = WSSST[7]
        object.WSERR = 100 * abs((WSSST[7] - WSSST[6]) / WSSST[7])
        object.DLOUT = DLFIN
        object.DLERR = 100 * abs((object.DLOUT - DLSST[7]) / object.DTOUT)
        object.DTOUT = (DXFIN + DYFIN) / 2
        object.DTERR = 100 * abs((object.DTOUT - DRSST[7]) / object.DTOUT)
        ATMP = (ALFN[7] + ALFNJ[7] + ALFNE[7]) / 3
        ATMP2 = (ALFN[6] + ALFNJ[6] + ALFNE[6]) / 3
        ATER = abs((ATMP - ATMP2) / ATMP)
        object.ALPHSST = ATMP / (1 - ATTOINT)
        object.ALPHERR = 100 * sqrt(ATER ** 2 + object.ATTIOER ** 2)
        object.ATTSST = ATTOINT * ATMP / (1 - ATTOINT)
        if (object.ATTOION != 0):
            object.ATTERR = 100 * sqrt(ATER ** 2 + object.ATTATER ** 2)
        else:
            object.ATTERR = 0.0

