from Magboltz cimport Magboltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow

import cython
import MONTEFTT, MONTEFDT
from FRIEDLANDT cimport FRIEDLANDT
from PT cimport PT
from TOF import TOF
from SST import SST


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ALPCALCT(Magboltz object):
    cdef int IMAX,JPRT,I
    cdef double ANETP,ANET,TCUTH,TCUTL,ZCUTH,ZCUTL,ALPHAD,ALP1,ATT1,ALP1ERR,ATT1ER,ANET2,ZSTEPM,WRZN,FC1,FC2,ALPTEST
    IMAX = <int>(object.NMAX / 10000000)
    if IMAX < 5:
        IMAX = 5
    object.NMAX = IMAX * 10000000
    object.CORR = 760 * (273.15 + object.TEMPC) / (293.15 * object.TORR)
    object.ALPP = object.ALPHA / object.CORR
    object.ATTP = object.ATT / object.CORR
    ANETP = object.ALPP - object.ATTP
    ANET = object.ALPHA - object.ATT
    TCUTH = 1.2e-10 * object.CORR
    TCUTL = 1e-13 * object.CORR
    ZCUTH = 1.2e-2 * object.CORR
    ZCUTL = 1e-5 * object.CORR

    if ANETP > 30:
        ALPHAD = 0.0
        ALP1 = object.ALPHA
        ATT1 = object.ATT
    else:
        if abs(ANETP) < 100:
            ALPHAD = abs(object.ATT) * 0.8
        elif abs(ANETP) >= 100 and abs(ANETP) < 1000:
            ALPHAD = abs(ANET) * 0.65
        elif abs(ANETP) >= 1000 and abs(ANETP) < 10000:
            ALPHAD = abs(ANET) * 0.6
        elif abs(ANETP) >= 10000 and abs(ANETP) < 100000:
            ALPHAD = abs(ANET) * 0.5
        elif abs(ANETP) >= 100000 and abs(ANETP) < 2000000:
            ALPHAD = abs(ANET) * 0.2
        else:
            raise ValueError("ATTACHMENT TOO LARGE PROGRAM STOPPED")
        object.VDST = object.WZ * 1e-5
        object.FAKEI = ALPHAD * object.WZ * 1e-12
        object.ALPHAST = 0.85 * abs(ALPHAD + ANET)
        object.TSTEP = log(3) / (object.ALPHAST * object.VDST * 1e5)
        if object.TSTEP > TCUTH:
            object.TSTEP = TCUTH
        if object.TSTEP < TCUTL:
            object.TSTEP = TCUTL
        for J in range(object.NumberOfGases):
            object.TCFMAX[J] += abs(object.FAKEI) / object.NumberOfGases

        # CONVERT TO PICOSECONDS
        object.TSTEP *= 1e12
        object.TFINAL = 7 * object.TSTEP
        object.ITFINAL = 7
        JPRT = 0
        MONTEFTT.monte(object,JPRT)
        PT(object,JPRT)
        TOF(object,JPRT)
        ALP1 = object.RALPHA / object.TOFWR * 1e7
        ALP1ER = object.RALPER * ALP1 / 100
        ATT1 = object.RATTOF / object.TOFWR * 1e7
        ATT1ER = object.RATOFER * ATT1 / 100
        for J in range(object.NumberOfGases):
            object.TCFMAX[J] -= abs(object.FAKEI) / object.NumberOfGases
        ALPHAD = ATT1 * 1.2
        if ALP1 - ATT1 > 30 * object.CORR:
            ALPHAD = abs(ATT1) * 0.4
        if abs(ALP1 - ATT1) < (ALP1 / 10) or abs(ALP1 - ATT1) < (ATT1 / 10):
            ALPHAD = abs(ATT1) * 0.3
        if ALP1 - ATT1 > 100 * object.CORR:
            ALPHAD = 0.0
        object.WZ = object.TOFWR * 1e5

    object.VDST = object.WZ * 1e-5
    object.FAKEI = ALPHAD * object.WZ * 1e-12
    object.ALPHAST = 0.85 * abs(ALPHAD + ALP1 - ATT1)
    if ALP1 + ALPHAD > 10 * object.ALPHAST or ATT1 > 10 * object.ALPHAST:
        if ALP1 + ALPHAD > 100 * object.CORR:
            object.ALPHAST *= 15
        elif ALP1 + ALPHAD > 50 * object.CORR:
            object.ALPHAST *= 12
        else:
            object.ALPHAST *= 8

    object.TSTEP = log(3) / (object.ALPHAST * object.VDST * 1e5)
    object.ZSTEP = log(3) / object.ALPHAST

    if object.TSTEP > TCUTH and ALPHAD != 0.0:
        object.TSTEP = TCUTH
    if object.ZSTEP > ZCUTH and ALPHAD != 0.0:
        object.ZSTEP = ZCUTH
    for J in range(object.NumberOfGases):
        object.TCFMAX[J] += abs(object.FAKEI) / object.NumberOfGases
    ANET2 = ALP1 - ATT1
    object.TSTEP *= 1e12
    object.ZSTEP *= 0.01
    #TODO: check index -1
    object.TFINAL = 7 * object.TSTEP
    object.ITFINAL = 7
    object.ZFINAL = 8 * object.ZSTEP
    object.IZFINAL = 8

    for I in range(1, 9):
        object.ZPLANE[I - 1] = I * object.ZSTEP
    ZSTEPM = object.ZSTEP * 1e6

    JPRT = 1
    MONTEFDT.monte(object)
    SST(object)
    object.ALPHA = object.ALPHSST
    object.ALPER = object.ALPHERR
    object.ATT = object.ATTSST
    object.ATTER = object.ATTERR

    MONTEFTT.monte(object,JPRT)
    FRIEDLANDT(object)
    PT(object,JPRT)
    TOF(object,JPRT)


    WRZN = object.TOFWR*1e5
    FC1 = WRZN/(2*object.TOFDL)
    FC2 = ((object.RALPHA-object.RATTOF)*1e12)/object.TOFDL
    ALPTEST = FC1-sqrt(FC1**2-FC2)
