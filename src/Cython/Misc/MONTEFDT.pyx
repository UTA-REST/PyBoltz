from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan
from libc.string cimport memset
from MBSorts cimport MBSortT
from libc.stdlib cimport malloc, free
import cython
import numpy as np
from SPLANET cimport SPLANET
from PyBoltz cimport drand48

from TCALCT cimport TCALCT

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = drand48(dummy)
    return r

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void GERJAN(double RDUM, double API, double *RNMX):
    cdef double RAN1, RAN2, TWOPI
    cdef int J
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RDUM)
        RAN2 = random_uniform(RDUM)
        TWOPI = 2.0 * API
        RNMX[J] = sqrt(-1 * log(RAN1)) * cos(RAN2 * TWOPI)
        RNMX[J + 1] = sqrt(-1 * log(RAN1)) * sin(RAN2 * TWOPI)

cpdef GOTO544(PyBoltz object):
    global NCLUS, IZPLANE, TZSTOP, f700, DCZ1, DCX1, DCY1, E1, TSSTRT, ZSTRT, EPRM, IDUM, IESPECP, TDASH, NELEC, TZSTOP1, DCZ100, DCX100
    global DCY100, E100, ITER
    object.IPRIM += 1
    IZPLANE = 0
    TZSTOP = 1000
    if object.IPRIM > 1:
        if ITER > object.NMAX:
            object.IPRIM -= 1
            f700 = 1
        else:
            object.X = 0.0
            object.Y = 0.0
            object.Z = 0.0
            DCZ1 = DCZ100
            DCX1 = DCX100
            DCY1 = DCY100
            E1 = E100
            NCLUS += 1
            object.ST = 0.0
            TSSTRT = 0.0
            ZSTRT = 0.0
    # IF PRIM >10000000: TOO MANY PRIMARIES
    if f700 == 0:
        EPRM[<int> (object.IPRIM - 1)] = E1
        IDUM = int(E1)
        IDUM = min(IDUM, 99)
        IESPECP[IDUM] += 1
        TDASH = 0.0
        NELEC += 1
        TZSTOP1 = 0

cpdef GOTO555():
    global TDASH, NELEC, TZSTOP1
    TDASH = 0.0
    NELEC += 1
    TZSTOP1 = 0

cpdef GOTO18n20(PyBoltz object, int f20):
    global NELEC, NCLUS, E1, DCX1, DCY1, DCZ1, IZPLANE, NPONT, ZSTRT, TSSTRT, EPOT, ISOL, R, TZSTOP, TZSTOP1, ISOL
    while (1):
        if f20 != 1:
            ZTOT += object.Z
            TTOT += object.ST
            ZTOTS += object.Z - ZSTRT
            TTOTS += object.ST - TSSTRT
            if NELEC == NCLUS + 1:
                GOTO544(object)
                continue
        f20 = 0
        object.X = object.XSS[NPONT]
        object.Y = object.YSS[NPONT]
        object.Z = object.ZSS[NPONT]
        object.ST = object.TSS[NPONT]
        E1 = object.ESS[NPONT]
        DCX1 = object.DCXS[NPONT]
        DCY1 = object.DCYS[NPONT]
        DCZ1 = object.DCZS[NPONT]
        IZPLANE = <int> (object.IPLS[NPONT])
        NPONT -= 1
        ZSTRT = object.Z
        TSSTRT = object.ST
        if object.Z > object.ZFINAL:
            EPOT = object.EFIELD * (object.Z - object.ZFINAL) * 100
            if E1 < EPOT:
                NELEC += 1
                ISOL = 1
                continue
        R = TCALCT(object, DCZ1, E1, TZSTOP, TZSTOP1, ISOL, IZPLANE)
        ISOL = R[0]
        IZPLANE = int(R[1])
        TZSTOP = R[2]
        TZSTOP1 = R[3]
        if TZSTOP == -99:
            NELEC += 1
            ISOL = 1
            continue

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz object):
    cdef double *EPRM, *IESPECP, RNMA
    EPRM = <double *> malloc(10000000 * sizeof(double))
    memset(EPRM, 0, 10000000 * sizeof(double))
    IESPECP = <double *> malloc(100 * sizeof(double))
    memset(IESPECP, 0, 100 * sizeof(double))
    cdef double ** TEMP = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TEMP[i] = <double *> malloc(4000 * sizeof(double))
    cdef int f700 = 0

    cdef int I, I100, K, ID, NCOL, NELEC, NEION, NMXADD, NTMPFLG, NPONT, NCLUS, J1, IMBPT, JPRINT, IPRINT, ITER, IPLANE, IDUM, KGAS, IE, IT, NCLTMP, IPT
    cdef int KDUM, JCT, JFL, IZPLANE, IDM1
    cdef double S, RDUM, E1, CONST9, CONST10, ZSTRT, TSSTRT, ABSFAKEI, DCX1, DCY1, DCZ1, E100, DCZ100, DCY100, DCX100, BP, F1, F2, F4, TDASH, AP
    cdef double TSTOP, R1, T, E, R2, VGX, VGY, VGZ, CONST6, DCX2, DCY2, DCZ2, VEX, VEY, VEZ, TEST1, R5, EOK, TEST2, R9, TEST3, A, DXCOM, DYCOM, DZCOM
    cdef double CONST11, T2, CONST7, EI, S1, EISTR, ESEC, NAUG, EAVAUG, R3, F3, F6, F5, R4, PHI0, F8, F9, TPEN, RAN, TSTOP1, IPLANE1, S2, R31, THETA0
    cdef double ARG1, D, U, CSQD, ARGZ, F5S, F6S, THSEC, RAN1, Q, PHIS, F8S, F9S, CONST12, VXLAB, VYLAB, VZLAB, W, EPRMBAR, E2PRM, EBAR, EERR, EPOT
    cdef double TZSTOP, TLFT, ISOL, TZSTOP1, R[4]

    global NELEC, NCLUS, E1, DCX1, DCY1, DCZ1, IZPLANE, NPONT, ZSTRT, TSSTRT, EPOT, ISOL, R, TZSTOP, TZSTOP1, ISOL, f700, EPRM, DCZ100, DCX100, DCY100, E100, ITER

    I = 0
    S = 0.0
    object.ST = 0.0
    object.X = 0.0
    object.Y = 0.0
    object.Z = 0.0
    I100 = 0
    ZTOT = 0.0
    ZTOTS = 0.0
    TTOT = 0.0
    TTOTS = 0.0
    object.SMALL = 1e-20
    object.TMAX1 = 0.0
    RDUM = object.RSTART
    E1 = object.ESTART
    CONST9 = object.CONST3 * 0.01
    CONST10 = CONST9 ** 2
    object.API = acos(-1)

    for I in range(300):
        object.TIME[I] = 0.0

    for K in range(6):
        for I in range(5):
            object.ICOLL[K][I] = 0

    for K in range(6):
        for I in range(290):
            object.ICOLN[K][I] = 0

    for K in range(6):
        for I in range(10):
            object.ICOLNN[K][I] = 0
    for I in range(4000):
        object.SPEC[I] = 0

    for I in range(8):
        object.ESPL[I] = 0.0
        object.XSPL[I] = 0.0
        object.YSPL[I] = 0.0
        object.ZSPL[I] = 0.0
        object.TSPL[I] = 0.0
        object.XXSPL[I] = 0.0
        object.YYSPL[I] = 0.0
        object.ZZSPL[I] = 0.0
        object.TSSUM[I] = 0.0
        object.VZSPL[I] = 0.0
        object.TSSUM2[I] = 0.0
        object.TTMSPL[I] = 0.0
        object.TMSPL[I] = 0.0
        object.RSPL[I] = 0.0
        object.RRSPL[I] = 0.0
        object.RRSPM[I] = 0.0
        object.NESST[I] = 0
    object.NESST[8] = 0
    ID = 0
    NCOL = 0
    object.NNULL = 0
    NELEC = 0
    NEION = 0
    NMXADD = 0
    NTMPFLG = 0
    NPONT = 0
    NCLUS = 0
    J1 = 1
    ZSTRT = 0.0
    TSSTRT = 0.0
    for K in range(6):
        for J in range(4000):
            TEMP[K][J] = object.TCF[K][J] + object.TCFN[K][J]

    ABSFAKEI = abs(object.FAKEI)
    object.IFAKE = 0

    for J in range(8):
        object.IFAKED[J] = 0
    GERJAN(object.RSTART, object.API, object.RNMX)

    DCZ1 = cos(object.THETA)
    DCX1 = sin(object.THETA) * cos(object.PHI)
    DCY1 = sin(object.THETA) * sin(object.PHI)
    E100 = E1
    DCZ100 = DCZ1
    DCX100 = DCX1
    DCY100 = DCY1
    BP = object.EFIELD ** 2 * object.CONST1
    F1 = object.EFIELD * object.CONST2
    F2 = object.EFIELD * object.CONST3
    F4 = 2 * object.API

    IMBPT = 0
    JPRINT = <int> (object.NMAX / 10)
    IPRINT = 0
    ITER = 0
    ISOL = 0
    object.IPRIM = 0
    TZSTOP = 1000

    object.IPRIM += 1
    IZPLANE = 0
    TZSTOP = 1000
    if object.IPRIM > 1:
        if ITER > object.NMAX:
            object.IPRIM -= 1
            f700 = 1
        else:
            object.X = 0.0
            object.Y = 0.0
            object.Z = 0.0
            DCZ1 = DCZ100
            DCX1 = DCX100
            DCY1 = DCY100
            E1 = E100
            NCLUS += 1
            object.ST = 0.0
            TSSTRT = 0.0
            ZSTRT = 0.0
    # IF PRIM >10000000: TOO MANY PRIMARIES
    if f700 == 0:
        EPRM[<int> (object.IPRIM - 1)] = E1
        IDUM = int(E1)
        IDUM = min(IDUM, 99)
        IESPECP[IDUM] += 1
        TDASH = 0.0
        NELEC += 1
        TZSTOP1 = 0

    while (f700 == 0):
        R1 = random_uniform(RDUM)
        T = -1 * log(R1) / object.TCFMX + TDASH
        TOLD = TDASH
        TDASH = T
        AP = DCZ1 * F2 * sqrt(E1)
        while (1):
            if T >= TZSTOP and TOLD < TZSTOP:
                TLFT = TZSTOP
                SPLANET(object, T, E1, DCX1, DCY1, DCZ1, AP, BP, TLFT, IZPLANE)
                if IZPLANE >= object.IZFINAL + 1:
                    while (1):
                        ZTOT += object.Z
                        TTOT += object.ST
                        ZTOTS += object.Z - ZSTRT
                        TTOTS += object.ST - TSSTRT
                        if NELEC == NCLUS + 1:
                            GOTO544(object)
                            continue
                        object.X = object.XSS[NPONT]
                        object.Y = object.YSS[NPONT]
                        object.Z = object.ZSS[NPONT]
                        object.ST = object.TSS[NPONT]
                        E1 = object.ESS[NPONT]
                        DCX1 = object.DCXS[NPONT]
                        DCY1 = object.DCYS[NPONT]
                        DCZ1 = object.DCZS[NPONT]
                        IZPLANE = <int> (object.IPLS[NPONT])
                        NPONT -= 1
                        ZSTRT = object.Z
                        TSSTRT = object.ST
                        if object.Z > object.ZFINAL:
                            EPOT = object.EFIELD * (object.Z - object.ZFINAL) * 100
                            if E1 < EPOT:
                                NELEC += 1
                                ISOL = 1
                                continue
                        R = TCALCT(object, DCZ1, E1, TZSTOP, TZSTOP1, ISOL, IZPLANE)
                        ISOL = R[0]
                        IZPLANE = int(R[1])
                        TZSTOP = R[2]
                        TZSTOP1 = R[3]
                        if TZSTOP == -99:
                            NELEC += 1
                            ISOL = 1
                            continue
                    GOTO555()
                    continue
                if ISOL == 2:
                    TZSTOP = TZSTOP1
                    ISOL = 1
                    continue
        E = E1 + (AP + BP * T) * T
        if E < 0:
            E = 0.001

        R2 = random_uniform(RDUM)
        if object.NumberOfGases == 1:
            KGAS = 0
        while (object.TCFMXG[KGAS] < R2):
            KGAS = KGAS + 1

        IMBPT += 1
        if (IMBPT > 5):
            GERJAN(object.RSTART, object.API, object.RNMX)
            IMBPT = 0
        VGX = object.VTMB[KGAS] * object.RNMX[IMBPT % 6]
        IMBPT += 1
        VGY = object.VTMB[KGAS] * object.RNMX[IMBPT % 6]
        IMBPT += 1
        VGZ = object.VTMB[KGAS] * object.RNMX[IMBPT % 6]

        CONST6 = sqrt(E1 / E)

        DCX2 = DCX1 * CONST6
        DCY2 = DCY1 * CONST6
        DCZ2 = DCZ1 * CONST6 + object.EFIELD * T * object.CONST5 / sqrt(E)

        AIS = 1
        if DCZ2 < 0:
            AIS = -1
        DCZ2 = AIS * sqrt(1 - DCX2 ** 2 - DCY2 ** 2)

        VTOT = CONST9 * sqrt(E)
        VEX = DCX2 * VTOT
        VEY = DCY2 * VTOT
        VEZ = DCZ2 * VTOT

        EOK = ((VEX - VGX) ** 2 + (VEY - VGY) ** 2 + (VEZ - VGZ) ** 2) / CONST10
        IE = <int> (EOK / object.ESTEP)
        IE = min(IE, 3999)
        R5 = random_uniform(RDUM)
        TEST1 = object.TCF[KGAS][IE] / object.TCFMAX[KGAS]
        if R5 > TEST1:
            object.NNULL += 1
            TEST2 = TEMP[KGAS][IE] / object.TCFMAX[KGAS]
            if R5 < TEST2:
                if object.NPLAST[KGAS] == 0:
                    continue
                R2 = random_uniform(RDUM)
                I = 0
                while object.CFN[KGAS][IE][I] < R2:
                    I += 1

                object.ICOLNN[KGAS][I] += 1
                continue
            else:
                TEST3 = (TEMP[KGAS][IE] + ABSFAKEI) / object.TCFMAX[KGAS]
                if R5 < TEST3:
                    # FAKE IONISATION INCREMENT COUNTER
                    object.IFAKE += 1
                    object.IFAKED[IZPLANE] += 1
                    if object.FAKEI < 0.0:
                        NEION += 1
                        if NELEC == NCLUS + 1:
                            GOTO544(object)
                            continue
                        GOTO18n20(object,1)
                        GOTO555()
                        continue
                NCLUS += 1
                NPONT += 1
                NMXADD = max(NPONT, NMXADD)
                if NPONT >= 2000:
                    raise ValueError("NPONT>2000")
                A = T * CONST9 * sqrt(E1)
                object.XSS[NPONT] = object.X + DCX1 * A
                object.YSS[NPONT] = object.Y + DCY1 * A
                object.ZSS[NPONT] = object.Z + DCZ1 * A + T * T * F1
                object.TSS[NPONT] = object.ST + T
                object.ESS[NPONT] = E
                object.DCXS[NPONT] = DCX2
                object.DCYS[NPONT] = DCY2
                AIS = 1.0
                if DCZ2 < 0.0:
                    AIS = -1
                object.DCZS[NPONT] = AIS * sqrt(1 - DCX2 ** 2 - DCY2 ** 2)
                IDM1 = 1 + int(object.ZSS[NPONT] / object.ZSTEP)
                if IDM1 < 1:
                    IDM1 = 1
                if IDM1 > 9:
                    IDM1 = 9
                object.IPLS[NPONT] = IDM1
                object.NESST[int(object.IPLS[int(NPONT)] - 1)] += 1
                continue
            continue
        NCOL += 1
        CONST11 = 1 / (CONST9 * sqrt(EOK))
        DXCOM = (VEX - VGX) * CONST11
        DYCOM = (VEY - VGY) * CONST11
        DZCOM = (VEZ - VGZ) * CONST11

        T2 = T ** 2

        if (T >= object.TMAX1):
            object.TMAX1 = T
        TDASH = 0.0

        CONST7 = CONST9 * sqrt(E1)
        A = T * CONST7
        NCOL += 1
        CZ1 = DCZ1 * CONST7
        object.X += DCX1 * A
        object.Y += DCY1 * A
        object.Z += DCZ1 * A + T2 * F1
        object.ST += T
        IT = int(T)
        IT = min(IT, 299)
        object.TIME[IT] += 1
        object.SPEC[IE] += 1

        CX1 = DCX1 * CONST7
        CY1 = DCY1 * CONST7
        R2 = random_uniform(RDUM)
        I = MBSortT(KGAS, I, R2, IE, object)

        while object.CF[KGAS][IE][I] < R2:
            I += 1
        S1 = object.RGAS[KGAS][I]
        EI = object.EIN[KGAS][I]
        if EOK < EI:
            EI = EOK - 0.0001

        if object.IPN[KGAS][I] != 0:
            if object.IPN[KGAS][I] == -1:
                NEION += 1
                IPT = int(object.IARRY[KGAS][I])
                ID += 1
                ITER += 1
                IPRINT += 1
                object.ICOLL[KGAS][int(IPT)] += 1
                object.ICOLN[KGAS][I] += 1
                IT = int(T)
                IT = min(IT, 299)
                object.TIME[IT] += 1
                ZTOT += object.Z
                TTOT += object.ST
                ZTOTS += object.Z - ZSTRT
                TTOTS += object.ST - TSSTRT
                IDM1 = 1 + int(object.Z / object.ZSTEP)
                if IDM1 < 1:
                    IDM1 = 1
                if IDM1 > 9:
                    IDM1 = 9
                object.NESST[IDM1 - 1] -= 1
                if NELEC == NCLUS + 1:
                    GOTO544(object)
                    continue
                GOTO18n20(object,1)
                GOTO555()
                continue
            EISTR = EI
            R9 = random_uniform(RDUM)
            ESEC = object.WPL[KGAS][I] * tan(R9 * atan((EOK - EI) / (2 * object.WPL[KGAS][I])))
            ESEC = object.WPL[KGAS][I] * (ESEC / object.WPL[KGAS][I]) ** 0.9524
            EI = ESEC + EI
            NCLUS += 1
            NPONT += 1
            NMXADD = max(NPONT, NMXADD)
            if NPONT >= 2000:
                raise ValueError("NPONT>2000")
            object.XSS[NPONT] = object.X
            object.YSS[NPONT] = object.Y
            object.ZSS[NPONT] = object.Z
            object.TSS[NPONT] = object.ST
            object.ESS[NPONT] = ESEC
            NTMPFLG = 1
            NCLTMP = NPONT
            IDM1 = 1 + int(object.Z / object.ZSTEP)
            if IDM1 < 1:
                IDM1 = 1
            if IDM1 > 9:
                IDM1 = 9
            object.IPLS[NPONT] = IDM1
            object.NESST[IDM1 - 1] += 1
            if EISTR > 30:
                NAUG = object.NC0[KGAS][I]
                EAVAUG = object.EC0[KGAS][I] / float(NAUG)
                for JFL in range(int(NAUG)):
                    NCLUS += 1
                    NPONT += 1
                    object.XSS[NPONT] = object.X
                    object.YSS[NPONT] = object.Y
                    object.ZSS[NPONT] = object.Z
                    object.TSS[NPONT] = object.ST
                    object.ESS[NPONT] = EAVAUG
                    R3 = random_uniform(RDUM)
                    F3 = 1 - 2 * R3
                    THETA0 = acos(F3)
                    F6 = cos(THETA0)
                    F5 = sin(THETA0)
                    R4 = random_uniform(RDUM)
                    PHI0 = F4 * R4
                    F8 = sin(PHI0)
                    F9 = cos(PHI0)
                    object.DCXS[NPONT] = F9 * F5
                    object.DCYS[NPONT] = F8 * F5
                    object.DCZS[NPONT] = F6
                    IDM1 = 1 + int(object.Z / object.ZSTEP)
                    if IDM1 < 1:
                        IDM1 = 1
                    if IDM1 > 9:
                        IDM1 = 9
                    object.IPLS[NPONT] = IDM1
                    object.NESST[IDM1 - 1] += 1
        IPT = int(object.IARRY[KGAS][I])
        ID += 1
        ITER += 1
        IPRINT += 1
        object.ICOLL[KGAS][int(IPT)] += 1
        object.ICOLN[KGAS][I] += 1

        if object.EnablePenning != 0:
            if object.PENFRA[KGAS][0][I] != 0.0:
                RAN = random_uniform(RDUM)
                if RAN <= object.PENFRA[KGAS][0][I]:
                    NCLUS += 1
                    NPONT += 1
                    if NPONT >= 2000:
                        raise ValueError("NPONT>2000")
                    if object.PENFRA[KGAS][1][I] == 0.0:
                        object.XSS[NPONT] = object.X
                        object.YSS[NPONT] = object.Y
                        object.ZSS[NPONT] = object.Z
                        if object.ZSS[NPONT] > object.ZFINAL or object.ZSS[NPONT] < 0:
                            NPONT -= 1
                            NCLUS -= 1
                        else:
                            TPEN = object.ST
                            if object.PENFRA[KGAS][2][I] != 0:
                                RAN = random_uniform(RDUM)
                                TPEN = object.ST - log(RAN) * object.PENFRA[KGAS][2][I]
                            object.TSS[NPONT] = TPEN
                            object.ESS[NPONT] = 1.0
                            object.DCXS[NPONT] = DCX1
                            object.DCYS[NPONT] = DCY1
                            object.DCZS[NPONT] = DCZ1
                            IDM1 = 1 + int(object.Z / object.ZSTEP)
                            if IDM1 < 1:
                                IDM1 = 1
                            if IDM1 > 9:
                                IDM1 = 9
                            object.IPLS[NPONT] = IDM1
                            object.NESST[IDM1 - 1] += 1
                            NPONT -= 1
                            NCLUS -= 1
                    else:
                        ASIGN = 1.0
                        RAN = random_uniform(RDUM)
                        RAN1 = random_uniform(RDUM)
                        if RAN1 < 0.5:
                            ASIGN = -1 * ASIGN
                        object.XSS[NPONT] = object.X - log(RAN) * object.PENFRA[KGAS][1][I] * ASIGN
                        RAN = random_uniform(RDUM)
                        RAN1 = random_uniform(RDUM)
                        if RAN1 < 0.5:
                            ASIGN = -1 * ASIGN
                        object.YSS[NPONT] = object.Y - log(RAN) * object.PENFRA[KGAS][1][I] * ASIGN
                        RAN = random_uniform(RDUM)
                        RAN1 = random_uniform(RDUM)
                        if RAN1 < 0.5:
                            ASIGN = -1 * ASIGN
                        object.ZSS[NPONT] = object.Z - log(RAN) * object.PENFRA[KGAS][1][I] * ASIGN
                        if object.ZSS[NPONT] > object.ZFINAL or object.ZSS[NPONT] < 0.0:
                            NPONT -= 1
                            NCLUS -= 1
                        else:
                            TPEN = object.ST

                            if object.PENFRA[KGAS][2][I] != 0:
                                RAN = random_uniform(RDUM)
                                TPEN = object.ST - log(RAN) * object.PENFRA[KGAS][2][I]
                            object.TSS[NPONT] = TPEN
                            object.ESS[NPONT] = 1.0
                            object.DCXS[NPONT] = DCX1
                            object.DCYS[NPONT] = DCY1
                            object.DCZS[NPONT] = DCZ1
                            IDM1 = 1 + int(object.Z / object.ZSTEP)
                            if IDM1 < 1:
                                IDM1 = 1
                            if IDM1 > 9:
                                IDM1 = 9
                            object.IPLS[NPONT] = IDM1
                            object.NESST[IDM1 - 1] += 1

        S2 = (S1 ** 2) / (S1 - 1)
        R3 = random_uniform(RDUM)
        if object.INDEX[KGAS][I] == 1:
            R31 = random_uniform(RDUM)
            F3 = 1.0 - R3 * object.ANGCT[KGAS][IE][I]
            if R31 > object.PSCT[KGAS][IE][I]:
                F3 = -1 * F3
        elif object.INDEX[KGAS][I] == 2:
            EPSI = object.PSCT[KGAS][IE][I]
            F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
        else:
            F3 = 1 - 2 * R3
        THETA0 = acos(F3)
        R4 = random_uniform(RDUM)
        PHI0 = F4 * R4
        F8 = sin(PHI0)
        F9 = cos(PHI0)
        if EOK < EI:
            EI = EOK - 0.0001
        ARG1 = 1 - S1 * EI / EOK
        ARG1 = max(ARG1, object.SMALL)
        D = 1 - F3 * sqrt(ARG1)
        E1 = EOK * (1 - EI / (S1 * EOK) - 2 * D / S2)
        E1 = max(E1, object.SMALL)
        Q = sqrt((EOK / E1) * ARG1) / S1
        Q = min(Q, 1)
        object.THETA = asin(Q * sin(THETA0))
        F6 = cos(object.THETA)
        U = (S1 - 1) * (S1 - 1) / ARG1

        CSQD = F3 * F3

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6
        F5 = sin(object.THETA)
        DZCOM = min(DZCOM, 1)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
        if ARGZ == 0:
            DCZ1 = F6
            DCX1 = F9 * F5
            DCY1 = F8 * F5
            if NTMPFLG == 1:
                F5S = F5 * sqrt(E1 / object.ES[NCLTMP])
                if F5S > 1.0:
                    F5S = 1.0
                THSEC = asin(F5S)
                F5S = sin(THSEC)
                F6S = sin(THSEC)
                if F6S < 0:
                    F6S *= -1
                PHIS = PHI0 + object.API
                if PHIS > F4:
                    PHIS = PHI0 - F4
                F8S = sin(PHIS)
                F9S = cos(PHIS)
                object.DCZS[NCLTMP] = F6S
                object.DCXS[NCLTMP] = F9S * F5S
                object.DCYS[NCLTMP] = F8S * F5S
                NTMPFLG = 0
        else:
            DCZ1 = DZCOM * F6 + ARGZ * F5 * F8
            DCY1 = DYCOM * F6 + (F5 / ARGZ) * (DXCOM * F9 - DYCOM * DZCOM * F8)
            DCX1 = DXCOM * F6 - (F5 / ARGZ) * (DYCOM * F9 + DXCOM * DZCOM * F8)
            if NTMPFLG == 1:
                F5S = F5 * sqrt(E1 / object.ES[NCLTMP])
                if F5S > 1.0:
                    F5S = 1.0
                THSEC = asin(F5S)
                F5S = sin(THSEC)
                F6S = sin(THSEC)
                if F6S < 0:
                    F6S *= -1
                PHIS = PHI0 + object.API
                if PHIS > F4:
                    PHIS = PHI0 - F4
                F8S = sin(PHIS)
                F9S = cos(PHIS)
                object.DCZS[NCLTMP] = DZCOM * F6S + ARGZ * F5S * F8S
                object.DCYS[NCLTMP] = DYCOM * F6S + (F5S / ARGZ) * (DXCOM * F9S - DYCOM * DZCOM * F8S)
                object.DCXS[NCLTMP] = DXCOM * F6S - (F5S / ARGZ) * (DYCOM * F9S + DXCOM * DZCOM * F8S)
                NTMPFLG = 0
        CONST12 = CONST9 * sqrt(E1)
        VXLAB = DCX1 * CONST12 + VGX
        VYLAB = DCY1 * CONST12 + VGY
        VZLAB = DCZ1 * CONST12 + VGZ
        E1 = (VXLAB ** 2 + VYLAB ** 2 + VZLAB ** 2)
        CONST11 = 1 / (CONST9 * sqrt(E1))
        DCX1 = VXLAB * CONST11
        DCY1 = VYLAB * CONST11
        DCZ1 = VZLAB * CONST11
        I100 += 1
        if I100 == 200:
            DCZ100 = DCZ1
            DCX100 = DCX1
            DCY100 = DCY1
            E100 = E1
            I100 = 0
        if object.Z > object.ZFINAL:
            EPOT = object.EFIELD * (object.Z - object.ZFINAL) * 100
            if E1 < EPOT:
                GOTO18n20(object,0)
                GOTO555()
                continue
        R = TCALCT(object, DCZ1, E1, TZSTOP, TZSTOP1, ISOL, IZPLANE)
        ISOL = R[0]
        IZPLANE = int(R[1])
        TZSTOP = R[2]
        TZSTOP1 = R[3]
        if TZSTOP == -99:
            GOTO18n20(object,0)
            GOTO555()
            continue
        if IPRINT <= JPRINT:
            continue
        IPRINT = 0
        W = ZTOTS / TTOTS
        W *= 1e9
        XID = float(ID)
        JCT = ID / 100000
        J1 += 1

    XID = float(ID)
    if NELEC > object.IPRIM:
        ANEION = float(NEION)
        ANBT = float(NELEC - object.IPRIM)
        ATTOINT = ANEION / ANBT
        object.ATTERT = sqrt(ANEION) / ANEION
        object.AIOERT = sqrt(ANBT) / ANBT
    else:
        ANEION = float(NEION)
        ATTOINT = -1
        object.ATTERT = sqrt(ANEION) / ANEION
    JCT = ID / 100000

    EPRMBAR = 0.0
    E2PRM = 0.0
    if object.IPRIM == 1:
        return
    for I in range(int(object.IPRIM)):
        E2PRM = E2PRM + EPRM[I] * EPRM[I]
        EPRMBAR += EPRM[I]
    EBAR = EPRMBAR / (object.IPRIM)
    EERR = sqrt(E2PRM / (object.IPRIM) - EBAR ** 2)
    # IF ITER >NMAX
