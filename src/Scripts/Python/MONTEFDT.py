import numpy as np
import math
from GERJAN import GERJAN
from SORTT import SORTT
from TPLANET import TPLANET


def MONTEFDT(Magboltz):
    EPRM = np.zeros(10000000)
    IESPECP = np.zeros(100)
    TEMP = np.zeros(shape=(6, 4000))

    S = 0.0
    Magboltz.ST = 0.0
    Magboltz.X = 0.0
    Magboltz.Y = 0.0
    Magboltz.Z = 0.0
    Magboltz.ZTOT = 0.0
    Magboltz.ZTOTS = 0.0
    Magboltz.TTOT = 0.0
    Magboltz.TTOTS = 0.0
    Magboltz.SMALL = 1e-20
    Magboltz.TMAX1 = 0.0
    RDUM = Magboltz.RSTART
    E1 = Magboltz.ESTART
    CONST9 = Magboltz.CONST3 * 0.01
    CONST10 = CONST9 ** 2
    Magboltz.API = math.acos(-1)

    for I in range(300):
        Magboltz.TIME[I] = 0.0

    for K in range(6):
        for I in range(5):
            Magboltz.ICOLL[K][I] = 0

    for K in range(6):
        for I in range(290):
            Magboltz.ICOLN[K][I] = 0

    for K in range(6):
        for I in range(10):
            Magboltz.ICOLNN[K][I] = 0
    for I in range(4000):
        Magboltz.SPEC[I] = 0

    for I in range(8):
        Magboltz.ESPL[I] = 0.0
        Magboltz.XSPL[I] = 0.0
        Magboltz.YSPL[I] = 0.0
        Magboltz.ZSPL[I] = 0.0
        Magboltz.TSPL[I] = 0.0
        Magboltz.XXSPL[I] = 0.0
        Magboltz.YYSPL[I] = 0.0
        Magboltz.ZZSPL[I] = 0.0
        Magboltz.TSSUM[I] = 0.0
        Magboltz.TSSUM2[I] = 0.0
        Magboltz.TTMSPL[I] = 0.0
        Magboltz.TMSPL[I] = 0.0
        Magboltz.RSPL[I] = 0.0
        Magboltz.RRSPL[I] = 0.0
        Magboltz.RRSPM[I] = 0.0
        Magboltz.NESST[I] = 0
    Magboltz.NESST[8] = 0
    ID = 0
    NCOL = 0
    Magboltz.NNULL = 0
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
            TEMP[K][J] = Magboltz.TCF[K][J] + Magboltz.TCFN[K][J]

    ABSFAKEI = abs(Magboltz.FAKEI)
    Magboltz.IFAKE = 0

    for J in range(8):
        Magboltz.IFAKED[J] = 0
    Magboltz.RNMX = GERJAN(Magboltz.RAND48, Magboltz.API)

    DCZ1 = math.cos(Magboltz.THETA)
    DCX1 = math.sin(Magboltz.THETA) * math.cos(Magboltz.PHI)
    DCY1 = math.sin(Magboltz.THETA) * math.sin(Magboltz.PHI)
    E100 = E1
    DCZ100 = DCZ1
    DCX100 = DCX1
    DCY100 = DCY1
    BP = Magboltz.EFIELD ** 2 * Magboltz.CONST1
    F1 = Magboltz.EFIELD * Magboltz.CONST2
    F2 = Magboltz.EFIELD * Magboltz.CONST3
    F4 = 2 * Magboltz.API

    IMBPT = 0
    JPRINT = Magboltz.NMAX / 10
    IPRINT = 0
    ITER = 0

    Magboltz.IPRIM = 0
    TZSTOP = 1000

    while (True):
        Magboltz.IPRIM += 1
        IZPLANE = 0
        TZSTOP = 1000
        if Magboltz.IPRIM > 1:
            if ITER > Magboltz.NMAX:
                Magboltz.IPRIM -= 1
                break

            Magboltz.X = 0.0
            Magboltz.Y = 0.0
            Magboltz.Z = 0.0
            DCZ1 = DCZ100
            DCX1 = DCX100
            DCY1 = DCY100
            E1 = E100
            NCLUS += 1
            Magboltz.ST = 0.0
            TSSTRT = 0.0
            ZSTRT = 0.0
            IPLANE = 0
        # IF PRIM >10000000: TOO MANY PRIMARIES
        EPRM[Magboltz.IPRIM - 1] = E1
        IDUM = int(E1)
        IDUM = min(IDUM, 99)
        IESPECP[IDUM] += 1
        GT1F = 0
        while (True):
            # 555
            if GT1F == 0:
                TDASH = 0.0
                NELEC += 1
            GT1F = 0
            R1 = Magboltz.RAND48.drand()
            T = -1 * np.log(R1) / Magboltz.TCFMX + TDASH
            TOLD = TDASH
            TDASH = T
            AP = DCZ1 * F2 * math.sqrt(E1)
            if T >= TZSTOP and TOLD < TZSTOP:
                TLFT = TZSTOP
                Magboltz = SPLANET(Magboltz, T, E1, DCX1, DCY1, DCZ1, AP, BP,TLFT, IZPLANE)
                if IZPLANE>=Magboltz.IZFINAL+1:
                    Magboltz.ZTOT += Magboltz.Z
                    Magboltz.TTOT += Magboltz.ST
                    Magboltz.ZTOTS += Magboltz.Z - ZSTRT
                    Magboltz.TTOTS += Magboltz.ST - TSSTRT
                    if NELEC == NCLUS + 1:
                        break
                    Magboltz.X = Magboltz.XSS[NPONT]
                    Magboltz.Y = Magboltz.YSS[NPONT]
                    Magboltz.Z = Magboltz.ZSS[NPONT]
                    Magboltz.ST = Magboltz.TSS[NPONT]
                    E1 = Magboltz.ESS[NPONT]
                    DCX1 = Magboltz.DCXS[NPONT]
                    DCY1 = Magboltz.DCYS[NPONT]
                    DCZ1 = Magboltz.DCZS[NPONT]
                    IZPLANE = Magboltz.IPLS[NPONT]
                    NPONT -= 1
                    ZSTRT = Magboltz.Z
                    TSSTRT = Magboltz.ST