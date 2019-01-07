import numpy as np
import math
from COLF import COLF
from COLFT import COLFT


def SST(Magboltz):
    ESST = np.zeros(8)
    VDSST = np.zeros(8)
    WSSST = np.zeros(8)
    DXSST = np.zeros(8)
    DYSST = np.zeros(8)
    WTEMP = np.zeros(8)
    DRSST = np.zeros(8)
    ALFNE = np.zeros(8)
    ALFNJ = np.zeros(8)
    ALFN = np.zeros(8)
    ZSST = np.zeros(8)
    DLSST = np.zeros(8)
    DRSS1 = np.zeros(8)
    DRSS2 = np.zeros(8)
    DRSS3 = np.zeros(8)
    ALFEX1 = np.zeros(8)
    NEPL = np.zeros(8)

    Magboltz.VDOUT = 0.0
    Magboltz.VDERR = 0.0
    Magboltz.WSOUT = 0.0
    Magboltz.WSERR = 0.0
    Magboltz.DLOUT = 0.0
    Magboltz.DLERR = 0.0
    Magboltz.DTOUT = 0.0
    Magboltz.DTERR = 0.0
    Magboltz.ALPHSST = 0.0
    Magboltz.ALPHERR = 0.0
    Magboltz.ATTSST = 0.0
    Magboltz.ATTERR = 0.0
    JPRINT = Magboltz.IZFINAL
    NEPL[0] = Magboltz.IPRIM  + Magboltz.NESST[0]
    for K in range(1, JPRINT):
        NEPL[K] = NEPL[K - 1] + Magboltz.NESST[K]
    for K in range(JPRINT):
        Magboltz.NESST[K] = NEPL[K]

    for I in range(JPRINT):
        if Magboltz.NESST[I] == 0:
            JPRINT = I - 1
            break
    ESST[0] = Magboltz.ESPL[0] / Magboltz.TSSUM[0]
    ZSST[0] = Magboltz.ZSPL[0] / Magboltz.TSSUM[0]
    VDSST[0] = Magboltz.VZSPL[0] / Magboltz.TSSUM[0]
    WTEMP[0] = Magboltz.ZSTEP * Magboltz.TSSUM[0] / Magboltz.TMSPL[0]
    WSSST[0] = WTEMP[0]
    DXSST[0] = ((Magboltz.XXSPL[0] / Magboltz.TSSUM[0]) - (Magboltz.XSPL[0] / Magboltz.TSSUM[0]) ** 2) * WSSST[0] / (
            2 * Magboltz.ZSTEP)
    DYSST[0] = ((Magboltz.YYSPL[0] / Magboltz.TSSUM[0]) - (Magboltz.YSPL[0] / Magboltz.TSSUM[0]) ** 2) * WSSST[0] / (
            2 * Magboltz.ZSTEP)
    DLSST[0] = ((Magboltz.TTMSPL[0] / Magboltz.TSSUM[0]) - (Magboltz.TMSPL[0] / Magboltz.TSSUM[0]) ** 2) * WSSST[
        0] ** 3 / (
                       2 * Magboltz.ZSTEP)
    DUM = np.zeros(6)
    if Magboltz.ITHRM == 1:
        DUM = COLFT(Magboltz)
    if Magboltz.ITHRM == 0:
        DUM = COLF(Magboltz)
    FRION = DUM[2]
    FRATT = DUM[3]
    ATTOINT = FRATT / FRION
    CORERR = abs((Magboltz.FAKEI + FRION - FRATT) / (FRION - FRATT))
    CORF = (FRION - FRATT) / (Magboltz.FAKEI + FRION - FRATT)
    if Magboltz.NESST[0] != 0:
        ALFNE[0] = CORF * (np.log(float(Magboltz.NESST[0])) - np.log(float(Magboltz.IPRIM ))) / Magboltz.ZSTEP
    ALFNJ[0] = 0.0
    ALFN[0] = 0.0
    if JPRINT == 8:
        JPRT = JPRINT
    else:
        JPRT = JPRINT + 1
    for I in range(1, JPRT):
        ESST[I] = Magboltz.ESPL[I] / Magboltz.TSSUM[I]
        ZSST[I] = Magboltz.ZSPL[I] / Magboltz.TSSUM[I]
        VDSST[I] = Magboltz.VZSPL[I] / Magboltz.TSSUM[I]
        WTEMP[I] = Magboltz.ZSTEP * float(I + 1) * Magboltz.TSSUM[I] / Magboltz.TMSPL[I]
        WSSST[I] = (WTEMP[I] * WTEMP[I - 1]) / ((I + 1) * WTEMP[I - 1] - I * WTEMP[I])
        DXSST[I] = ((Magboltz.XXSPL[I] / Magboltz.TSSUM[I]) - (Magboltz.XSPL[I] / Magboltz.TSSUM[I]) ** 2 - (
                Magboltz.XXSPL[I - 1] / Magboltz.TSSUM[I - 1]) + (
                            Magboltz.XSPL[I - 1] / Magboltz.TSSUM[I - 1]) ** 2) * WSSST[I] / (2 * Magboltz.ZSTEP)
        DYSST[I] = ((Magboltz.YYSPL[I] / Magboltz.TSSUM[I]) - (Magboltz.YSPL[I] / Magboltz.TSSUM[I]) ** 2 - (
                Magboltz.YYSPL[I - 1] / Magboltz.TSSUM[I - 1]) + (
                            Magboltz.YSPL[I - 1] / Magboltz.TSSUM[I - 1]) ** 2) * WSSST[I] / (2 * Magboltz.ZSTEP)

        DLSST[I] = ((Magboltz.TTMSPL[I] / Magboltz.TSSUM[I]) - (Magboltz.TMSPL[I] / Magboltz.TSSUM[I]) ** 2 - (
                Magboltz.TTMSPL[I - 1] / Magboltz.TSSUM[I - 1]) + (
                            Magboltz.TMSPL[I - 1] / Magboltz.TSSUM[I - 1]) ** 2) * WSSST ** 3 / (2 * Magboltz.ZSTEP)

        ALFN[I] = (np.log(Magboltz.TSSUM[I]) - np.log(Magboltz.TSSUM[I - 1])) / Magboltz.ZSTEP
        ALFNJ[I] = (np.log(Magboltz.TSSUM[I] * VDSST[I]) - np.log(
            Magboltz.TSSUM[I - 1] * VDSST[I - 1])) / Magboltz.ZSTEP
        ALFNE[I] = 0.0

        # weird if statement

        ALFNE[I] = (np.log(float(Magboltz.NESST[I])) - np.log(float(Magboltz.NESST[I - 1]))) / Magboltz.ZSTEP
        ALFN[I] = CORF * ALFN[I]
        ALFNE[I] *= CORF
        ALFNJ[I] *= CORF

    if JPRINT == 8:
        JPRINT -= 1

    DXFIN = ((Magboltz.XXSPL[JPRINT] / Magboltz.TSSUM[JPRINT]) - (
            Magboltz.XSPL[JPRINT] / Magboltz.TSSUM[JPRINT]) ** 2) * WSSST[JPRINT] / (
                    (JPRINT + 1) * 2 * Magboltz.ZSTEP)
    DXFIN *= 1e16

    DYFIN = ((Magboltz.YYSPL[JPRINT] / Magboltz.TSSUM[JPRINT]) - (
            Magboltz.YSPL[JPRINT] / Magboltz.TSSUM[JPRINT]) ** 2) * WSSST[JPRINT] / (
                    (JPRINT + 1) * 2 * Magboltz.ZSTEP)
    DYFIN *= 1e16

    DLFIN = ((Magboltz.TTMSPL[JPRINT] / Magboltz.TSSUM[JPRINT]) - (
            Magboltz.TMSPL[JPRINT] / Magboltz.TSSUM[JPRINT]) ** 2) * WSSST[JPRINT] ** 3 / (
                    (JPRINT + 1) * 2 * Magboltz.ZSTEP)
    DLFIN *= 1e16

    ALNGTH = Magboltz.ZSTEP * float(JPRINT + 1)
    ALFIN = np.log(float(Magboltz.NESST[JPRINT]) / float(Magboltz.IPRIM)) / ALNGTH
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

    if Magboltz.NESST[0] > Magboltz.NESST[4]:
        Magboltz.VDOUT = VDSST[1]
        Magboltz.VDERR = 100 * abs((WSSST[1] - WSSST[2]) / (2 * WSSST[1]))
        Magboltz.WSOUT = WSSST[1]
        Magboltz.DLOUT = DLSST[1]
        Magboltz.DLERR = 100 * abs((DLSST[1] - DLSST[2]) / (2 * DLSST[1]))
        Magboltz.DTOUT = DRSST[1]
        Magboltz.DTERR = 100 * abs((DRSST[1] - DRSST[2]) / (2 * DRSST[1]))
        if ATTOINT == -1:
            Magboltz.ALPHSST = 0.0
            Magboltz.ALPHERR = 0.0
            ANST2 = float(Magboltz.NESST[1])
            ANST3 = float(Magboltz.NESST[2])
            ANST4 = ANST3 - math.sqrt(ANST3)
            ANST5 = np.log(ANST2 / ANST3)
            ANST6 = np.log(ANST2 / ANST4)
            ANST7 = ANST6 / ANST5
            ANST8 = ANST7 - 1.0
            Magboltz.ATTSST = -1 * (ALFN[1] + ALFNJ[1] + ALFNE[1]) / 3
            Magboltz.ATTERR = 100 * math.sqrt(ANST8 ** 2 + Magboltz.ATTATER ** 2)
        else:
            ANST2 = float(Magboltz.NESST[1])
            ANST3 = float(Magboltz.NESST[2])
            ANST4 = ANST3 - math.sqrt(ANST3)
            ANST5 = np.log(ANST2 / ANST3)
            ANST6 = np.log(ANST2 / ANST4)
            ANST7 = ANST6 / ANST5
            ANST8 = ANST7 - 1.0
            ATMP = (ALFN[1] + ALFNJ[1] + ALFNE[1]) / 3
            Magboltz.ALPHSST = ATMP / (1 - ATTOINT)
            Magboltz.ALPHERR = 100 * math.sqrt(ANST8 ** 2 + Magboltz.ATTIOER ** 2)
            Magboltz.ATTSST = ATTOINT * ATMP / (1 - ATTOINT)
            Magboltz.ATTERR = 100 * math.sqrt(ANST8 ** 2 + Magboltz.ATTATER ** 2)
    else:
        Magboltz.VDOUT = VDSST[7]
        Magboltz.VDERR = 100 * abs((VDSST[7] - VDSST[6]) / VDSST[7])
        Magboltz.WSOUT = WSSST[7]
        Magboltz.WSERR = 100 * abs((WSSST[7] - WSSST[6]) / WSSST[7])
        Magboltz.DLOUT = DLFIN
        Magboltz.DLERR = 100 * abs((Magboltz.DLOUT - DLSST[7]) / Magboltz.DTOUT)
        Magboltz.DTOUT = (DXFIN + DYFIN) / 2
        Magboltz.DTERR = 100 * abs((Magboltz.DTOUT - DRSST[7]) / Magboltz.DTOUT)
        ATMP = (ALFN[7] + ALFNJ[7] + ALFNE[7]) / 3
        ATMP2 = (ALFN[6] + ALFNJ[6] + ALFNE[6]) / 3
        ATER = abs((ATMP - ATMP2) / ATMP)
        Magboltz.ALPHSST = ATMP / (1 - ATTOINT)
        Magboltz.ALPHERR = 100 * math.sqrt(ATER ** 2 + Magboltz.ATTIOER ** 2)
        Magboltz.ATTSST = ATTOINT * ATMP / (1 - ATTOINT)
        if (Magboltz.ATTOION != 0):
            Magboltz.ATTERR = 100 * math.sqrt(ATER ** 2 + Magboltz.ATTATER ** 2)
        else:
            Magboltz.ATTERR = 0.0
    return Magboltz
