import numpy as np
import math
from FRIEDLAND import FRIEDLAND
from MONTEFTG import MONTEFTG
from PTG import PTG
from TOFG import TOFG


def ALPCLCB(Magboltz):

    IMAX = Magboltz.NMAX / 10000000
    if IMAX < 5:
        IMAX = 5

    Magboltz.CORR = 760 * (273.15 + Magboltz.TEMPC) / (293.15 * Magboltz.TORR)
    Magboltz.ALPP = Magboltz.ALPHA / Magboltz.CORR
    Magboltz.ATTP = Magboltz.ATT / Magboltz.CORR
    ANETP = Magboltz.ALPP - Magboltz.ATTP
    ANET = Magboltz.ALPHA - Magboltz.ATT
    TCUTH = 1.2e-10 * Magboltz.CORR
    TCUTL = 1e-13 * Magboltz.CORR

    if ANETP > 30:
        ALPHAD = 0.0
        ALP1 = Magboltz.ALPHA
        ATT1 = Magboltz.ATT
    else:
        if abs(ANETP) < 100:
            ALPHAD = abs(Magboltz.ATT) * 0.8
        elif abs(ANETP) >= 100 and abs(ANETP) < 1000:
            ALPHAD = abs(ANET) * 0.65
        elif abs(ANETP) >= 1000 and abs(ANETP) < 10000:
            ALPHAD = abs(ANET) * 0.6
        elif abs(ANETP) >= 10000 and abs(ANETP) < 100000:
            ALPHAD = abs(ANET) * 0.5
        elif abs(ANETP) >= 100000 and abs(ANETP) < 2000000:
            ALPHAD = abs(ANET) * 0.4
        else:
            raise ValueError("ATTACHMENT TOO LARGE PROGRAM STOPPED")
        Magboltz.VDST = (Magboltz.WZ ** 2 + Magboltz.WY ** 2) * 1e-5
        Magboltz.FAKEI = ALPHAD * math.sqrt(Magboltz.WZ ** 2 + Magboltz.WY ** 2) * 1e-12
        Magboltz.ALPHAST = 0.85 * abs(ALPHAD + ANET)
        Magboltz.TSTEP = np.log(3) / (Magboltz.ALPHAST * Magboltz.VDST * 1e5)
        if Magboltz.TSTEP > TCUTH:
            Magboltz.TSTEP = TCUTH
        if Magboltz.TSTEP < TCUTL:
            Magboltz.TSTEP = TCUTL
        for J in range(8):
            Magboltz.TCFMAX[J] += abs(Magboltz.FAKEI)
            # CONVERT TO PICOSECONDS
        Magboltz.TSTEP *= 1e12
        Magboltz.TFINAL = 7 * Magboltz.TSTEP
        Magboltz.ITFINAL = 7
        JPRT = 0
        MONTEFTG(Magboltz,JPRT)
        PTG(Magboltz,JPRT)
        TOFG(Magboltz,JPRT)
        Magboltz.TOFWR = math.sqrt(Magboltz.TOFWRZ ** 2 + Magboltz.TOFWRY ** 2)
        ALP1 = Magboltz.RALPHA / Magboltz.TOFWR * 1e7
        ALP1ER = Magboltz.RALPER * ALP1 / 100
        ATT1 = Magboltz.RATTOF / Magboltz.TOFWR * 1e7
        ATT1ER = Magboltz.RATTOFER * ATT1 / 100
        for J in range(Magboltz.NGAS):
            Magboltz.TCFMAX[J] -= abs(Magboltz.FAKEI) / Magboltz.NGAS
        ALPHAD = abs(ATT1) * 1.2
        if (ALP1 - ATT1) > 30 * Magboltz.CORR:
            ALPHAD = abs(ATT1) * 0.4
        if abs(ALP1 - ATT1) < (ALP1 / 10) or abs(ALP1 - ATT1) < (ATT1 / 10):
            ALPHAD = abs(ATT1) * 0.3
        if (ALP1 - ATT1) > 100 * Magboltz.CORR:
            ALPHAD = 0.0
        Magboltz.WZ = Magboltz.TOFWRZ * 1e5
        Magboltz.WY = Magboltz.TOFWRY * 1e5
    VTOT = math.sqrt(Magboltz.WZ ** 2 + Magboltz.WY ** 2)
    Magboltz.VDST = VTOT * 1e-5
    Magboltz.FAKEI = ALPHAD * VTOT * 1e-12
    Magboltz.ALPHAST = 0.85 * abs(ALPHAD + ALP1 - ATT1)

    if ALP1 + ALPHAD > 10 * Magboltz.ALPHAST or ATT1 > 10 * Magboltz.ALPHAST:
        if ALP1 + ALPHAD > 100 * Magboltz.CORR:
            Magboltz.ALPHAST *= 15
        elif ALP1 + ALPHAD > 50 * Magboltz.CORR:
            Magboltz.ALPHAST *= 12
        else:
            Magboltz.ALPHAST *= 8
    Magboltz.TSTEP = np.log(3) / (Magboltz.ALPHAST * Magboltz.VDST * 1e5)
    if Magboltz.TSTEP > TCUTH and ALPHAD != 0.0:
        Magboltz.TSTEP = TCUTH
    for J in range(Magboltz.NGAS):
        Magboltz.TCFMAX[J] += abs(Magboltz.FAKEI) / Magboltz.NGAS
    ANET1 = ALP1 - ATT1

    Magboltz.TSTEP *= 1e12
    # TODO: check index -1
    Magboltz.TFINAL = 7 * Magboltz.TSTEP
    Magboltz.ITFINAL = 7
    JPRT = 1
    MONTEFTG(Magboltz,JPRT)
    FRIEDLAND(Magboltz)
    PTG(Magboltz,JPRT)
    TOFG(Magboltz,JPRT)

    Magboltz.TOFWR = math.sqrt(Magboltz.TOFWRZ ** 2 + Magboltz.TOFWRY ** 2)
    WRN = Magboltz.TOFWR * 1e5
    FC1 = WRN / (2 * Magboltz.TOFDZZ)
    FC2 = ((Magboltz.RALPHA - Magboltz.RATTOF) * 1e12) / Magboltz.TOFDZZ
    ALPZZ = FC1 - math.sqrt(FC1 ** 2 - FC2)

    Magboltz.ALPHA = Magboltz.RALPHA / Magboltz.TOFWR * 1e7
    Magboltz.ALPER = Magboltz.RALPER * Magboltz.ALPHA / 100
    Magboltz.ATT = Magboltz.RATTOF / Magboltz.TOFWR * 1e7
    Magboltz.ATTER = Magboltz.RATTOFER * Magboltz.ATT / 100
