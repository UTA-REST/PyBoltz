import numpy as np
import math
from FRIEDLANDT import FRIEDLANDT
from PTH import PTH
from MONTEFTHT import MONTEFTHT
from TOFH import TOFH


def ALPCLCCT(Magboltz):
    IMAX = Magboltz.NMAX / 10000000

    if IMAX < 5:
        IMAX = 5
    Magboltz.NMAX = IMAX * 10000000
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
        Magboltz.VDST = math.sqrt(Magboltz.WZ ** 2 + Magboltz.WY ** 2+ Magboltz.WX**2) * 1e-5
        Magboltz.FAKEI = ALPHAD * math.sqrt(Magboltz.WZ ** 2 + Magboltz.WY ** 2+ Magboltz.WX**2) * 1e-12
        Magboltz.ALPHAST = 0.85 * abs(ALPHAD + ANET)
        Magboltz.TSTEP = np.log(3) / (Magboltz.ALPHAST * Magboltz.VDST * 1e5)
        if Magboltz.TSTEP > TCUTH:
            Magboltz.TSTEP = TCUTH
        if Magboltz.TSTEP < TCUTL:
            Magboltz.TSTEP = TCUTL
        for J in range(Magboltz.NGAS):
            Magboltz.TCFMAX[J] += abs(Magboltz.FAKEI) / Magboltz.NGAS


        # CONVERT TO PICOSECONDS
        Magboltz.TSTEP *= 1e12
        Magboltz.TFINAL = 7 * Magboltz.TSTEP
        Magboltz.ITFINAL = 7
        JPRT = 0
        Magboltz = MONTEFTHT(Magboltz, JPRT)
        Magboltz = PTH(Magboltz, JPRT)
        Magboltz = TOFH(Magboltz, JPRT)
        Magboltz.TOFWR = math.sqrt(Magboltz.TOFWRZ ** 2 + Magboltz.TOFWRY ** 2 + Magboltz.TOFWRX ** 2)
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
        Magboltz.WX = Magboltz.TOFWRX * 1e5
    VTTT = math.sqrt(Magboltz.WZ ** 2 + Magboltz.WY ** 2 + Magboltz.WX ** 2)
    Magboltz.VDST = VTTT * 1e-5
    Magboltz.FAKEI = ALPHAD * VTTT * 1e-12
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
    Magboltz = MONTEFTHT(Magboltz, JPRT)
    Magboltz = FRIEDLANDT(Magboltz)
    Magboltz = PTH(Magboltz, JPRT)
    Magboltz = TOFH(Magboltz, JPRT)

    Magboltz.TOFWR = math.sqrt(Magboltz.TOFWRZ ** 2 + Magboltz.TOFWRY ** 2 + Magboltz.TOFWRX ** 2)
    WRN = Magboltz.TOFWR * 1e5
    FC1 = WRN / (2 * Magboltz.TOFDZZ)
    FC2 = ((Magboltz.RALPHA - Magboltz.RATTOF) * 1e12) / Magboltz.TOFDZZ
    ALPZZ = FC1 - math.sqrt(FC1 ** 2 - FC2)

    Magboltz.ALPHA = Magboltz.RALPHA / Magboltz.TOFWR * 1e7
    Magboltz.ALPER = Magboltz.RALPER * Magboltz.ALPHA / 100
    Magboltz.ATT = Magboltz.RATTOF / Magboltz.TOFWR * 1e7
    Magboltz.ATTER = Magboltz.RATTOFER * Magboltz.ATT / 100
    return Magboltz
