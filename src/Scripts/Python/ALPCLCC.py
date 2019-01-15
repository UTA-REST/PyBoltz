import numpy as np
import math
from FRIEDLAND import FRIEDLAND
from PTH import PTH
from MONTEFTH import MONTEFTH
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
