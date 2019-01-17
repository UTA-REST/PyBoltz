import numpy as np
import math
from GERJAN import GERJAN


def TPLANEGT(T, E1, CX1, CY1, CZ1, IPLANE):
    global Magboltz
    TIMESP = IPLANE * Magboltz.TSTEP
    TIMLFT = TIMESP - Magboltz.ST
    T2LFT = TIMLFT ** 2
    WBT = Magboltz.WB * TIMLFT
    COSWT = math.cos(WBT)
    SINWT = math.sin(WBT)
    CX2 = CX1
    CY2 = (CY1 - Magboltz.EOVB) * COSWT + CZ1 * SINWT + Magboltz.EOVB
    CZ2 = CZ1 * COSWT - (CY1 - Magboltz.EOVB) * SINWT
    VTOT = math.sqrt(CX2 ** 2 + CY2 ** 2 + CZ2 ** 2)
    DCZ2 = CZ2 / VTOT
    DCY2 = CY2 / VTOT
    XPLANE = Magboltz.X + CX1 * TIMLFT
    YPLANE = Magboltz.Y + Magboltz.EOVB * TIMLFT + ((CY1 - Magboltz.EOVB) * SINWT + CZ1 * (1 - COSWT)) / Magboltz.WB
    DZ = (CZ1 * SINWT + (Magboltz.EOVB - CY1) * (1 - COSWT)) / Magboltz.WB
    ZPLANE = Magboltz.Z + DZ
    EPLANE = E1 + DZ * Magboltz.EFIELD * 100
    VZPLANE = DCZ2 * math.sqrt(EPLANE) * Magboltz.CONST3 * 0.01
    VYPLANE = DCY2 * math.sqrt(EPLANE) * Magboltz.CONST3 * 0.01
    Magboltz.XTPL[IPLANE] += XPLANE
    Magboltz.YTPL[IPLANE] += YPLANE
    Magboltz.ZTPL[IPLANE] += ZPLANE
    Magboltz.XXTPL[IPLANE] += XPLANE ** 2
    Magboltz.YYTPL[IPLANE] += YPLANE ** 2
    Magboltz.ZZTPL[IPLANE] += ZPLANE ** 2
    Magboltz.YZTPL[IPLANE] += YPLANE * ZPLANE
    Magboltz.ETPL[IPLANE] += EPLANE
    Magboltz.TTPL[IPLANE] += Magboltz.ST + TIMLFT
    Magboltz.VZTPL[IPLANE] += VZPLANE
    Magboltz.VYTPL[IPLANE] += VYPLANE
    Magboltz.NETPL[IPLANE] += 1

