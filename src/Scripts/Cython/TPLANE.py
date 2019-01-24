import numpy as np
import math
from GERJAN import GERJAN


def TPLANE(Magboltz,T, E1, DCX1, DCY1, DCZ1, AP, BP, IPLANE):
    
    TIMESP = IPLANE * Magboltz.TSTEP
    TIMLFT = TIMESP - Magboltz.ST
    T2LFT = TIMLFT ** 2
    A = AP * TIMLFT
    B = BP * TIMLFT
    EPLANE = E1 + A + B
    CONST6 = math.sqrt(E1 / EPLANE)

    DCZ2 = DCZ1 * CONST6 + Magboltz.EFIELD * TIMLFT * Magboltz.CONST5 / math.sqrt(EPLANE)
    XPLANE = Magboltz.X + DCX1 * TIMLFT * math.sqrt(E1) * Magboltz.CONST3 * 0.01
    YPLANE = Magboltz.Y + DCY1 * TIMLFT * math.sqrt(E1) * Magboltz.CONST3 * 0.01
    ZPLANE = Magboltz.Z + DCZ1 * TIMLFT * math.sqrt(
        E1) * Magboltz.CONST3 * 0.01 + T2LFT * Magboltz.EFIELD * Magboltz.CONST2
    VZPLANE = DCZ2 * math.sqrt(EPLANE) * Magboltz.CONST3 * 0.01
    Magboltz.XTPL[IPLANE] += XPLANE
    Magboltz.YTPL[IPLANE] += YPLANE
    Magboltz.ZTPL[IPLANE] += ZPLANE
    Magboltz.XXTPL[IPLANE] += XPLANE ** 2
    Magboltz.YYTPL[IPLANE] += YPLANE ** 2
    Magboltz.ZZTPL[IPLANE] += ZPLANE ** 2
    Magboltz.ETPL[IPLANE] += EPLANE
    Magboltz.TTPL[IPLANE] += Magboltz.ST + TIMLFT
    Magboltz.VZTPL[IPLANE] += VZPLANE
    Magboltz.NETPL[IPLANE] += 1

    