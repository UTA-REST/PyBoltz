import numpy as np
import math
from GERJAN import GERJAN


def TPLANEAT(Magboltz,T, E1, CX1, CY1, DCZ1, AP, BP, IPLANE):
    
    TIMESP = IPLANE * Magboltz.TSTEP
    TIMLFT = TIMESP - Magboltz.ST
    T2LFT = TIMLFT ** 2
    A = AP * TIMLFT
    B = BP * T2LFT
    EPLANE = E1 + A + B
    CONST6 = math.sqrt(E1 / EPLANE)
    WBT = Magboltz.WB * TIMLFT
    SINWT = math.sin(WBT)
    COSWT = math.cos(WBT)
    DCZ2 = DCZ1 * CONST6 + Magboltz.EFIELD * TIMLFT * Magboltz.CONST5 / math.sqrt(EPLANE)
    XPLANE = Magboltz.X + (CX1 * SINWT-CY1*(1-COSWT))/Magboltz.WB
    YPLANE = Magboltz.Y + (CY1 * SINWT+CX1*(1-COSWT))/Magboltz.WB
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


