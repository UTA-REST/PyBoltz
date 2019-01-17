import numpy as np
import math


def SPLANE(T, E1, DCX1, DCY1, DCZ1, AP, BP, TIMLFT, IZPLANE):
    global Magboltz
    if IZPLANE > 8:
        return
    T2LFT = TIMLFT ** 2
    A = AP * TIMLFT
    B = BP * T2LFT
    EPLANE = E1 + A + B
    CONST6 = math.sqrt(E1 / EPLANE)
    DCZ2 = DCZ1 * CONST6 + Magboltz.EFIELD * TIMLFT * Magboltz.CONST5 / math.sqrt(EPLANE)
    XPLANE = Magboltz.X + DCX1 * TIMLFT * math.sqrt(E1) * Magboltz.CONST3 * 0.01
    YPLANE = Magboltz.Y + DCY1 * TIMLFT * math.sqrt(E1) * Magboltz.CONST3 * 0.01
    ZPLANE = Magboltz.Z + DCZ1 * TIMLFT * math.sqrt(E1) * Magboltz.CONST3 * 0.01
    VZPLANE = DCZ2 * math.sqrt(EPLANE) * Magboltz.CONST3 * 0.01
    WGHT = abs(1 / VZPLANE)
    RPLANE = math.sqrt(XPLANE ** 2 + YPLANE ** 2)
    Magboltz.XSPL[IZPLANE - 1] += XPLANE * WGHT
    Magboltz.YSPL[IZPLANE - 1] += YPLANE * WGHT
    Magboltz.RSPL[IZPLANE - 1] += RPLANE * WGHT
    Magboltz.ZSPL[IZPLANE - 1] += ZPLANE * WGHT
    Magboltz.TMSPL[IZPLANE - 1] += (Magboltz.ST + TIMLFT) * WGHT
    Magboltz.TTMSPL[IZPLANE - 1] += (Magboltz.ST + TIMLFT) * (Magboltz.ST + TIMLFT) * WGHT
    Magboltz.XXSPL[IZPLANE - 1] += WGHT * XPLANE ** 2
    Magboltz.YYSPL[IZPLANE - 1] += WGHT * YPLANE ** 2
    Magboltz.ZZSPL[IZPLANE - 1] += WGHT * ZPLANE ** 2
    Magboltz.RRSPM[IZPLANE - 1] += WGHT * RPLANE ** 2
    Magboltz.ESPL[IZPLANE - 1] += EPLANE * WGHT
    Magboltz.TSPL[IZPLANE - 1] += WGHT / (Magboltz.ST + TIMLFT)
    Magboltz.VZSPL[IZPLANE - 1] += VZPLANE * WGHT
    Magboltz.TSSUM[IZPLANE - 1] += WGHT
    Magboltz.TSSUM2[IZPLANE - 1] += WGHT ** 2


