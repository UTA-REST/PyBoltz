import math


def TPLANEH(Magboltz, T, E1, CX1, CY1, CZ1, IPLANE, EOVBR, F1, RCS, RSN):
    TIMESP = IPLANE * Magboltz.TSTEP
    TIMLFT = TIMESP - Magboltz.ST
    T2LFT = TIMLFT ** 2
    WBT = Magboltz.WB * TIMLFT
    COSWT = math.cos(WBT)
    SINWT = math.sin(WBT)

    RTHETA = Magboltz.BTHETA * Magboltz.API / 180
    EFZ100 = Magboltz.EFIELD * 100 * math.sin(RTHETA)
    EFX100 = Magboltz.EFIELD * 100 * math.cos(RTHETA)
    CX2 = CX1 + 2 * F1 * TIMLFT
    CY2 = (CY1 - EOVBR) * COSWT + CZ1 * SINWT + EOVBR
    CZ2 = CZ1 * COSWT - (CY1 - EOVBR) * SINWT
    VTOT = math.sqrt(CX2 ** 2 + CY2 ** 2 + CZ2 ** 2)
    DCZ2 = CZ2 / VTOT
    DCY2 = CY2 / VTOT
    DCX2 = CX2 / VTOT
    DX = CX1 * TIMLFT + F1 * TIMLFT * TIMLFT
    XPLANE = Magboltz.X + DX
    YPLANE = Magboltz.Y + EOVBR * TIMLFT + ((CY1 - EOVBR) * SINWT + CZ1 * (1 - COSWT)) / Magboltz.WB
    DZ = (CZ1 * SINWT + (EOVBR - CY1) * (1 - COSWT)) / Magboltz.WB
    ZPLANE = Magboltz.Z + DZ
    ZPLANER = ZPLANE * RCS - XPLANE * RSN
    YPLANER = YPLANE
    XPLANER = ZPLANE * RSN + XPLANE * RCS
    EPLANE = E1 + DZ * EFZ100 + DX * EFX100
    VZPLANE = DCZ2 * math.sqrt(EPLANE) * Magboltz.CONST3 * 0.01
    VYPLANE = DCY2 * math.sqrt(EPLANE) * Magboltz.CONST3 * 0.01
    VXPLANE = DCX2 * math.sqrt(EPLANE) * Magboltz.CONST3 * 0.01
    Magboltz.XTPL[IPLANE] += XPLANER
    Magboltz.YTPL[IPLANE] += YPLANER
    Magboltz.ZTPL[IPLANE] += ZPLANER
    Magboltz.XXTPL[IPLANE] += XPLANER ** 2
    Magboltz.YYTPL[IPLANE] += YPLANER ** 2
    Magboltz.ZZTPL[IPLANE] += ZPLANER ** 2
    Magboltz.YZTPL[IPLANE] += YPLANER * ZPLANER
    Magboltz.XZTPL[IPLANE] += XPLANER * ZPLANER
    Magboltz.XYTPL[IPLANE] += XPLANER * YPLANER
    Magboltz.ETPL[IPLANE] += EPLANE
    Magboltz.TTPL[IPLANE] += Magboltz.ST + TIMLFT
    VZPLNER = VZPLANE * RCS - VXPLANE * RSN
    VYPLNER = VYPLANE
    VXPLNER = VZPLANE * RSN + VXPLANE * RCS
    Magboltz.VZTPL[IPLANE] += VZPLNER
    Magboltz.VYTPL[IPLANE] += VYPLNER
    Magboltz.VXTPL[IPLANE] += VXPLNER
    Magboltz.NETPL[IPLANE] += 1
    return Magboltz
