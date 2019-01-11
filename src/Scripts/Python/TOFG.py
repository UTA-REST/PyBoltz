import numpy as np
import math
from COLF import COLF
from COLFT import COLFT


def TOFG(Magboltz, JPRT):
    DUM = np.zeros(6)
    if Magboltz.ITHRM == 1:
        DUM = COLFT(Magboltz)
    if Magboltz.ITHRM == 0:
        DUM = COLF(Magboltz)
    DZTF = np.zeros(8)
    DXTF = np.zeros(8)
    DYTF = np.zeros(8)
    WRZ = np.zeros(8)
    WRY = np.zeros(8)
    DYZTF = np.zeros(8)
    ANTPL = np.zeros(8)
    FRION = DUM[2]
    FRATT = DUM[3]
    ATTOINT = FRATT / FRION
    CORERR = abs((Magboltz.FAKEI + FRION - FRATT) / (FRION - FRATT))
    ANTPL[0] = float(Magboltz.NETPL[0])
    WRZ[0] = Magboltz.ZTPL[0] / (ANTPL[0] * Magboltz.TSTEP)
    DZTF[0] = ((Magboltz.ZZTPL[0] / ANTPL[0]) - (Magboltz.ZTPL[0] / ANTPL[0]) ** 2) / (2 * Magboltz.TSTEP)
    DXTF[0] = ((Magboltz.XXTPL[0] / ANTPL[0]) - (Magboltz.XTPL[0] / ANTPL[0]) ** 2) / (2 * Magboltz.TSTEP)
    DYTF[0] = ((Magboltz.YYTPL[0] / ANTPL[0]) - (Magboltz.YTPL[0] / ANTPL[0]) ** 2) / (2 * Magboltz.TSTEP)
    DYZTF[0] = ((Magboltz.YZTPL[0] / ANTPL[0]) - (Magboltz.YTPL[0] * Magboltz.ZTPL[0] / (ANTPL[0] ** 2))) / (
            2 * Magboltz.TSTEP)
    for I in range(1, Magboltz.ITFINAL):
        ANTPL[I] = float(Magboltz.NETPL[I])
        WRZ[I] = ((Magboltz.ZTPL[I] / ANTPL[I]) - (Magboltz.ZTPL[I - 1] / ANTPL[I - 1])) / Magboltz.TSTEP
        WRY[I] = ((Magboltz.YTPL[I] / ANTPL[I]) - (Magboltz.YTPL[I - 1] / ANTPL[I - 1])) / Magboltz.TSTEP
        DZTF[I] = ((Magboltz.ZZTPL[I] / ANTPL[I]) - (Magboltz.ZTPL[I] / ANTPL[I]) ** 2 - (
                Magboltz.ZZTPL[I - 1] / ANTPL[I - 1]) + (Magboltz.ZTPL[I - 1] / ANTPL[I - 1]) ** 2) / (
                          2 * Magboltz.TSTEP)
        DXTF[I] = ((Magboltz.XXTPL[I] / ANTPL[I]) - (Magboltz.XTPL[I] / ANTPL[I]) ** 2 - (
                Magboltz.XXTPL[I - 1] / ANTPL[I - 1]) + (Magboltz.XTPL[I - 1] / ANTPL[I - 1]) ** 2) / (
                          2 * Magboltz.TSTEP)
        DYTF[I] = ((Magboltz.YYTPL[I] / ANTPL[I]) - (Magboltz.YTPL[I] / ANTPL[I]) ** 2 - (
                Magboltz.YYTPL[I - 1] / ANTPL[I - 1]) + (Magboltz.YTPL[I - 1] / ANTPL[I - 1]) ** 2) / (
                          2 * Magboltz.TSTEP)
        DYZTF[I] = ((Magboltz.YZTPL[I] / ANTPL[I]) - (Magboltz.YTPL[I] * Magboltz.ZTPL[I] / (ANTPL[I] ** 2)) - (
                Magboltz.YZTPL[I - 1] / ANTPL[I - 1]) + (
                            Magboltz.YTPL[I - 1] * Magboltz.ZTPL[I - 1] / (ANTPL[I - 1] ** 2))) / (
                           2 * Magboltz.TSTEP)
    for I in range(Magboltz.ITFINAL):
        WRZ[I] *= 1e9
        WRY[I] *= 1e9
        DZTF[I] *= 1e16
        DXTF[I] *= 1e16
        DYTF[I] *= 1e16
        DYZTF[I] *= 1e16

    if Magboltz.NETPL[0] > Magboltz.NETPL[Magboltz.ITFINAL]:
        Magboltz.TOFENE = Magboltz.EPT[1]
        Magboltz.TOFENER = 100 * abs((Magboltz.EPT[1] - Magboltz.EPT[2]) / (2 * Magboltz.EPT[1]))
        Magboltz.TOFWVZ = Magboltz.VZPT[1]
        Magboltz.TOFWVZER = 100 * abs((Magboltz.VZPT[1] - Magboltz.VZPT[2]) / (2 * Magboltz.VZPT[1]))
        Magboltz.TOFWVY = Magboltz.VYPT[1]
        Magboltz.TOFWVYER = 100 * abs((Magboltz.VYPT[1] - Magboltz.VYPT[2]) / (2 * Magboltz.VYPT[1]))
        Magboltz.TOFDZZ = DZTF[1]
        Magboltz.TOFDZZER = 100 * abs((DZTF[1] - DZTF[2]) / (2 * DZTF[1]))
        Magboltz.TOFDYY = DYTF[1]
        Magboltz.TOFDYYER = 100 * abs((DYTF[1] - DYTF[2]) / (2 * DYTF[1]))
        Magboltz.TOFDXX = DXTF[1]
        Magboltz.TOFDXXER = 100 * abs((DXTF[1] - DXTF[2]) / (2 * DXTF[1]))
        Magboltz.TOFDYZ = DYZTF[1]
        Magboltz.TOFDYZER = 100 * abs((DYZTF[1] - DYZTF[2]) / (2 * DYZTF[1]))
        Magboltz.TOFWRZ = WRZ[1]
        Magboltz.TOFWRY = WRY[1]
        Magboltz.TOFWRZER = 100 * abs((WRZ[1] - WRZ[2])) / (2 * WRZ[1])
        Magboltz.TOFWRYER = 100 * abs((WRY[1] - WRY[2])) / (2 * WRY[1])
        ANST2 = float(Magboltz.NETPL[1])
        ANST3 = float(Magboltz.NETPL[2])
        ANST4 = ANST3 - math.sqrt(ANST3)
        ANST5 = np.log(ANST2 / ANST3)
        ANST6 = np.log(ANST2 / ANST4)
        ANST7 = ANST6 / ANST5
        ANST8 = ANST7 - 1
        if ATTOINT == -1:
            Magboltz.RALPHA = 0.0
            Magboltz.RALPER = 0.0
            Magboltz.RATTOF = -1 * Magboltz.RI[1]
            Magboltz.RATTOFER = 100 * math.sqrt(ANST8 ** 2 + Magboltz.ATTERT ** 2)
        else:
            Magboltz.RALPHA = Magboltz.RI[1] / (1 - ATTOINT)
            Magboltz.RALPER = 100 * math.sqrt(ANST8 ** 2 + Magboltz.AIOERT ** 2)
            Magboltz.RATTOF = ATTOINT * Magboltz.RI[1] / (1 - ATTOINT)
            Magboltz.RATTOFER = 100 * math.sqrt(ANST8 ** 2 + Magboltz.ATTERT ** 2)
    else:
        I1 = Magboltz.ITFINAL
        I2 = Magboltz.ITFINAL - 1
        Magboltz.TOFENE = Magboltz.EPT[I1]
        Magboltz.TOFENER = 100 * abs((Magboltz.EPT[I1] - Magboltz.EPT[I2]) / Magboltz.EPT[I1])
        Magboltz.TOFWVZ = Magboltz.VZPT[I1]
        Magboltz.TOFWVZER = 100 * abs((Magboltz.VZPT[I1] - Magboltz.VZPT[I2]) / Magboltz.VZPT[I1])
        Magboltz.TOFWVY = Magboltz.VYPT[I1]
        Magboltz.TOFWVYER = 100 * abs((Magboltz.VYPT[I1] - Magboltz.VYPT[I2]) / Magboltz.VYPT[I1])
        Magboltz.TOFDZZ = DZTF[I1]
        Magboltz.TOFDZZER = 100 * abs((DZTF[I1] - DZTF[I2]) / DZTF[I1])
        Magboltz.TOFDYY = DYTF[I1]
        Magboltz.TOFDYYER = 100 * abs((DYTF[I1] - DYTF[I2]) / DYTF[I1])
        Magboltz.TOFDXX = DXTF[I1]
        Magboltz.TOFDXXER = 100 * abs((DXTF[I1] - DXTF[I2]) / DXTF[I1])
        Magboltz.TOFDYZ = DYZTF[I1]
        Magboltz.TOFDYZER = 100 * abs((DYZTF[I1] - DYZTF[I2]) / DYZTF[I1])
        Magboltz.TOFWRZ = WRZ[I1]
        Magboltz.TOFWRY = WRY[I1]
        Magboltz.TOFWRYER = 100 * abs((WRY[I1] - WRY[I2]) / WRY[I1])
        Magboltz.TOFWRZER = 100 * abs((WRZ[I1] - WRZ[I2]) / WRZ[I1])
        ATER = abs((Magboltz.RI[I1] - Magboltz.RI[I2]) / Magboltz.RI[I1])
        Magboltz.RALPHA = Magboltz.RI[I1] / (1 - ATTOINT)
        Magboltz.RALPER = 100 * math.sqrt(ATER ** 2 + Magboltz.AIOERT ** 2)
        Magboltz.RATTOF = ATTOINT + Magboltz.RI[I1] / (1 - ATTOINT)
        if ATTOINT != 0.0:
            Magboltz.RATTOFER = 100 * math.sqrt(ATER ** 2 + Magboltz.ATTERT ** 2)
        else:
            Magboltz.RATTOFER = 0.0

    return Magboltz
