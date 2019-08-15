import numpy as np
import math
from COLF import COLF
from COLFT import COLFT

def TOF(Magboltz,JPRT):
    
    DUM = np.zeros(6)
    if Magboltz.ITHRM == 1:
        DUM = COLFT()
    if Magboltz.ITHRM == 0:
        DUM = COLF()
    DLTF = np.zeros(8)
    DXTF = np.zeros(8)
    DYTF = np.zeros(8)
    WR = np.zeros(8)
    ANTPL = np.zeros(8)
    FRION = DUM[2]
    FRATT = DUM[3]
    ATTOINT = FRATT / FRION
    CORERR = abs((Magboltz.FAKEI + FRION - FRATT) / (FRION - FRATT))
    ANTPL[0] = float(Magboltz.NETPL[0])
    WR[0] = Magboltz.ZTPL[0] / (ANTPL[0] * Magboltz.TSTEP)
    DLTF[0] = ((Magboltz.ZZTPL[0] / ANTPL[0]) - (Magboltz.ZTPL[0] / ANTPL[0]) ** 2) / (2 * Magboltz.TSTEP)
    DXTF[0] = ((Magboltz.XXTPL[0] / ANTPL[0]) - (Magboltz.XTPL[0] / ANTPL[0]) ** 2) / (2 * Magboltz.TSTEP)
    DYTF[0] = ((Magboltz.YYTPL[0] / ANTPL[0]) - (Magboltz.YTPL[0] / ANTPL[0]) ** 2) / (2 * Magboltz.TSTEP)

    for I in range(1, Magboltz.ITFINAL):
        ANTPL[I] = float(Magboltz.NETPL[I])
        WR[I] = ((Magboltz.ZTPL[I] / ANTPL[I]) - (Magboltz.ZTPL[I - 1] / ANTPL[I - 1])) / Magboltz.TSTEP
        DLTF[I] = ((Magboltz.ZZTPL[I] / ANTPL[I]) - (Magboltz.ZTPL[I] / ANTPL[I]) ** 2 - (
                Magboltz.ZZTPL[I - 1] / ANTPL[I - 1]) + (Magboltz.ZTPL[I - 1] / ANTPL[I - 1]) ** 2) / (
                          2 * Magboltz.TSTEP)
        DXTF[I] = ((Magboltz.XXTPL[I] / ANTPL[I]) - (Magboltz.XTPL[I] / ANTPL[I]) ** 2 - (
                Magboltz.XXTPL[I - 1] / ANTPL[I - 1]) + (Magboltz.XTPL[I - 1] / ANTPL[I - 1]) ** 2) / (
                          2 * Magboltz.TSTEP)
        DYTF[I] = ((Magboltz.YYTPL[I] / ANTPL[I]) - (Magboltz.YTPL[I] / ANTPL[I]) ** 2 - (
                Magboltz.YYTPL[I - 1] / ANTPL[I - 1]) + (Magboltz.YTPL[I - 1] / ANTPL[I - 1]) ** 2) / (
                          2 * Magboltz.TSTEP)
    for I in range(Magboltz.ITFINAL):
        WR[I] *= 1e9
        DLTF[I] *= 1e16
        DXTF[I] *= 1e16
        DYTF[I] *= 1e16

    # TODO: PRINT TIME OF FLIGHT RSULTS

    if Magboltz.NETPL[0] > Magboltz.NETPL[Magboltz.ITFINAL]:
        Magboltz.TOFENE = Magboltz.EPT[1]
        Magboltz.TOFENER = 100 * abs((Magboltz.EPT[1] - Magboltz.EPT[2]) / (2 * Magboltz.EPT[1]))
        Magboltz.TOFWV = Magboltz.VZPT[1]
        Magboltz.TOFWVER = 100 * abs((Magboltz.VZPT[1] - Magboltz.VZPT[2]) / (2 * Magboltz.VZPT[1]))
        Magboltz.TOFDL = DLTF[1]
        Magboltz.TOFDLER = 100 * abs((DLTF[1] - DLTF[2]) / (2 * DLTF[1]))
        TDT2 = (DXTF[1] + DYTF[1]) / 2
        TDT3 = (DXTF[2] + DYTF[2]) / 2
        Magboltz.TOFDT = TDT2
        Magboltz.TOFDTER = 100 * abs((TDT2 - TDT3) / (2 * TDT2))
        Magboltz.TOFWR = WR[1]
        Magboltz.TOFWRER = 100 * abs((WR[1] - WR[2]) / (2 * WR[1]))
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
        Magboltz.TOFWV = Magboltz.VZPT[I1]
        Magboltz.TOFWVER = 100 * abs((Magboltz.VZPT[I1] - Magboltz.VZPT[I2]) / Magboltz.VZPT[I1])
        Magboltz.TOFDL = DLTF[I1]
        Magboltz.TOFDLER = 100 * abs((DLTF[I1] - DLTF[I2]) / DLTF[I1])
        TDT1 = (DXTF[I1] + DYTF[I1]) / 2
        TDT2 = (DXTF[I2] + DYTF[I2]) / 2
        Magboltz.TOFDT = TDT1
        Magboltz.TOFDTER = 100 * abs((TDT1 - TDT2) / TDT1)
        Magboltz.TOFWR = WR[I1]
        Magboltz.TOFWRER = 100 * abs((WR[I1] - WR[I2]) / WR[I1])
        ATER = abs((Magboltz.RI[I1] - Magboltz.RI[I2]) / Magboltz.RI[I1])
        Magboltz.RALPHA = Magboltz.RI[I1] / (1 - ATTOINT)
        Magboltz.RALPER = 100 * math.sqrt(ATER ** 2 + Magboltz.AIOERT ** 2)
        Magboltz.RATTOF=ATTOINT+Magboltz.RI[I1]/(1-ATTOINT)
        if ATTOINT!=0.0:
            Magboltz.RATTOFER=100*math.sqrt(ATER**2+Magboltz.ATTERT**2)
        else:
            Magboltz.RATTOFER=0.0


