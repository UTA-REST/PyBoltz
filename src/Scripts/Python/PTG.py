import numpy as np
import math


def PTG(JPRT):
    global Magboltz
    ANTPL = np.zeros(8)
    ANTPL[0] = float(Magboltz.NETPL[0])
    Magboltz.RT[0] = (np.log(ANTPL[0]) - np.log(float(Magboltz.IPRIM))) / Magboltz.TSTEP
    Magboltz.RT[0] -= Magboltz.FAKEI
    Magboltz.EPT[0] = Magboltz.ETPL[0] / ANTPL[0]
    Magboltz.TTEST[0] = Magboltz.TTPL[0] / ANTPL[0]
    Magboltz.VZPT[0] = 1e9 * Magboltz.VZTPL[0] / ANTPL[0]
    Magboltz.VYPT[0] = 1e9 * Magboltz.VYTPL[0] / ANTPL[0]
    for I in range(1, Magboltz.ITFINAL):
        if Magboltz.NETPL[I] == 0:
            Magboltz.ITFINAL = I - 1
            break
        ANTPL[I] = float(Magboltz.NETPL[I])
        Magboltz.RI[I] = (np.log(ANTPL[I]) - np.log(ANTPL[I - 1])) / Magboltz.TSTEP
        Magboltz.RT[I] -= Magboltz.FAKEI
        Magboltz.EPT[I] = Magboltz.ETPL[I] / ANTPL[I]
        Magboltz.TTEST[I] = Magboltz.TTPL[I] / ANTPL[I]
        Magboltz.VYPT[I] = 1e9 * Magboltz.VYTPL[I] / ANTPL[I]
        Magboltz.VZPT[I] = 1e9 * Magboltz.VZTPL[I] / ANTPL[I]

