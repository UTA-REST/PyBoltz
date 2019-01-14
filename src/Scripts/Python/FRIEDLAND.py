import numpy as np
import math


def FRIEDLAND(Magboltz):
    FR = np.zeros(4000)
    ALFBAR = 0.0
    ATTBAR = 0.0
    EBAR = 0.0
    FSUM = 0.0
    for I in range(4000):
        FR[I] = Magboltz.SPEC[I] / Magboltz.TCF[I]
        EBAR += Magboltz.E[I] * Magboltz.SPEC[I] / Magboltz.TCF[I]
        ALFBAR += Magboltz.FCION[I] * Magboltz.SPEC[I] / Magboltz.TCF[I]
        ATTBAR += Magboltz.FCATT[I] * Magboltz.SPEC[I] / Magboltz.TCF[I]
        FSUM += FR[I]
    for I in range(4000):
        FR[I] = FR[I] / FSUM
    EBAR = EBAR / Magboltz.TTOTS
    ALFBAR /= Magboltz.TTOTS
    ATTBAR /= Magboltz.TTOTS
    # ESTIMATION USING FRIEDLAND
    return Magboltz
