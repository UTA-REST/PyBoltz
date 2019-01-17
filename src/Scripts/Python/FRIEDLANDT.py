import numpy as np
import math
from GERJAN import GERJAN
from SORTT import SORTT


def FRIEDLANDT():
    global Magboltz
    FR = np.zeros(4000)
    ALFBAR = 0.0
    ATTBAR = 0.0
    EBAR = 0.0
    FSUM = 0.0
    for I in range(4000):
        TCFSUM = 0.0
        for KGAS in range(Magboltz.NGAS):
            TCFSUM += Magboltz.TCF[KGAS][I]
        FR[I] = Magboltz.SPEC[I] / TCFSUM
        EBAR += Magboltz.E[I] * Magboltz.SPEC[I] / TCFSUM
        ALFBAR += Magboltz.FCION[I] * Magboltz.SPEC[I] / TCFSUM
        ATTBAR += Magboltz.FCATT[I] * Magboltz.SPEC[I] / TCFSUM
        FSUM+=FR[I]
    for I in range(4000):
        FR[I]=FR[I]/FSUM
    EBAR = EBAR/Magboltz.TTOTS
    ALFBAR/=Magboltz.TTOTS
    ATTBAR/=Magboltz.TTOTS
    # ESTIMATION USING FRIEDLAND
    return  Magboltz