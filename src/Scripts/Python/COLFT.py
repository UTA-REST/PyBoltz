import numpy as np
import math


def COLFT():
    global Magboltz
    NINEL=0
    NELA = 0
    NATT = 0
    NION = 0
    for J in range(Magboltz.NGAS):
        NINEL+=Magboltz.ICOLL[J][3]+Magboltz.ICOLL[J][4]
        NELA+=Magboltz.ICOLL[J][0]
        NATT+=Magboltz.ICOLL[J][2]
        NION+=Magboltz.ICOLL[J][1]
    NTOTAL=NELA+NATT+NION+NINEL
    if Magboltz.TTOTS == 0:
        NREAL = NTOTAL
        Magboltz.TTOTS = Magboltz.ST
    else:
        NREAL = NTOTAL

    DUM = np.zeros(6)
    DUM[5] = NTOTAL
    DUM[0] = float(NREAL) / Magboltz.TTOTS
    DUM[4] = float(NINEL) / Magboltz.TTOTS
    DUM[1] = float(NELA) / Magboltz.TTOTS
    DUM[2] = float(NION) / Magboltz.TTOTS
    DUM[3] = float(NATT) / Magboltz.TTOTS
    return DUM