import math

import numpy as np
from SORT import SORT


def ELIMITC(Magboltz):

    ISAMP = 10
    I = 0
    SMALL = 1.0e-20
    RTHETA = Magboltz.BTHETA * math.acos(-1) / 180
    EFZ100 = Magboltz.EFIELD * 100 * math.sin(RTHETA)
    EFX100 = Magboltz.EFIELD * 100 * math.cos(RTHETA)
    F1 = Magboltz.EFIELD * Magboltz.CONST2 * math.cos(Magboltz.RTHETA)
    EOVBR = Magboltz.EOVB * math.sin(RTHETA)
    E1 = Magboltz.ESTART
    INTEM = 8
    TDASH = 0.0
    CONST9 = Magboltz.CONST3 * 0.01

    TEMP = np.zeros(4000)
    for J in range(4000):
        TEMP[J] = Magboltz.TCFN[J] + Magboltz.TCF[J]

    DCZ1 = math.cos(Magboltz.THETA)
    DCX1 = math.sin(Magboltz.THETA) * math.cos(Magboltz.PHI)
    DCY1 = math.sin(Magboltz.THETA) * math.sin(Magboltz.PHI)
    VTOT = CONST9 * math.sqrt(E1)
    CX1 = DCX1 * VTOT
    CY1 = DCY1 * VTOT
    CZ1 = DCZ1 * VTOT

    F4 = 2 * math.acos(-1)
    DELTAE = Magboltz.EFINAL / float(INTEM)
    J2M = Magboltz.NMAX / ISAMP

    for J1 in range(int(J2M)):
        R5 = 1
        TEST1 = 0
        while R5 > TEST1:
            R1 = Magboltz.RAND48.drand()
            I = int(E1 / DELTAE) + 1
            I = min(I, INTEM) - 1
            TLIM = Magboltz.TCFMAX[I]
            T = -1 * np.log(R1) / TLIM + TDASH
            TDASH = T
            WBT = Magboltz.WB * T
            COSWT = math.cos(WBT)
            SINWT = math.sin(WBT)
            DZ = (CZ1 * SINWT + (EOVBR - CY1) * (1 - COSWT)) / Magboltz.WB
            DX = CX1 * T + F1 * T * T
            E = E1 + DZ * EFZ100 + DX * EFX100
            IE = int(E / Magboltz.ESTEP)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += np.log(R1) / TLIM
                Magboltz.TCFMAX[I] *= 1.05
                continue
            R5 = Magboltz.RAND48.drand()
            TEST1 = Magboltz.TCF[IE] / TLIM
        if IE == 3999:
            Magboltz.IELOW = 1

        TDASH = 0.0
        CX2 = CX1 + 2 * F1 * T
        CY2 = (CY1 - EOVBR) * COSWT + CZ1 * SINWT + EOVBR
        CZ2 = CZ1 * COSWT - (CY1 - EOVBR) * SINWT
        VTOT = math.sqrt(CX2 ** 2 + CY2 ** 2 + CZ2 ** 2)
        DCX2 = CX2 / VTOT
        DCY2 = CY2 / VTOT
        DCZ2 = CZ2 / VTOT
        R2 = Magboltz.RAND48.drand()
        I = 0
        I = SORT(I, R2, IE, Magboltz)
        while Magboltz.CF[IE][I] < R2:
            I = I + 1
        S1 = Magboltz.RGAS[I]
        EI = Magboltz.EIN[I]
        if Magboltz.IPN[I] > 0:
            R9 = Magboltz.RAND48.drand()
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI
        IPT = Magboltz.IARRY[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = Magboltz.RAND48.drand()

        if Magboltz.INDEX[I] == 1:
            R31 = Magboltz.RAND48.drand()
            F3 = 1.0 - R3 * Magboltz.ANGCT[IE][I]
            if R31 > Magboltz.PSCT[IE][I]:
                F3 = -1 * F3
            elif Magboltz.INDEX[I] == 2:
                EPSI = Magboltz.PSCT[IE][I]
                F3 = 1 - (2 * R3 * (1 - EPSI)) / (1 + EPSI * (1 - 2 * R3))
            else:
                F3 = 1 - 2 * R3
        THETA0 = math.acos(F3)
        R4 = Magboltz.RAND48.drand()
        PHI0 = F4 * R4
        F8 = math.sin(PHI0)
        F9 = math.cos(PHI0)
        ARG1 = 1 - S1 * EI / E
        ARG1 = max(ARG1, SMALL)

        D = 1 - F3 * math.sqrt(ARG1)
        E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = math.sqrt((E / E1) * ARG1) / S1
        Q = min(Q, 1)
        Magboltz.THETA = math.asin(Q * math.sin(THETA0))

        F6 = math.cos(Magboltz.THETA)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = F3 ** 2

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6
        F5 = math.sin(Magboltz.THETA)
        DCZ2 = min(DCZ2, 1)
        VTOT = CONST9 * math.sqrt(E1)
        ARGZ = math.sqrt(DCX2 * DCX2 + DCY2 * DCY2)
        if ARGZ == 0:
            DCZ1 = F6
            DCX1 = F9 * F5
            DCY1 = F8 * F5
        else:
            DCZ1 = DCZ2 * F6 + ARGZ * F5 * F8
            DCY1 = DCY2 * F6 + (F5 / ARGZ) * (DCX2 * F9 - DCY2 * DCZ2 * F8)
            DCX1 = DCX2 * F6 - (F5 / ARGZ) * (DCY2 * F9 + DCX2 * DCZ2 * F8)
        CX1 = DCX1 * VTOT
        CY1 = DCY1 * VTOT
        CZ1 = DCZ1 * VTOT
    Magboltz.IELOW = 0

