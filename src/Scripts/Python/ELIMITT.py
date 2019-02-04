import Magboltz
from GERJAN import GERJAN
import math
import random
import numpy as np
from SORTT import SORTT


def ELIMITT(Magboltz):
    print("ELIMITT")
    I = 0
    ISAMP = 10
    SMALL = 1.0e-20
    RDUM = Magboltz.RSTART
    E1 = Magboltz.ESTART
    N4000 = 4000
    TDASH = 0.0
    CONST9 = Magboltz.CONST3 * 0.01
    CONST10 = CONST9 * CONST9

    Magboltz.RNMX = GERJAN(Magboltz.RSTART, Magboltz.API)
    IMBPT = 0

    DCZ1 = math.cos(Magboltz.THETA)
    DCX1 = math.sin(Magboltz.THETA) * math.cos(Magboltz.PHI)
    DCY1 = math.sin(Magboltz.THETA) * math.sin(Magboltz.PHI)

    BP = (Magboltz.EFIELD ** 2) * Magboltz.CONST1
    F1 = Magboltz.EFIELD * Magboltz.CONST2
    F2 = Magboltz.EFIELD * Magboltz.CONST3
    F4 = 2 * math.acos(-1)
    J2M = Magboltz.NMAX / ISAMP
    random.seed(Magboltz.RSTART)
    for J1 in range(int(J2M)):
        R5 = 1
        TEST1 = 0
        while R5 > TEST1:
            R1 = random.uniform(RDUM)om()
            T = -1 * np.log(R1) / Magboltz.TCFMX + TDASH
            TDASH = T
            AP = DCZ1 * F2 * math.sqrt(E1)
            E = E1 + (AP + BP * T) * T
            if E1/E<0:
                R5 = 1
                TEST1 = 0
                continue
            CONST6 = math.sqrt(E1 / E)
            DCX2 = DCX1 * CONST6
            DCY2 = DCY1 * CONST6
            DCZ2 = DCZ1 * CONST6 + Magboltz.EFIELD * T * Magboltz.CONST5 / math.sqrt(E)
            R2 = random.random()
            KGAS = 0
            for KGAS in range(Magboltz.NGAS):
                if Magboltz.TCFMXG[KGAS] >= R2:
                    break
            IMBPT = IMBPT + 1
            if IMBPT > 6:
                Magboltz.RNMX = GERJAN(Magboltz.RSTART, Magboltz.NGAS)
                IMBPT = 1
            VGX = Magboltz.VTMB[KGAS] * Magboltz.RNMX[(IMBPT-1)%6]
            IMBPT = IMBPT + 1
            VGY = Magboltz.VTMB[KGAS] * Magboltz.RNMX[(IMBPT-1)%6]
            IMBPT = IMBPT + 1
            VGZ = Magboltz.VTMB[KGAS] * Magboltz.RNMX[(IMBPT-1)%6]

            VEX = DCX2 * CONST9 * math.sqrt(E)
            VEY = DCY2 * CONST9 * math.sqrt(E)
            VEZ = DCZ2 * CONST9 * math.sqrt(E)

            EOK = ((VEX - VGX) ** 2 + (VEY - VGY) ** 2 + (VEZ - VGZ) ** 2) / CONST10
            IE = int(EOK / Magboltz.ESTEP)
            IE = min(IE, 3999)
            R5 = random.random()
            TEST1 = Magboltz.TCF[KGAS][IE] / Magboltz.TCFMAX[KGAS]

        if IE == 3999:
            Magboltz.IELOW = 1
            return Magboltz
            
        TDASH = 0.0

        CONST11 = 1 / (CONST9 * math.sqrt(EOK))
        DXCOM = (VEX - VGX) * CONST11
        DYCOM = (VEY - VGY) * CONST11
        DZCOM = (VEZ - VGZ) * CONST11

        R2 = random.random()

        I = SORTT(KGAS, I, R2, IE, Magboltz)
        while Magboltz.CF[KGAS][IE][I] < R2:
            I = I + 1
        S1 = Magboltz.RGAS[KGAS][I]
        EI = Magboltz.EIN[KGAS][I]
        if Magboltz.IPN[KGAS][I] > 0:
            R9 = random.random()
            EXTRA = R9 * (EOK - EI)
            EI = EXTRA + EI
        IPT = Magboltz.IARRY[KGAS][I]
        if EOK < EI:
            EI = EOK - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random.random()

        if Magboltz.INDEX[KGAS][I] == 1:
            R31 = random.random()
            F3 = 1.0 - R3*Magboltz.ANGCT[KGAS][IE][I]
            if R31 > Magboltz.PSCT[KGAS][IE][I]:
                F3 = -1 * F3
        elif Magboltz.INDEX[KGAS][I] == 2:
            EPSI = Magboltz.PSCT[KGAS][IE][I]
            F3 = 1 - (2 * R3 * (1 - EPSI)) / (1 + EPSI * (1 - 2 * R3))
        else:
            F3 = 1 - 2 * R3
        THETA0 = math.acos(F3)
        R4 = random.random()
        PHI0 = F4 * R4
        F8 = math.sin(PHI0)
        F9 = math.cos(PHI0)
        ARG1 = 1 - S1 * EI / EOK
        ARG1 = max(ARG1, SMALL)

        D = 1 - F3 * math.sqrt(ARG1)
        E1 = EOK * (1 - EI / (S1 * EOK) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = math.sqrt((EOK / E1) * ARG1) / S1
        Q = min(Q, 1)
        Magboltz.THETA = math.asin(Q * math.sin(THETA0))

        F6 = math.cos(Magboltz.THETA)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = F3 ** 2

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6
        F5 = math.sin(Magboltz.THETA)
        DZCOM = min(DZCOM, 1)
        ARGZ = math.sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
        if ARGZ == 0:
            DCZ1 = F6
            DCX1 = F9 * F5
            DCY1 = F8 * F5
        else:
            DCZ1 = DZCOM * F6 + ARGZ * F5 * F8
            DCY1 = DYCOM * F6 + (F5 / ARGZ) * (DXCOM * F9 - DYCOM * DZCOM * F8)
            DCX1 = DXCOM * F6 - (F5 / ARGZ) * (DYCOM * F9 + DXCOM * DZCOM * F8)
        # TRANSFORM VELOCITY VECTORS TO LAB FRAME
        CONST12 = CONST9 * math.sqrt(E1)
        VXLAB = DCX1 * CONST12 + VGX
        VYLAB = DCY1 * CONST12 + VGY
        VZLAB = DCZ1 * CONST12 + VGZ
        #  CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
        E1 = (VXLAB * VXLAB + VYLAB * VYLAB + VZLAB * VZLAB) / CONST10
        CONST11 = 1.0 / (CONST9 * math.sqrt(E1))
        DCX1 = VXLAB * CONST11
        DCY1 = VYLAB * CONST11
        DCZ1 = VZLAB * CONST11

    Magboltz.IELOW = 0
    return Magboltz
    
