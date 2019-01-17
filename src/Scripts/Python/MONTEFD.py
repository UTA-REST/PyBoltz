import numpy as np
import math
from GERJAN import GERJAN
from SORT import SORT
from SPLANE import SPLANE
from TCALC import TCALC
from goto import with_goto

@with_goto
def MONTEFD(Magboltz):
    
    EPRM = np.zeros(10000000)
    IESPECP = np.zeros(100)
    TEMP = np.zeros(4000)
    S = 0.0
    Magboltz.ST = 0.0
    Magboltz.X = 0.0
    Magboltz.Y = 0.0
    Magboltz.Z = 0.0
    R = np.zeros(4)
    I100 = 0
    Magboltz.ZTOT = 0.0
    I=0
    Magboltz.ZTOTS = 0.0
    Magboltz.TTOT = 0.0
    Magboltz.TTOTS = 0.0
    Magboltz.SMALL = 1e-20
    Magboltz.TMAX1 = 0.0
    RDUM = Magboltz.RSTART
    E1 = Magboltz.ESTART
    CONST9 = Magboltz.CONST3 * 0.01
    CONST10 = CONST9 ** 2
    Magboltz.API = math.acos(-1)

    for I in range(300):
        Magboltz.TIME[I] = 0.0

    for I in range(30):
        Magboltz.ICOLL[I] = 0
    for I in range(60):
        Magboltz.ICOLNN[I] = 0
    for I in range(4000):
        Magboltz.SPEC[I] = 0

    for I in range(8):
        Magboltz.ESPL[I] = 0.0
        Magboltz.XSPL[I] = 0.0
        Magboltz.YSPL[I] = 0.0
        Magboltz.ZSPL[I] = 0.0
        Magboltz.TSPL[I] = 0.0
        Magboltz.XXSPL[I] = 0.0
        Magboltz.YYSPL[I] = 0.0
        Magboltz.ZZSPL[I] = 0.0
        Magboltz.TSSUM[I] = 0.0
        Magboltz.VZSPL[I] = 0.0
        Magboltz.TSSUM2[I] = 0.0
        Magboltz.TTMSPL[I] = 0.0
        Magboltz.TMSPL[I] = 0.0
        Magboltz.RSPL[I] = 0.0
        Magboltz.RRSPL[I] = 0.0
        Magboltz.RRSPM[I] = 0.0
        Magboltz.NESST[I] = 0

    Magboltz.NESST[8] = 0
    ID = 0
    NCOL = 0
    Magboltz.NNULL = 0
    NELEC = 0
    NEION = 0
    NMXADD = 0
    NTMPFLG = 0
    NPONT = 0
    NCLUS = 0
    J1 = 1
    ZSTRT = 0.0
    TSSTRT = 0.0
    INTEM = 8

    for J in range(4000):
        TEMP[J] = Magboltz.TCF[J] + Magboltz.TCFN[J]

    ABSFAKEI = abs(Magboltz.FAKEI)
    Magboltz.IFAKE = 0
    DCZ1 = math.cos(Magboltz.THETA)
    DCX1 = math.sin(Magboltz.THETA) * math.cos(Magboltz.PHI)
    DCY1 = math.sin(Magboltz.THETA) * math.sin(Magboltz.PHI)

    E100 = E1
    DCZ100 = DCZ1
    DCX100 = DCX1
    DCY100 = DCY1
    BP = Magboltz.EFIELD ** 2 * Magboltz.CONST1
    F1 = Magboltz.EFIELD * Magboltz.CONST2
    F2 = Magboltz.EFIELD * Magboltz.CONST3
    F4 = 2 * Magboltz.API

    TLIM = Magboltz.TCFMAX[0]
    for I in range(1, INTEM):
        if TLIM < Magboltz.TCFMAX[I]:
            TLIM = Magboltz.TCFMAX[I]

    JPRINT = Magboltz.NMAX / 10
    IPRINT = 0
    ITER = 0
    IPLANE = 0
    Magboltz.IPRIM = 0
    label.L544
    Magboltz.IPRIM += 1
    IZPLANE = 0
    TZSTOP = 1000
    if Magboltz.IPRIM > 1:
        if ITER > Magboltz.NMAX:
            Magboltz.IPRIM -= 1
            goto.L700

        Magboltz.X = 0.0
        Magboltz.Y = 0.0
        Magboltz.Z = 0.0
        DCZ1 = DCZ100
        DCX1 = DCX100
        DCY1 = DCY100
        E1 = E100
        NCLUS += 1
        Magboltz.ST = 0.0
        TSSTRT = 0.0
        ZSTRT = 0.0
    EPRM[Magboltz.IPRIM - 1] = E1
    IDUM = int(E1)
    IDUM = min(IDUM, 99)
    IESPECP[IDUM] += 1
    label.L555
    TDASH = 0.0
    NELEC += 1
    TZSTOP1 = 0

    label.L1
    R1 = Magboltz.RAND48.drand()
    T = -1 * np.log(R1) / Magboltz.TCFMX + TDASH
    TOLD = TDASH
    TDASH = T
    AP = DCZ1 * F2 * math.sqrt(E1)
    label.L15
    if T >= TZSTOP and TOLD < TZSTOP:
        TLFT = TZSTOP
        SPLANE(Magboltz,T, E1, DCX1, DCY1, DCZ1, AP, BP, TLFT, IZPLANE)
        if IZPLANE >= Magboltz.IZFINAL + 1:
            label.L18
            Magboltz.ZTOT += Magboltz.Z
            Magboltz.TTOT += Magboltz.ST
            Magboltz.ZTOTS += Magboltz.Z - ZSTRT
            Magboltz.TTOTS += Magboltz.ST - TSSTRT
            if NELEC == NCLUS + 1:
                goto.L544
            label.L20
            Magboltz.X = Magboltz.XSS[NPONT]
            Magboltz.Y = Magboltz.YSS[NPONT]
            Magboltz.Z = Magboltz.ZSS[NPONT]
            Magboltz.ST = Magboltz.TSS[NPONT]
            E1 = Magboltz.ESS[NPONT]
            DCX1 = Magboltz.DCXS[NPONT]
            DCY1 = Magboltz.DCYS[NPONT]
            DCZ1 = Magboltz.DCZS[NPONT]
            IZPLANE = Magboltz.IPLS[NPONT]
            NPONT -= 1
            ZSTRT = Magboltz.Z
            TSSTRT = Magboltz.ST
            if Magboltz.Z > Magboltz.ZFINAL:
                EPOT = Magboltz.EFIELD * (Magboltz.Z - Magboltz.ZFINAL) * 100
                if E1 < EPOT:
                    NELEC += 1
                    ISOL = 1
                    goto.L18
            R = TCALC(Magboltz, DCZ1, E1, TZSTOP, TZSTOP1, ISOL, IZPLANE)
            ISOL = R[0]
            IZPLANE = R[1]
            TZSTOP = R[2]
            TZSTOP1 = R[3]
            if TZSTOP == -99:
                NELEC += 1
                ISOL = 1
                goto.L18
            goto.L555
        if ISOL == 2:
            TZSTOP = TZSTOP1
            ISOL = 1
            goto.L15

    E = E1 + (AP + BP * T) * T
    if E < 0:
        E = 0.001
    IE = np.int(E / Magboltz.ESTEP)
    IE = min(IE, 3999)
    R5 = Magboltz.RAND48.drand()
    TEST1 = Magboltz.TCF[IE] / TLIM
    if R5 > TEST1:
        Magboltz.NNULL += 1
        TEST2 = TEMP[IE] / TLIM
        if R5 < TEST2:
            if Magboltz.NPLAST == 0:
                goto.L1
            R2 = Magboltz.RAND48.drand()
            I = 0
            while Magboltz.CFN[IE][I] < R2:
                I += 1

            Magboltz.ICOLNN[I] += 1
            goto.L1
        else:
            TEST3 = (TEMP[IE] + ABSFAKEI) / TLIM
            if R5 < TEST3:
                # FAKE IONISATION INCREMENT COUNTER
                Magboltz.IFAKE += 1
                Magboltz.IFAKED[IZPLANE] += 1
                if Magboltz.FAKEI < 0.0:
                    NEION += 1
                    if NELEC == NCLUS + 1:
                        goto.L544
                    goto.L20
            NCLUS += 1
            NPONT += 1
            NMXADD = max(NPONT, NMXADD)
            if NPONT >= 2000:
                raise ValueError("NPONT>2000")
            A = T * CONST9 * math.sqrt(E1)
            Magboltz.XSS[NPONT] = Magboltz.X + DCX1 * A
            Magboltz.YSS[NPONT] = Magboltz.Y + DCY1 * A
            Magboltz.ZSS[NPONT] = Magboltz.Z + DCZ1 * A + T * T * F1
            Magboltz.TSS[NPONT] = Magboltz.ST + T
            Magboltz.ESS[NPONT] = E
            CONST6 = math.sqrt(E1 / E)
            Magboltz.DCXS[NPONT] = DCX1 * CONST6
            Magboltz.DCYS[NPONT] = DCY1 * CONST6
            Magboltz.DCZS[NPONT] = DCZ1 * CONST6 + Magboltz.EFIELD * T * Magboltz.CONST5 / math.sqrt(E)
            IDM1 = 1 + int(Magboltz.ZSS[NPONT] / Magboltz.ZSTEP)
            if IDM1 < 1:
                IDM1 = 1
            if IDM1 > 9:
                IDM1 = 9
            Magboltz.IPLS[NPONT] = IDM1
            Magboltz.NESST[Magboltz.IPLS[NPONT] - 1] += 1
            goto.L1
        goto.L1

    T2 = T ** 2

    if (T >= Magboltz.TMAX1):
        Magboltz.TMAX1 = T
    CONST6 = math.sqrt(E1 / E)
    DCX2 = DCX1 * CONST6
    DCY2 = DCY1 * CONST6
    DCZ2 = DCZ1 * CONST6 + Magboltz.EFIELD * T * Magboltz.CONST5 / math.sqrt(E)
    A = AP * T
    B = BP * T2
    CONST7 = CONST9 * math.sqrt(E1)
    A = T * CONST7
    NCOL += 1
    CZ1 = DCZ1 * CONST7
    Magboltz.X += DCX1 * A
    Magboltz.Y += DCY1 * A
    Magboltz.Z += DCZ1 * A + T2 * F1
    Magboltz.ST += T
    IT = int(T)
    IT = min(IT, 299)
    Magboltz.TIME[IT] += 1
    Magboltz.SPEC[IE] += 1

    CX1 = DCX1 * CONST7
    CY1 = DCY1 * CONST7
    R2 = Magboltz.RAND48.drand()
    I = SORT(I, R2, IE, Magboltz)

    while Magboltz.CF[IE][I] < R2:
        I += 1
    S1 = Magboltz.RGAS[I]
    EI = Magboltz.EIN[I]
    if E < EI:
        EI = E - 0.0001
    if Magboltz.IPN[I] != 0:
        if Magboltz.IPN[I] == -1:
            NEION += 1
            IPT = Magboltz.IARRY[I]
            ID += 1
            ITER += 1
            IPRINT += 1
            Magboltz.ICOLL[IPT] += 1
            Magboltz.ICOLN[I] += 1
            IT = int(T)
            IT = min(IT, 299)
            Magboltz.TIME[IT] += 1
            Magboltz.ZTOT += Magboltz.Z
            Magboltz.TTOT += Magboltz.ST
            Magboltz.ZTOTS += Magboltz.Z - ZSTRT
            Magboltz.TTOTS += Magboltz.ST - TSSTRT
            IDM1 = 1 + int(Magboltz.Z / Magboltz.ZSTEP)
            if IDM1 < 1:
                IDM1 = 1
            if IDM1 > 9:
                IDM1 = 9
            Magboltz.NESST[IDM1 - 1] -= 1
            if NELEC == NCLUS + 1:
                goto.L544
            goto.L20
        EISTR = EI
        R9 = Magboltz.RAND48.drand()
        ESEC = Magboltz.WPL[I] * math.tan(R9 * math.atan((E - EI) / (2 * Magboltz.WPL[I])))
        ESEC = Magboltz.WPL[I] * (ESEC / Magboltz.WPL[I]) ** 0.9524
        EI = ESEC + EI
        NCLUS += 1
        NPONT += 1
        NMXADD = max(NPONT, NMXADD)
        if NPONT >= 2000:
            raise ValueError("NPONT>2000")
        Magboltz.XSS[NPONT] = Magboltz.X
        Magboltz.YSS[NPONT] = Magboltz.Y
        Magboltz.ZSS[NPONT] = Magboltz.Z
        Magboltz.TSS[NPONT] = Magboltz.ST
        Magboltz.ESS[NPONT] = ESEC
        NTMPFLG = 1
        NCLTMP = NPONT
        IDM1 = 1 + int(Magboltz.Z / Magboltz.ZSTEP)
        if IDM1 < 1:
            IDM1 = 1
        if IDM1 > 9:
            IDM1 = 9
        Magboltz.IPLS[NPONT] = IDM1
        if EISTR > 30:
            NAUG = Magboltz.NC0[I]
            EAVAUG = Magboltz.EC0[I] / float(NAUG)
            for JFL in range(int(NAUG)):
                NCLUS += 1
                NPONT += 1
                Magboltz.XSS[NPONT] = Magboltz.X
                Magboltz.YSS[NPONT] = Magboltz.Y
                Magboltz.ZSS[NPONT] = Magboltz.Z
                Magboltz.TSS[NPONT] = Magboltz.ST
                Magboltz.ESS[NPONT] = EAVAUG
                R3 = Magboltz.RAND48.drand()
                F3 = 1 - 2 * R3
                THETA0 = math.acos(F3)
                F6 = math.cos(THETA0)
                F5 = math.sin(THETA0)
                R4 = Magboltz.RAND48.drand()
                PHI0 = F4 * R4
                F8 = math.sin(PHI0)
                F9 = math.cos(PHI0)
                Magboltz.DCXS[NPONT] = F9 * F5
                Magboltz.DCYS[NPONT] = F8 * F5
                Magboltz.DCZS[NPONT] = F6
                IDM1 = 1 + int(Magboltz.Z / Magboltz.ZSTEP)
                if IDM1 < 1:
                    IDM1 = 1
                if IDM1 > 9:
                    IDM1 = 9
                Magboltz.IPLS[NPONT] = IDM1
                Magboltz.NESST[IDM1 - 1] += 1
    IPT = Magboltz.IARRY[I]
    ID += 1
    ITER += 1
    IPRINT += 1
    Magboltz.ICOLL[IPT] += 1
    Magboltz.ICOLN[I] += 1

    if Magboltz.IPEN != 0:
        if Magboltz.PENFRA[0][I] != 0.0:
            RAN = Magboltz.RAND48.drand()
            if RAN > Magboltz.PENFRA[0][I]:
                goto.L5
            NCLUS += 1
            NPONT += 1
            if NPONT >= 2000:
                raise ValueError("NPONT>2000")
            if Magboltz.PENFRA[1][I] == 0.0:
                Magboltz.XSS[NPONT] = Magboltz.X
                Magboltz.YSS[NPONT] = Magboltz.Y
                Magboltz.ZSS[NPONT] = Magboltz.Z
                if Magboltz.ZSS[NPONT] > Magboltz.ZFINAL or Magboltz.ZSS[NPONT] < 0:
                    goto.L669
                goto.L667
            ASIGN = 1.0
            RAN = Magboltz.RAND48.drand()
            RAN1 = Magboltz.RAND48.drand()
            if RAN1 < 0.5:
                ASIGN = -1 * ASIGN
            Magboltz.XSS[NPONT] = Magboltz.X - np.log(RAN) * Magboltz.PENFRA[1][I] * ASIGN
            RAN = Magboltz.RAND48.drand()
            RAN1 = Magboltz.RAND48.drand()
            if RAN1 < 0.5:
                ASIGN = -1 * ASIGN
            Magboltz.YSS[NPONT] = Magboltz.Y - np.log(RAN) * Magboltz.PENFRA[1][I] * ASIGN
            RAN = Magboltz.RAND48.drand()
            RAN1 = Magboltz.RAND48.drand()
            if RAN1 < 0.5:
                ASIGN = -1 * ASIGN
            Magboltz.ZSS[NPONT] = Magboltz.Z - np.log(RAN) * Magboltz.PENFRA[1][I] * ASIGN
            if Magboltz.ZSS[NPONT] < 0.0:
                goto.L669
            if Magboltz.ZSS[NPONT] > Magboltz.ZFINAL or Magboltz.ZSS[NPONT] < 0.0:
                goto.L669
            label.L667
            TPEN = Magboltz.ST

            if Magboltz.PENFRA[2][I] == 0:
                goto.L668
            RAN = Magboltz.RAND48.drand()
            TPEN = Magboltz.ST - np.log(RAN) * Magboltz.PENFRA[2][I]
            label.L668
            Magboltz.TSS[NPONT] = TPEN
            Magboltz.ESS[NPONT] = 1.0
            Magboltz.DCXS[NPONT] = DCX1
            Magboltz.DCYS[NPONT] = DCY1
            Magboltz.DCZS[NPONT] = DCZ1
            IDM1 = 1 + int(Magboltz.Z / Magboltz.ZSTEP)
            if IDM1 < 1:
                IDM1 = 1
            if IDM1 > 9:
                IDM1 = 9
            Magboltz.IPLS[NPONT] = IDM1
            Magboltz.NESST[IDM1 - 1] += 1
            goto.L5
            label.L669
            NPONT -= 1
            NCLUS -= 1
    label.L5
    S2 = (S1 ** 2) / (S1 - 1)
    R3 = Magboltz.RAND48.drand()
    if Magboltz.INDEX[I] == 1:
        R31 = Magboltz.RAND48.drand()
        F3 = 1.0 - R3 * Magboltz.ANGCT[IE][I]
        if R31 > Magboltz.PSCT[IE][I]:
            F3 = -1 * F3
    elif Magboltz.INDEX[I] == 2:
        EPSI = Magboltz.PSCT[IE][I]
        F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
    else:
        F3 = 1 - 2 * R3
    THETA0 = math.acos(F3)
    R4 = Magboltz.RAND48.drand()
    PHI0 = F4 * R4
    F8 = math.sin(PHI0)
    F9 = math.cos(PHI0)
    if E < EI:
        EI = 0.0
    ARG1 = 1 - S1 * EI / E
    ARG1 = max(ARG1, Magboltz.SMALL)
    D = 1 - F3 * math.sqrt(ARG1)
    E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)
    E1 = max(E1, Magboltz.SMALL)
    Q = math.sqrt((E / E1) * ARG1) / S1
    Q = min(Q, 1)
    Magboltz.THETA = math.asin(Q * math.sin(THETA0))
    F6 = math.cos(Magboltz.THETA)
    U = (S1 - 1) * (S1 - 1) / ARG1

    CSQD = F3 * F3

    if F3 < 0 and CSQD > U:
        F6 = -1 * F6
    F5 = math.sin(Magboltz.THETA)
    DCZ2 = min(DCZ2, 1)
    ARGZ = math.sqrt(DCX2 * DCX2 + DCY2 * DCY2)
    if ARGZ == 0:
        DCZ1 = F6
        DCX1 = F9 * F5
        DCY1 = F8 * F5
        if NTMPFLG == 1:
            F5S = F5 * math.sqrt(E1 / Magboltz.ES[NCLTMP])
            if F5S > 1.0:
                F5S = 1.0
            THSEC = math.asin(F5S)
            F5S = math.sin(THSEC)
            F6S = math.sin(THSEC)
            if F6S < 0:
                F6S *= -1
            PHIS = PHI0 + Magboltz.API
            if PHIS > F4:
                PHIS = PHI0 - F4
            F8S = math.sin(PHIS)
            F9S = math.cos(PHIS)
            Magboltz.DCZS[NCLTMP] = F6S
            Magboltz.DCXS[NCLTMP] = F9S * F5S
            Magboltz.DCYS[NCLTMP] = F8S * F5S
            NTMPFLG = 0
    else:
        DCZ1 = DCZ2 * F6 + ARGZ * F5 * F8
        DCY1 = DCY2 * F6 + (F5 / ARGZ) * (DCX2 * F9 - DCY2 * DCZ2 * F8)
        DCX1 = DCX2 * F6 - (F5 / ARGZ) * (DCY2 * F9 + DCX2 * DCZ2 * F8)
    I100 += 1
    if I100 == 200:
        DCZ100 = DCZ1
        DCX100 = DCX1
        DCY100 = DCY1
        E100 = E1
        I100 = 0
    if Magboltz.Z > Magboltz.ZFINAL:
        EPOT = Magboltz.EFIELD * (Magboltz.Z - Magboltz.ZFINAL) * 100
        if E1 < EPOT:
            goto.L18
    R = TCALC(Magboltz, DCZ1, E1, TZSTOP, TZSTOP1, ISOL, IZPLANE)
    ISOL = R[0]
    IZPLANE = R[1]
    TZSTOP = R[2]
    TZSTOP1 = R[3]
    if TZSTOP == -99:
        goto.L18
    if IPRINT <= JPRINT:
        goto.L1
    IPRINT = 0
    W = Magboltz.ZTOTS / Magboltz.TTOTS
    W *= 1e9
    XID = float(ID)
    JCT = ID / 100000
    J1 += 1
    goto.L1

    label.L700
    XID = float(ID)
    if NELEC > Magboltz.IPRIM:
        ANEION = float(NEION)
        ANBT = float(NELEC - Magboltz.IPRIM)
        ATTOINT = ANEION / ANBT
        Magboltz.ATTERT = math.sqrt(ANEION) / ANEION
        Magboltz.AIOERT = math.sqrt(ANBT) / ANBT
    else:
        ANEION = float(NEION)
        ATTOINT = -1
        Magboltz.ATTERT = math.sqrt(ANEION) / ANEION
    JCT = ID / 100000

    EPRMBAR = 0.0
    E2PRM = 0.0
    if Magboltz.IPRIM == 1:
        return
    for I in range(int(Magboltz.IPRIM)):
        E2PRM = E2PRM + EPRM[I] * EPRM[I]
        EPRMBAR += EPRM
    EBAR = EPRMBAR / (Magboltz.IPRIM)
    EERR = math.sqrt(E2PRM / (Magboltz.IPRIM) - EBAR ** 2)
    # IF ITER >NMAX

