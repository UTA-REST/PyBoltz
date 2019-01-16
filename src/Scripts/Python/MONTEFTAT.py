import numpy as np
import math
from GERJAN import GERJAN
from SORTT import SORTT
from TPLANEAT import TPLANEAT

from goto import with_goto

@with_goto
def MONTEFTAT(Magboltz, JPRT):
    EPRM = np.zeros(10000000)
    IESPECP = np.zeros(100)
    TEMP = np.zeros(shape=(6, 4000))
    IMBPT = 0
    if JPRT == 0:
        Magboltz.NMAX = Magboltz.NMAXOLD
        if Magboltz.NMAXOLD > 80000000:
            Magboltz.NMAX = 80000000
    else:
        Magboltz.NMAX = Magboltz.NMAXOLD
    S = 0.0
    Magboltz.ST = 0.0
    Magboltz.X = 0.0
    Magboltz.Y = 0.0
    Magboltz.Z = 0.0
    I100 = 0
    Magboltz.ZTOT = 0.0
    Magboltz.ZTOTS = 0.0
    Magboltz.TTOT = 0.0
    Magboltz.TTOTS = 0.0
    Magboltz.SMALL = 1e-20
    Magboltz.TMAX1 = 0.0
    RDUM = Magboltz.RSTART
    E1 = Magboltz.ESTART
    CONST9 = Magboltz.CONST3 * 0.01
    CONST10 = CONST9 ** 2

    for I in range(300):
        Magboltz.TIME[I] = 0.0

    for K in range(6):
        for I in range(5):
            Magboltz.ICOLL[K][I] = 0

    for K in range(6):
        for I in range(290):
            Magboltz.ICOLN[K][I] = 0

    for K in range(6):
        for I in range(10):
            Magboltz.ICOLNN[K][I] = 0
    for I in range(4000):
        Magboltz.SPEC[I] = 0

    for I in range(8):
        Magboltz.ETPL[I] = 0.0
        Magboltz.XTPL[I] = 0.0
        Magboltz.YTPL[I] = 0.0
        Magboltz.ZTPL[I] = 0.0
        Magboltz.TTPL[I] = 0.0
        Magboltz.XXTPL[I] = 0.0
        Magboltz.YYTPL[I] = 0.0
        Magboltz.ZZTPL[I] = 0.0
        Magboltz.VZTPL[I] = 0.0
        Magboltz.NETPL[I] = 0.0
        Magboltz.IFAKET[I] = 0
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
    for K in range(6):
        for J in range(4000):
            TEMP[K][J] = Magboltz.TCF[K][J] + Magboltz.TCFN[K][J]

    ABSFAKEI = abs(Magboltz.FAKEI)
    Magboltz.IFAKE = 0
    for J in range(8):
        Magboltz.IFAKET[J] = 0
    Magboltz.RNMX = GERJAN(Magboltz.RAND48, Magboltz.API)
    DCZ1 = math.cos(Magboltz.THETA)
    DCX1 = math.sin(Magboltz.THETA) * math.cos(Magboltz.PHI)
    DCY1 = math.sin(Magboltz.THETA) * math.sin(Magboltz.PHI)

    VTOT = CONST9 * math.sqrt(E1)
    CX1 = DCX1 * VTOT
    CY1 = DCY1 * VTOT
    CZ1 = DCZ1 * VTOT

    E100 = E1
    DCZ100 = DCZ1
    DCX100 = DCX1
    DCY100 = DCY1
    BP = Magboltz.EFIELD ** 2 * Magboltz.CONST1
    F1 = Magboltz.EFIELD * Magboltz.CONST2
    F2 = Magboltz.EFIELD * Magboltz.CONST3
    Magboltz.API = math.acos(-1)
    F4 = 2 * Magboltz.API
    JPRINT = Magboltz.NMAX / 10

    IPRINT = 0
    ITER = 0
    IPLANE = 0
    Magboltz.IPRIM = 0

    label.L544
    Magboltz.IPRIM += 1
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
        VTOT = CONST9 * math.sqrt(E1)
        CX1 = DCX1 * VTOT
        CY1 = DCY1 * VTOT
        CZ1 = DCZ1 * VTOT
        NCLUS += 1
        Magboltz.ST = 0.0
        TSSTRT = 0.0
        ZSTRT = 0.0
        IPLANE = 0
    if Magboltz.IPRIM > 10000000:
        goto.L700
    EPRM[Magboltz.IPRIM - 1] = E1
    IDUM = int(E1)
    IDUM = min(IDUM, 99)
    IESPECP[IDUM] += 1

    label.L555
    TDASH = 0.0
    NELEC += 1
    TSTOP = Magboltz.TSTEP + IPLANE * Magboltz.TSTEP
    label.L1
    R1 = Magboltz.RAND48.drand()
    T = -1 * np.log(R1) / Magboltz.TCFMX + TDASH
    TDASH = T
    AP = DCZ1 * F2 * math.sqrt(E1)
    label.L15
    if T + Magboltz.ST >= TSTOP:
        IPLANE += 1
        TSTOP += Magboltz.TSTEP
        Magboltz = TPLANEAT(Magboltz, T, E1, CX1, CY1, DCZ1, AP, BP, IPLANE - 1)
        if T + Magboltz.ST >= TSTOP and TSTOP <= Magboltz.TFINAL:
            goto.L15
        if T + Magboltz.ST >= Magboltz.TFINAL:
            Magboltz.ZTOT += Magboltz.Z
            Magboltz.TTOT += Magboltz.ST
            Magboltz.ZTOTS += Magboltz.Z - ZSTRT
            Magboltz.TTOTS += Magboltz.ST - TSSTRT
            TSTOP = Magboltz.TSTEP
            if NELEC == NCLUS + 1:
                goto.L544
            label.L20
            Magboltz.X = Magboltz.XS[NPONT]
            Magboltz.Y = Magboltz.YS[NPONT]
            Magboltz.Z = Magboltz.ZS[NPONT]
            Magboltz.ST = Magboltz.TS[NPONT]
            E1 = Magboltz.ES[NPONT]
            DCX1 = Magboltz.DCX[NPONT]
            DCY1 = Magboltz.DCY[NPONT]
            DCZ1 = Magboltz.DCZ[NPONT]
            VTOT = CONST9 * math.sqrt(E1)
            CX1 = DCX1 * VTOT
            CY1 = DCY1 * VTOT
            CZ1 = DCZ1 * VTOT
            IPLANE = Magboltz.IPL[NPONT]
            NPONT -= 1
            ZSTRT = Magboltz.Z
            TSSTRT = Magboltz.ST
        goto.L555
    E = E1 + (AP + BP * T) * T
    if E < 0:
        E = 0.001
    WBT = Magboltz.WB * T
    COSWT = math.cos(WBT)
    SINWT = math.sin(WBT)
    CONST6 = math.sqrt(E1 / E)
    R2 = Magboltz.RAND48.drand()
    if Magboltz.NGAS == 1:
        KGAS = 0
    while (Magboltz.TCFMXG[KGAS] < R2):
        KGAS = KGAS + 1

    CX2 = CX1 * COSWT - CY1 * SINWT
    CY2 = CY1 * COSWT + CX1 * SINWT
    VTOT = CONST9 * math.sqrt(E)
    CZ2 = VTOT * (DCZ1 * CONST6 + Magboltz.EFIELD * T * Magboltz.CONST5 / math.sqrt(E))

    IMBPT += 1
    if (IMBPT > 5):
        Magboltz.RNMX = GERJAN(Magboltz.RAND48, Magboltz.API)
        IMBPT = 0
    VGX = Magboltz.VTMB[KGAS] * Magboltz.RNMX[IMBPT % 6]
    IMBPT += 1
    VGY = Magboltz.VTMB[KGAS] * Magboltz.RNMX[IMBPT % 6]
    IMBPT += 1
    VGZ = Magboltz.VTMB[KGAS] * Magboltz.RNMX[IMBPT % 6]
    EOK = ((CX2 - VGX) ** 2 + (CY2 - VGY) ** 2 + (CZ2 - VGZ) ** 2) / CONST10

    IE = np.int(EOK / Magboltz.ESTEP) + 1
    IE = min(IE, 3999)
    R5 = Magboltz.RAND48.drand()
    TEST1 = Magboltz.TCF[KGAS][IE] / Magboltz.TCFMAX[KGAS]

    if R5 > TEST1:
        Magboltz.NNULL += 1
        TEST2 = TEMP[KGAS][IE] / Magboltz.TCFMAX[KGAS]
        if R5 < TEST2:
            if Magboltz.NPLAST == 0:
                goto.L1
            R2 = Magboltz.RAND48.drand()
            I = 0
            while Magboltz.CFN[KGAS][IE][I] < R2:
                I += 1

            Magboltz.ICOLNN[KGAS][I] += 1
            goto.L1
        else:
            TEST3 = (TEMP[KGAS][IE] + ABSFAKEI) / Magboltz.TCFMAX[KGAS]
            if R5 < TEST3:
                # FAKE IONISATION INCREMENT COUNTER
                Magboltz.IFAKE += 1
                Magboltz.IFAKET[IPLANE - 1] += 1
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
            Magboltz.XS[NPONT] = Magboltz.X + (CX1 * SINWT - CY1 * (1 - COSWT)) / Magboltz.WB
            Magboltz.YS[NPONT] = Magboltz.Y + (CY1 * SINWT + CX1 * (1 - COSWT)) / Magboltz.WB
            Magboltz.ZS[NPONT] = Magboltz.Z + DCZ1 * T * CONST9 * math.sqrt(E1) + T * T * F1
            Magboltz.TS[NPONT] = Magboltz.ST + T
            Magboltz.ES[NPONT] = E
            Magboltz.IPL[NPONT] = IPLANE
            BOT = 1 / math.sqrt((CX2 * CX2 + CY2 * CY2 + CZ2 * CZ2))
            Magboltz.DCX[NPONT] = CX2 * BOT
            Magboltz.DCY[NPONT] = CY2 * BOT
            Magboltz.DCZ[NPONT] = CZ2 * BOT
            goto.L1
    NCOL += 1
    CONST11 = 1 / (CONST9 * math.sqrt(EOK))
    DXCOM = (CX2 - VGX) * CONST11
    DYCOM = (CY2 - VGY) * CONST11
    DZCOM = (CZ2 - VGZ) * CONST11

    T2 = T ** 2

    if (T >= Magboltz.TMAX1):
        Magboltz.TMAX1 = T
    TDASH = 0.0

    CONST7 = CONST9 * math.sqrt(E1)
    A = T * CONST7
    DX = (CX1 * SINWT - CY1 * (1 - COSWT)) / Magboltz.WB
    Magboltz.X += DX
    DY = (CY1 * SINWT + CX1 * (1 - COSWT)) / Magboltz.WB
    Magboltz.Y += DY
    Magboltz.Z += DCZ1 * A + T2 * F1
    Magboltz.ST += T
    IT = int(T)
    IT = min(IT, 299)
    Magboltz.TIME[IT] += 1
    Magboltz.SPEC[IE] += 1

    R2 = Magboltz.RAND48.drand()
    I = SORTT(KGAS, I, R2, IE)

    while Magboltz.CF[KGAS][IE][I] < R2:
        I += 1
    S1 = Magboltz.RGAS[KGAS][I]
    EI = Magboltz.EIN[KGAS][I]
    if EOK < EI:
        EI = EOK - 0.0001

    if Magboltz.IPN[KGAS][I] != 0:
        if Magboltz.IPN[KGAS][I] == -1:
            NEION += 1
            IPT = Magboltz.IARRY[KGAS][I]
            ID += 1
            ITER += 1
            IPRINT += 1
            Magboltz.ICOLL[KGAS][IPT] += 1
            Magboltz.ICOLN[KGAS][I] += 1
            IT = int(T)
            IT = min(IT, 299)
            Magboltz.TIME[IT] += 1
            Magboltz.ZTOT += Magboltz.Z
            Magboltz.TTOT += Magboltz.ST
            Magboltz.ZTOTS += Magboltz.Z - ZSTRT
            Magboltz.TTOTS += Magboltz.ST - TSSTRT
            if NELEC == NCLUS + 1:
                goto.L544
            goto.L20
        EISTR = EI
        R9 = Magboltz.RAND48.drand()
        ESEC = Magboltz.WPL[KGAS][I] * math.tan(R9 * math.atan((EOK - EI) / (2 * Magboltz.WPL[KGAS][I])))
        ESEC = Magboltz.WPL[KGAS][I] * (ESEC / Magboltz.WPL[KGAS][I]) ** 0.9524
        EI = ESEC + EI
        NCLUS += 1
        NPONT += 1
        NMXADD = max(NPONT, NMXADD)
        if NPONT >= 2000:
            raise ValueError("NPONT>2000")
        Magboltz.XS[NPONT] = Magboltz.X
        Magboltz.YS[NPONT] = Magboltz.Y
        Magboltz.ZS[NPONT] = Magboltz.Z
        Magboltz.TS[NPONT] = Magboltz.ST
        Magboltz.ES[NPONT] = ESEC
        NTMPFLG = 1
        NCLTMP = NPONT
        Magboltz.IPL[NPONT] = IPLANE
        if EISTR > 30:
            NAUG = Magboltz.NC0[KGAS][I]
            EAVAUG = Magboltz.EC0[KGAS][I] / float(NAUG)
            for JFL in range(int(NAUG)):
                NCLUS += 1
                NPONT += 1
                Magboltz.XS[NPONT] = Magboltz.X
                Magboltz.YS[NPONT] = Magboltz.Y
                Magboltz.ZS[NPONT] = Magboltz.Z
                Magboltz.TS[NPONT] = Magboltz.ST
                Magboltz.ES[NPONT] = EAVAUG
                R3 = Magboltz.RAND48.drand()
                F3 = 1 - 2 * R3
                THETA0 = math.acos(F3)
                F6 = math.cos(THETA0)
                F5 = math.sin(THETA0)
                R4 = Magboltz.RAND48.drand()
                PHI0 = F4 * R4
                F8 = math.sin(PHI0)
                F9 = math.cos(PHI0)
                Magboltz.DCX[NPONT] = F9 * F5
                Magboltz.DCY[NPONT] = F8 * F5
                Magboltz.DCZ[NPONT] = F6
                Magboltz.IPL[NPONT] = IPLANE
    IPT = Magboltz.IARRY[KGAS][I]
    ID += 1
    ITER += 1
    IPRINT += 1
    Magboltz.ICOLL[KGAS][IPT] += 1
    Magboltz.ICOLN[KGAS][I] += 1

    if Magboltz.IPEN != 0:
        if Magboltz.PENFRA[KGAS][0][I] != 0.0:
            RAN = Magboltz.RAND48.drand()
            if RAN > Magboltz.PENFRA[KGAS][0][I]:
                goto.L5
            NCLUS += 1
            NPONT += 1
            if NPONT >= 2000:
                raise ValueError("NPONT>2000")
            if Magboltz.PENFRA[KGAS][1][I] == 0.0:
                Magboltz.XSS[NPONT] = Magboltz.X
                Magboltz.YSS[NPONT] = Magboltz.Y
                Magboltz.ZSS[NPONT] = Magboltz.Z
                goto.L667
            ASIGN = 1.0
            RAN = Magboltz.RAND48.drand()
            RAN1 = Magboltz.RAND48.drand()
            if RAN1 < 0.5:
                ASIGN = -1 * ASIGN
            Magboltz.XS[NPONT] = Magboltz.X - np.log(RAN) * Magboltz.PENFRA[KGAS][1][I] * ASIGN
            RAN = Magboltz.RAND48.drand()
            RAN1 = Magboltz.RAND48.drand()
            if RAN1 < 0.5:
                ASIGN = -1 * ASIGN
            Magboltz.YS[NPONT] = Magboltz.Y - np.log(RAN) * Magboltz.PENFRA[KGAS][1][I] * ASIGN
            RAN = Magboltz.RAND48.drand()
            RAN1 = Magboltz.RAND48.drand()
            if RAN1 < 0.5:
                ASIGN = -1 * ASIGN
            Magboltz.ZS[NPONT] = Magboltz.Z - np.log(RAN) * Magboltz.PENFRA[KGAS][1][I] * ASIGN
            label.L667
            TPEN = Magboltz.ST

            if Magboltz.PENFRA[KGAS][2][I] == 0:
                goto.L668
            RAN = Magboltz.RAND48.drand()
            TPEN = Magboltz.ST - np.log(RAN) * Magboltz.PENFRA[KGAS][2][I]
            label.L668
            Magboltz.TS[NPONT] = TPEN
            Magboltz.ES[NPONT] = 1.0
            Magboltz.DCX[NPONT] = DCX1
            Magboltz.DCY[NPONT] = DCY1
            Magboltz.DCZ[NPONT] = DCZ1

            TSTOP1 = 0.0
            IPLANE1 = 0
            for KDUM in range(int(Magboltz.ITFINAL)):
                TSTOP1 += Magboltz.TSTEP
                if TPEN < TSTOP1:
                    Magboltz.IPL[NPONT] = IPLANE1
                    break
                IPLANE1 += 1
            if TPEN >= TSTOP1:
                NPONT -= 1
                NCLUS -= 1
    label.L5

    S2 = (S1 ** 2) / (S1 - 1)
    R3 = Magboltz.RAND48.drand()
    if Magboltz.INDEX[KGAS][I] == 1:
        R31 = Magboltz.RAND48.drand()
        F3 = 1.0 - R3 * Magboltz.ANGCT[KGAS][IE][I]
        if R31 > Magboltz.PSCT[KGAS][IE][I]:
            F3 = -1 * F3
    elif Magboltz.INDEX[KGAS][I] == 2:
        EPSI = Magboltz.PSCT[KGAS][IE][I]
        F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
    else:
        F3 = 1 - 2 * R3
    THETA0 = math.acos(F3)
    R4 = Magboltz.RAND48.drand()
    PHI0 = F4 * R4
    F8 = math.sin(PHI0)
    F9 = math.cos(PHI0)

    ARG1 = 1 - S1 * EI / EOK
    ARG1 = max(ARG1, Magboltz.SMALL)
    D = 1 - F3 * math.sqrt(ARG1)
    E1 = EOK * (1 - EI / (S1 * EOK) - 2 * D / S2)
    E1 = max(E1, Magboltz.SMALL)
    Q = math.sqrt((EOK / E1) * ARG1) / S1
    Q = min(Q, 1)
    Magboltz.THETA = math.asin(Q * math.sin(THETA0))
    F6 = math.cos(Magboltz.THETA)
    U = (S1 - 1) * (S1 - 1) / ARG1

    CSQD = F3 * F3

    if F3 < 0 and CSQD > U:
        F6 = -1 * F6
    F5 = math.sin(Magboltz.THETA)
    DZCOM = min(DZCOM, 1)
    ARGZ = math.sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
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
            Magboltz.DCZ[NCLTMP] = F6S
            Magboltz.DCX[NCLTMP] = F9S * F5S
            Magboltz.DCY[NCLTMP] = F8S * F5S
            NTMPFLG = 0
    else:
        DCZ1 = DZCOM * F6 + ARGZ * F5 * F8
        DCY1 = DYCOM * F6 + (F5 / ARGZ) * (DXCOM * F9 - DYCOM * DZCOM * F8)
        DCX1 = DXCOM * F6 - (F5 / ARGZ) * (DYCOM * F9 + DXCOM * DZCOM * F8)
        if NTMPFLG == 1:
            F5S = F5 * math.sqrt(E1 / Magboltz.ES[NCLTMP])
            if F5S > 1.0:
                F5S = 1.0
            THSEC = math.asin(F5S)
            F5S = math.sin(THSEC)
            F6S = math.sin(THSEC)
            if F6 < 0:
                F6S *= -1
            PHIS = PHI0 + Magboltz.API
            if PHIS > F4:
                PHIS = PHI0 - F4
            F8S = math.sin(PHIS)
            F9S = math.cos(PHIS)
            Magboltz.DCZS[NCLTMP] = DZCOM * F6S + ARGZ * F5S * F8S
            Magboltz.DCYS[NCLTMP] = DYCOM * F6S + (F5S / ARGZ) * (DXCOM * F9S - DYCOM * DZCOM * F8S)
            Magboltz.DCXS[NCLTMP] = DXCOM * F6S - (F5S / ARGZ) * (DYCOM * F9S + DXCOM * DZCOM * F8S)
            NTMPFLG = 0
    VTOT = CONST9 * math.sqrt(E1)
    CX1 = DCX1 * VTOT + VGX
    CY1 = DCY1 * VTOT + VGY
    CZ1 = DCZ1 * VTOT + VGZ
    E1 = (CX1 ** 2 + CY1 ** 2 + CZ1 ** 2) / CONST10
    CONST11 = 1 / (CONST9 * math.sqrt(E1))
    DCX1 = CX1 * CONST11
    DCY1 = CY1 * CONST11
    DCZ1 = CZ1 * CONST11
    I100 += 1
    if I100 == 200:
        DCZ100 = DCZ1
        DCX100 = DCX1
        DCY100 = DCY1
        E100 = E1
        I100 = 0

    if IPRINT <= JPRINT:
        goto.L1
    IPRINT = 0
    W = Magboltz.ZTOTS / Magboltz.TTOTS
    W *= 1e9
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
        return Magboltz
    for I in range(int(Magboltz.IPRIM)):
        E2PRM = E2PRM + EPRM[I] * EPRM[I]
        EPRMBAR += EPRM
    EBAR = EPRMBAR / (Magboltz.IPRIM)
    EERR = math.sqrt(E2PRM / (Magboltz.IPRIM) - EBAR ** 2)
    # IF ITER >NMAX
    return Magboltz
