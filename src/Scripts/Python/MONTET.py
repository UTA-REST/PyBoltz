from GERJAN import GERJAN
import numpy as np
import math
from RAND48 import Rand48

from SORTT import SORTT


def MONTET(Magboltz):
    Magboltz.WX = 0.0
    Magboltz.WY = 0.0
    Magboltz.X = 0.0
    Magboltz.Y = 0.0
    Magboltz.Z = 0.0
    ST = 0.0
    ST1 = 0.0
    ST2 = 0.0
    SUME2 = 0.0
    SUMXX = 0.0
    SUMYY = 0.0
    SUMZZ = 0.0
    SUMVX = 0.0
    SUMVY = 0.0
    ZOLD = 0.0
    STOLD = 0.0
    ST1OLD = 0.0
    ST2OLD = 0.0
    SZZOLD = 0.0
    SXXOLD = 0.0
    SYYOLD = 0.0
    SVXOLD = 0.0
    SVYOLD = 0.0
    SME2OLD = 0.0
    STO = np.zeros(2000000)
    XST = np.zeros(2000000)
    YST = np.zeros(2000000)
    ZST = np.zeros(2000000)
    WZST = np.zeros(10)
    AVEST = np.zeros(10)
    DFZZST = np.zeros(10)
    DFYYST = np.zeros(10)
    DFXXST = np.zeros(10)
    Magboltz.SMALL = 1.0e-20
    Magboltz.TMAX1 = 0.0
    RDUM = Magboltz.RSTART
    E1 = Magboltz.ESTART
    CONST9 = Magboltz.CONST3 * 0.01
    CONST10 = CONST9 ** 2
    Magboltz.ITMAX = 10
    ID = 0
    Magboltz.XID = 0
    NCOL = 0
    IEXTRA = 0
    Magboltz.RNMX = GERJAN(Magboltz.RAND48, Magboltz.API)
    IMBPT = 0
    TDASH = 0.0
    TEMP = np.zeros(shape=(6, 4000))
    for K in range(6):
        for J in range(4000):
            TEMP[K][J] = Magboltz.TCF[K][J] + Magboltz.TCFN[K][J]
    ABSFAKEI = Magboltz.FAKEI
    Magboltz.IFAKE = 0

    DCZ1 = math.cos(Magboltz.THETA)
    DCX1 = math.sin(Magboltz.THETA) * math.cos(Magboltz.PHI)
    DCY1 = math.sin(Magboltz.THETA) * math.sin(Magboltz.PHI)

    BP = (Magboltz.EFIELD ** 2) * Magboltz.CONST1
    F1 = Magboltz.EFIELD * Magboltz.CONST2
    F2 = Magboltz.EFIELD * Magboltz.CONST3
    F4 = 2 * math.acos(-1)
    J2M = Magboltz.NMAX / Magboltz.ITMAX
    Magboltz.RAND48.seed(RDUM)
    for J1 in range(Magboltz.ITMAX):
        for J2 in range(J2M):
            while True:
                R1 = Magboltz.RAND48.drand()
                T = -1 * np.log(R1) / Magboltz.TCFMX + TDASH
                TDASH = T
                AP = DCZ1 * F2 * math.sqrt(E1)
                E = E1 + (AP + BP * T) * T
                CONST6 = math.sqrt(E1 / E)
                # CALCULATE DIRECTION COSINES BEFORE COLLISION
                DCX2 = DCX1 * CONST6
                DCY2 = DCY1 * CONST6
                DCZ2 = DCZ1 * CONST6 + Magboltz.EFIELD * T * Magboltz.CONST5 / math.sqrt(E)
                # FIND IDENTITY OF GAS FOR COLLISION
                KGAS = 0
                R2 = Magboltz.RAND48.drand()
                if Magboltz.NGAS == 1:
                    KGAS = 0

                while (Magboltz.TCFMXG[KGAS] < R2):
                    KGAS = KGAS + 1
                IMBPT += 1
                if (IMBPT > 5):
                    Magboltz.RNMX = GERJAN(Magboltz.RAND48, Magboltz.API)
                    IMBPT = 0

                VGX = Magboltz.VTMB[KGAS] * Magboltz.RNMX[IMBPT%6]
                IMBPT += 1
                VGY = Magboltz.VTMB[KGAS] * Magboltz.RNMX[IMBPT%6]
                IMBPT += 1
                VGZ = Magboltz.VTMB[KGAS] * Magboltz.RNMX[IMBPT%6]
                # CALCULATE ELECTRON VELOCITY VECTORS VEX VEY VEZ
                VEX = DCX2 * CONST9 * math.sqrt(E)
                VEY = DCY2 * CONST9 * math.sqrt(E)
                VEZ = DCZ2 * CONST9 * math.sqrt(E)
                # CALCULATE ENERGY WITH STATIONARY GAS TARGET, EOK

                EOK = ((VEX - VGX) ** 2 + (VEY - VGY) ** 2 + (VEZ - VGZ) ** 2) / CONST10
                IE = int(EOK / Magboltz.ESTEP)
                IE = min(IE, 3999)
                # TEST FOR REAL OR NULL COLLISION

                R5 = Magboltz.RAND48.drand()
                TEST1 = Magboltz.TCF[KGAS][IE] / Magboltz.TCFMAX[KGAS]

                if R5 > TEST1:
                    Magboltz.NNULL += 1
                    TEST2 = TEMP[KGAS][IE] / Magboltz.TCFMAX[KGAS]
                    if R5 < TEST2:
                        if Magboltz.NPLAST == 0:
                            continue
                        R2 = Magboltz.RAND48.drand()
                        I = 0
                        while Magboltz.CFN[KGAS][IE][I] < R2:
                            I += 1

                        Magboltz.ICOLNN[KGAS][I] += 1
                        continue
                    else:
                        TEST3 = (TEMP[KGAS][IE] + ABSFAKEI) / Magboltz.TCFMAX[KGAS]
                        if R5 < TEST3:
                            # FAKE IONISATION INCREMENT COUNTER
                            Magboltz.IFAKE += 1
                            continue
                        continue
                else:
                    break
            NCOL += 1
            CONST11 = 1 / (CONST9 * math.sqrt(EOK))
            DXCOM = (VEX - VGX) * CONST11
            DYCOM = (VEY - VGY) * CONST11
            DZCOM = (VEZ - VGZ) * CONST11
            # CALCULATE POSITIONS AT INSTANT BEFORE COLLISION, & UPDATE DIFFUSION AND ENERGY CALCULATIONS
            T2 = T ** 2
            if (T >= Magboltz.TMAX1):
                Magboltz.TMAX1 = T
            TDASH = 0.0
            A = AP * T
            B = BP * T2
            SUME2 = SUME2 + T * (E1 + A / 2.0 + B / 3.0)
            CONST7 = CONST9 * math.sqrt(E1)
            A = T * CONST7
            CX1 = DCX1 * CONST7
            CY1 = DCY1 * CONST7
            Magboltz.X = Magboltz.X + DCX1 * A
            Magboltz.Y = Magboltz.Y + DCY1 * A
            Magboltz.Z = Magboltz.Z + DCZ1 * A + T2 * F1
            ST = ST + T
            IT = int(T)
            IT = min(IT, 299)
            Magboltz.TIME[IT] += 1
            Magboltz.SPEC[IE] += 1
            Magboltz.WZ = Magboltz.Z / ST
            SUMVX = SUMVX + CX1 * CX1 * T2
            SUMVY = SUMVY + CY1 * CY1 * T2
            if ID != 0:
                KDUM = 0
                for JDUM in range(Magboltz.NCORST):
                    ST2 = ST2 + T
                    NCOLDM = NCOL + KDUM
                    if NCOLDM > Magboltz.NCOLM:
                        NCOLDM = NCOLDM - Magboltz.NCOLM
                    SDIF = ST - STO[NCOLDM]
                    SUMXX += ((Magboltz.X - XST[NCOLDM]) ** 2) * T / SDIF
                    SUMYY += ((Magboltz.Y - YST[NCOLDM]) ** 2) * T / SDIF
                    KDUM += Magboltz.NCORLN
                    if J1 >= 2:
                        ST1 += T
                        SUMZZ += ((Magboltz.Z - ZST[NCOLDM] - Magboltz.WZ * SDIF) ** 2) * T / SDIF
            XST[NCOL] = Magboltz.X
            YST[NCOL] = Magboltz.Y
            ZST[NCOL] = Magboltz.Z
            STO[NCOL] = ST
            if NCOL >= Magboltz.NCOLM:
                ID += 1
                Magboltz.XID = float(ID)
                NCOL = 0

            R3 = Magboltz.RAND48.drand()

            I = SORTT(KGAS, I, R3, IE)

            while Magboltz.CF[KGAS][IE][I] < R3:
                I += 1
            S1 = Magboltz.RGAS[KGAS][I]
            EI = Magboltz.EIN[KGAS][I]

            if Magboltz.IPN[KGAS][I] > 0:
                R9 = Magboltz.RAND48.drand()
                EXTRA = R9 * (EOK - EI)
                EI = EXTRA + EI
                IEXTRA += Magboltz.NC0[KGAS][I]
            IPT = Magboltz.IARRY[KGAS][I]
            Magboltz.ICOLL[KGAS][IPT] += 1
            Magboltz.ICOLN[KGAS][I] += 1
            if EOK < EI:
                EI = EOK - 0.0001

            if Magboltz.IPEN != 0:
                if Magboltz.PENFRA[KGAS][0][I] != 0:
                    RAN = Magboltz.RAND48.drand()
                    if RAN <= Magboltz.PENFRA[KGAS][0][I]:
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

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
            else:
                DCZ1 = DZCOM * F6 + ARGZ * F5 * F8
                DCY1 = DYCOM * F6 + (F5 / ARGZ) * (DXCOM * F9 - DYCOM * DZCOM * F8)
                DCX1 = DXCOM * F6 - (F5 / ARGZ) * (DYCOM * F9 + DXCOM * DZCOM * F8)
            CONST12 = CONST9 * math.sqrt(E1)
            VXLAB = DCX1 * CONST12 + VGX
            VYLAB = DCY1 * CONST12 + VGY
            VZLAB = DCZ1 * CONST12 + VGZ

            E1 = (VXLAB * VXLAB + VYLAB * VYLAB + VZLAB * VZLAB) / CONST10
            CONST11 = 1 / (CONST9 * math.sqrt(E1))
            DCX1 = VXLAB * CONST11
            DCY1 = VYLAB * CONST11
            DCZ1 = VZLAB * CONST11
        # TODO: TABLE PRINT
        Magboltz.WZ *= 1e9
        Magboltz.AVE = SUME2 / ST
        DIFLN = 0.0
        if Magboltz.NISO == 0:
            Magboltz.DIFXX = 5e15 * SUMVX / ST
            Magboltz.DIFYY = 5e15 * SUMVY / ST
            DFXXST[J1] = 5e15 * (SUMVX - SVXOLD) / (ST - STOLD)
            DFYYST[J1] = 5e15 * (SUMVY - SVYOLD) / (ST - STOLD)
        else:
            if ST2 != 0.0:
                Magboltz.DIFYY = 5e15 * SUMYY / ST2
                Magboltz.DIFXX = 5e15 * SUMXX / ST2
                DFXXST[J1] = 5e15 * (SUMXX - SXXOLD) / (ST2 - ST2OLD)
                DFYYST[J1] = 5e15 * (SUMYY - SYYOLD) / (ST2 - ST2OLD)
            else:
                DFXXST[J1] = 0.0
                DFYYST[J1] = 0.0

        if ST1 != 0.0:
            Magboltz.DIFZZ = 5e15 * SUMZZ / ST1
            DFZZST[J1] = 5e15 * (SUMZZ - SZZOLD) / (ST1 - ST1OLD)
        else:
            DFZZST[J1] = 0.0
        WZST[J1] = (Magboltz.Z - ZOLD) / (ST - STOLD) * 1e9
        AVEST[J1] = (SUME2 - SME2OLD) / (ST - STOLD)
        ZOLD = Magboltz.Z
        STOLD = ST
        ST1OLD = ST1
        ST2OLD = ST2
        SVXOLD = SUMVX
        SVYOLD = SUMVY
        SZZOLD = SUMZZ
        SYYOLD = SUMYY
        SXXOLD = SUMXX
        SME2OLD = SUME2
        # TODO: CONTINUE TABLE PRINTING

        if Magboltz.SPEC[3999] > (1000 * float(J1)):
            raise ValueError("WARNING ENERGY OUT OF RANGE, INCREASE ELECTRON ENERGY INTEGRATION RANGE")
    TWZST = 0.0
    TAVE = 0.0
    T2WZST = 0.0
    T2AVE = 0.0
    TZZST = 0.0
    TYYST = 0.0
    TXXST = 0.0
    T2ZZST = 0.0
    T2YYST = 0.0
    T2XXST = 0.0
    for K in range(10):
        TWZST = TWZST + WZST[K]
        TAVE = TAVE + AVEST[K]
        T2WZST = T2WZST + WZST[K] * WZST[K]
        T2AVE = T2AVE + AVEST[K] * AVEST[K]
        TXXST = TXXST + DFXXST[K]
        TYYST = TYYST + DFYYST[K]
        T2YYST = T2YYST + DFYYST[K] ** 2
        T2XXST = T2XXST + DFXXST[K] ** 2
        if K >= 2:
            TZZST = TZZST + DFZZST[K]
            T2ZZST += DFZZST[K] ** 2
    Magboltz.DWZ = 100 * math.sqrt((T2WZST - TWZST * TWZST / 10.0) / 9.0) / Magboltz.WZ
    Magboltz.DEN = 100 * math.sqrt((T2AVE - TAVE * TAVE / 10.0) / 9.0) / Magboltz.AVE
    Magboltz.DXXER = 100 * math.sqrt((T2XXST - TXXST * TXXST / 10.0) / 9.0) / Magboltz.DIFXX
    Magboltz.DYYER = 100 * math.sqrt((T2YYST - TYYST * TYYST / 10.0) / 9.0) / Magboltz.DIFYY
    Magboltz.DZZER = 100 * math.sqrt((T2ZZST - TZZST * TZZST / 8.0) / 7.0) / Magboltz.DIFZZ
    Magboltz.DWZ = Magboltz.DWZ / math.sqrt(10)
    Magboltz.DEN = Magboltz.DEN / math.sqrt(10)
    Magboltz.DXXER = Magboltz.DXXER / math.sqrt(10)
    Magboltz.DYYER = Magboltz.DYYER / math.sqrt(10)
    Magboltz.DZZER = Magboltz.DZZER / math.sqrt(8)
    DIFLN = Magboltz.DIFZZ
    Magboltz.DIFTR = (Magboltz.DIFXX + Magboltz.DYYER) / 2
    # CONVERT CM/SEC
    Magboltz.WZ *= 1e5
    Magboltz.DFLER = Magboltz.DZZER
    Magboltz.DFTER = (Magboltz.DXXER + Magboltz.DYYER) / 2.0

    ANCATT = 0.0
    ANCION = 0.0
    for I in range(Magboltz.NGAS):
        ANCATT += Magboltz.ICOLL[I][2]
        ANCION += Magboltz.ICOLL[I][1]
    ANCION += IEXTRA
    Magboltz.ATTER = 0.0

    if ANCATT != 0:
        Magboltz.ATTER = 100 * math.sqrt(ANCATT) / ANCATT
    Magboltz.ATT = ANCATT / (ST * Magboltz.WZ) * 1e12
    Magboltz.ALPER = 0.0
    if ANCION != 0:
        Magboltz.ALPER = 100 * math.sqrt(ANCION) / ANCION
    Magboltz.ALPHA = ANCION / (ST * Magboltz.WZ) * 1e12

    return Magboltz
