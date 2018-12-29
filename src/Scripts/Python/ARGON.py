import h5py
import numpy as np
import math


def Gas2(object):
    gd = h5py.File(r"gases.hdf5")
    APOL = 11.08
    LMAX = 100
    AA = -1.459
    DD = 68.93
    FF = -97.0
    A1 = 8.69
    EMASS2 = 1021997.804
    API = math.acos(-1)
    A0 = 0.52917720859e-8
    RY = 13.60569193
    BBCONST = 16.0 * API * A0 * A0 * RY * RY / EMASS2
    CONST = 1.873884e-20
    AM2 = 3.593
    C = 39.70
    PSCALE = 0.9
    AUGL3 = 2.0
    AUGL2 = 1.0
    AUGL1 = 2.63
    AUGK = 3.39
    object.NION = 7
    object.NATT = 1
    object.NIN = 44
    object.NNULL = 0
    NBREM = 25

    for i in range(0, 6):
        object.KEL[i] = object.NANISO
    for i in range(0, object.NIN):
        object.KIN[i] = object.NANISO
    NDATA = 117
    NEPSI = 217
    NIDATA = 75
    NION2 = 47
    NION3 = 36
    NKSH = 89
    NL1S = 101
    NL2S = 104
    NL3S = 104
    N1S5 = 71
    N1S4 = 79
    N1S3 = 70
    N1S2 = 70
    N2P10 = 54
    N2P9 = 17
    N2P8 = 15
    N2P7 = 17
    N2P6 = 16
    N2P5 = 17
    N2P4 = 17
    N2P3 = 17
    N2P2 = 16
    N2P1 = 17
    N3D6 = 19
    N3D5 = 26
    N3D3 = 20
    N3D4P = 20
    N3D4 = 23
    N3D1PP = 19
    N2S5 = 19
    N3D1P = 16
    N3S1PPPP = 21
    N3S1PP = 21
    N3S1PPP = 16
    N2S3 = 19
    PIR2 = 8.7973554297e-17
    EMASS = 9.10938291e-31
    AMU = 1.660538921e-27
    object.E = [0.0, 1.0, 15.9, 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * EMASS / (39.948 * AMU)
    EOBY = [9.5, 18.0, 34.0, 110.0, 110.0, 150.0, 1800]
    object.EION = [15.75961, 43.38928, 84.124, 248.4, 250.6, 326.3, 3205.9]
    LEGAS = [0, 0, 0, 1, 1, 1, 1]
    ISHELL = [0, 0, 0, 4, 3, 2, 1]
    object.NC0 = [0, 1, 2, 2, 2, 3, 4]
    object.EC0 = [0.0, 6.0, 12.0, 210.5, 202.2, 240.8, 3071]
    WKLM = [0.0, 0.0, 0.0, 0.00147, 0.00147, 0.00147, 0.12]
    object.EFL = [0.0, 0.0, 0.0, 232, 235, 310, 2957]
    object.NG1 = [0, 0, 0, 1, 1, 2, 3]
    object.EG1 = [0.0, 0.0, 0.0, 210.5, 202.2, 240.8, 2850]
    object.NG2 = [0.0, 0.0, 0.0, 1, 1, 1, 2]
    object.EG2 = [0.0, 0.0, 5.0, 5.0, 5.0, 220]

    IOFFION = [0 for x in range(10)]
    IOFFN = [0 for x in range(44)]
    for j in range(0,object.NION):
        for i in range(0, 4000):
            if object.EG[i] > object.EION[j]:
                IOFFION[j] = i - 1
                break
    EIN = gd['gas2/EIN']
    XEN = gd['gas2/XEN']
    YSEC = gd['gas2/YSEC']
    YEL = gd['gas2/YEL']
    XEPS = gd['gas2/XEPS']
    YEPS = gd['gas2/YEPS']
    XENI = gd['gas2/XENI']
    YENI = gd['gas2/YENI']
    YENC = gd['gas2/YENC']
    YEN1 = gd['gas2/YEN1']
    XEN2 = gd['gas2/XEN2']
    YEN2 = gd['gas2/YEN2']
    XEN3 = gd['gas2/XEN3']
    YEN3 = gd['gas2/YEN3']
    XKSH = gd['gas2/XKSH']
    YKSH = gd['gas2/YKSH']
    XL1S = gd['gas2/XL1S']
    YL1S = gd['gas2/YL1S']
    XL2S = gd['gas2/XL2S']
    YL2S = gd['gas2/YL2S']
    XL3S = gd['gas2/XL3S']
    YL3S = gd['gas2/YL3S']
    X1S5 = gd['gas2/X1S5']
    Y1S5 = gd['gas2/Y1S5']
    YEPS1 = gd['gas2/YEPS1']
    X1S4 = gd['gas2/X1S4']
    Y1S4 = gd['gas2/Y1S4']
    YEPS2 = gd['gas2/YEPS2']
    X1S3 = gd['gas2/X1S3']
    Y1S3 = gd['gas2/Y1S3']
    YEPS3 = gd['gas2/YEPS3']
    X1S2 = gd['gas2/X1S2']
    Y1S2 = gd['gas2/Y1S2']
    YEPS4 = gd['gas2/YEPS4']
    X2P10 = gd['gas2/X2P10']
    Y2P10 = gd['gas2/Y2P10']
    YEP2P10 = gd['gas2/YEP2P10']
    X2P9 = gd['gas2/X2P9']
    Y2P9 = gd['gas2/Y2P9']
    YEP2P9 = gd['gas2/YEP2P9']
    X2P8 = gd['gas2/X2P8']
    Y2P8 = gd['gas2/Y2P8']
    YEP2P8 = gd['gas2/YEP2P8']
    X2P7 = gd['gas2/X2P7']
    Y2P7 = gd['gas2/Y2P7']
    YEP2P7 = gd['gas2/YEP2P7']
    X2P6 = gd['gas2/X2P6']
    Y2P6 = gd['gas2/Y2P6']
    YEP2P6 = gd['gas2/YEP2P6']
    X2P5 = gd['gas2/X2P5']
    Y2P5 = gd['gas2/Y2P5']
    YEP2P5 = gd['gas2/YEP2P5']
    X2P4 = gd['gas2/X2P4']
    Y2P4 = gd['gas2/Y2P4']
    YEP2P4 = gd['gas2/YEP2P4']
    X2P3 = gd['gas2/X2P3']
    Y2P3 = gd['gas2/Y2P3']
    YEP2P3 = gd['gas2/YEP2P3']
    X2P2 = gd['gas2/X2P2']
    Y2P2 = gd['gas2/Y2P2']
    YEP2P2 = gd['gas2/YEP2P2']
    X2P1 = gd['gas2/X2P1']
    Y2P1 = gd['gas2/Y2P1']
    YEP2P1 = gd['gas2/YEP2P1']
    X3D6 = gd['gas2/X3D6']
    Y3D6 = gd['gas2/Y3D6']
    YEP3D6 = gd['gas2/YEP3D6']
    X3D5 = gd['gas2/X3D5']
    Y3D5 = gd['gas2/Y3D5']
    YEP3D5 = gd['gas2/YEP3D5']
    X3D4P = gd['gas2/X3D4P']
    Y3D4P = gd['gas2/Y3D4P']
    YEP3D4P = gd['gas2/YEP3D4P']
    X3D4 = gd['gas2/X3D4']
    Y3D4 = gd['gas2/Y3D4']
    YEP3D4 = gd['gas2/YEP3D4']
    X3D3 = gd['gas2/X3D3']
    Y3D3 = gd['gas2/Y3D3']
    YEP3D3 = gd['gas2/YEP3D3']
    X3D1PP = gd['gas2/X3D1PP']
    Y3D1PP = gd['gas2/Y3D1PP']
    YEP3D1PP = gd['gas2/YEP3D1PP']
    X3D1P = gd['gas2/X3D1P']
    Y3D1P = gd['gas2/Y3D1P']
    YEP3D1P = gd['gas2/YEP3D1P']
    X3S1PPPP = gd['gas2/X3S1PPPP']
    Y3S1PPPP = gd['gas2/Y3S1PPPP']
    YEP3S1PPPP = gd['gas2/YEP3S1PPPP']
    X3S1PPP = gd['gas2/X3S1PPP']
    Y3S1PPP = gd['gas2/Y3S1PPP']
    YEP3S1PPP = gd['gas2/YEP3S1PPP']
    X3S1PP = gd['gas2/X3S1PP']
    Y3S1PP = gd['gas2/Y3S1PP']
    YEP3S1PP = gd['gas2/YEP3S1PP']
    X2S5 = gd['gas2/X2S5']
    Y2S5 = gd['gas2/Y2S5']
    YEP2S5 = gd['gas2/YEP2S5']
    X2S3 = gd['gas2/X2S3']
    Y2S3 = gd['gas2/Y2S3']
    YEP2S3 = gd['gas2/YEP2S3']
    for i in range(object.NIN):
        object.PENFRA[0][i] = 0.2
        object.PENFRA[1][i] = 1.0
        object.PENFRA[2][i] = 1.0
    for i in range(object.NIN):
        for j in range(4000):
            if object.EG[j] > EIN[i]:
                IOFFN[i] = j - 1
                break

    for I in range(0, object.NSTEP):
        EN = object.EG[I]
        # EN=EN+object.ESTEP
        if EN > EIN[0]:
            GAMMA1 = (EMASS2 + 2.0 * EN) / EMASS2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = np.sqrt(1.00 - 1.00 / GAMMA2)
            BETA2 = BETA * BETA
        if EN <= 1:
            if EN == 0:
                QELA = 7.491E-16
                QMOM = 7.491E-16
            if EN != 0:
                AK = np.sqrt(EN / object.ARY)
                AK2 = AK * AK
                AK3 = AK2 * AK
                AK4 = AK3 * AK
                AN0 = -AA * AK * (1.0 + (4.0 * APOL / 3.0) * AK2 * np.log(AK)) - (
                        API * APOL / 3.0) * AK2 + DD * AK3 + FF * AK4
                AN1 = (API / 15.0) * APOL * AK - A1 * AK3
                AN2 = API * APOL * AK2 / 105.0
                AN0 = math.atan(AN0)
                AN1 = math.atan(AN1)
                AN2 = math.atan(AN2)
                ANHIGH = AN2
                SUM = (math.sin(AN0 - AN1)) ** 2
                SUM = SUM + 2.0 * (math.sin(AN1 - AN2)) ** 2
                SIGEL = (math.sin(AN0)) ** 2 + 3.0 * (math.sin(AN1)) ** 2
                for J in range(2, -1, -1):
                    ANLOW = ANHIGH
                    SUMI = 6.0 / ((2.0 * J + 5.0) * (2.0 * J + 3.0) * (2.0 * J + 1.0) * (2.0 * J - 1.0))
                    SUM = SUM + (J + 1.0) * (math.sin(math.atan(API * APOL * AK2 * SUMI))) ** 2
                    ANHIGH = math.atan(API * APOL * AK2 / ((2.0 * J + 5.0) * (2.0 * J + 3.0) * (2.0 * J + 1.0)))
                    SIGEL = SIGEL + (2.0 * J + 1.0) * (math.sin(ANLOW)) ** 2
                QELA = SIGEL * 4.0 * PIR2 / AK2
                QMOM = SUM * 4.0 * PIR2 / AK2
        if EN != 0:
            for J in range(1, NDATA):
                if EN < XEN[J]:
                    break

            A = (YEL[J] - YEL[J - 1]) / (XEN[J] - XEN[J - 1])
            B = (XEN[J - 1] * YEL[J] - XEN[J] * YEL[J - 1]) / (XEN[J - 1] - XEN[J])
            QELA = (A * EN + B) * 1.0e-16
            A = (YSEC[J] - YSEC[J - 1]) / (XEN[J] - XEN[j - 1])
            B = (XEN[J - 1] * YSEC[J] - XEN[J] * YSEC[J - 1]) / (XEN[J - 1] - XEN[J])
            QMOM = (A * EN + B) * 1.0e-16
        PQ1 = 0.5 + (QELA - QMOM) / QELA

        for J in range(1, NEPSI):
            if EN < XEPS[j]:
                break
        A = (YEPS(J) - YEPS(J - 1)) / (XEPS(J) - XEPS(J - 1))
        B = (XEPS(J - 1) * YEPS(J) - XEPS(J) * YEPS(J - 1)) / (XEPS(J - 1) - XEPS(J))
        PQ2 = A * EN + B
        # EPSILON = 1 - PQ2
        PQ2 = 1.0 - PQ2
        PQ = [0.5, PQ1, PQ2]
        object.PEQEL[1][I] = PQ[object.NANISO]
        object.Q[1][I] = QELA
        if object.NANISO == 0:
            object.Q[1][I] = QMOM
        object.QION[0][I] = 0.0
        object.PEQION[0][I] = 0.5

        if object.NANISO == 2:
            object.object.PEQIN[0][I] = 0
        if EN > object.EION[0]:
            if EN <= XENI[NIDATA - 1]:
                j = 0
                for j in range(1, NIDATA):
                    if EN <= XENI[j]:
                        break
                A = (YENI[j] - YENI[j - 1]) / (XENI[j] - XENI[j - 1])
                B = (XENI[j - 1] * YENI[j] - XENI[j] * YENI[j - 1]) / (XENI[j - 1] - XENI[j])
                object.QION[0][I] = (A * EN + B) * 1e-16

            else:
                # USE BORN BETHE X-SECTION ABOVE XENI[NIDATA] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[0][I] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.9466
            if EN > 2 * object.EION[0]:
                object.PEQION[0][I] = object.PEQEL[0][(I - IOFFION[0])]

        object.QION[1][I] = 0.0
        object.PEQION[1][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[1][I] = 0
        if EN > object.EION[1]:
            if EN <= XEN2[NION2 - 1]:
                j = 0
                for j in range(1, NION2):
                    if EN <= XEN2[j]:
                        break
                A = (YEN2[j] - YEN2[j - 1]) / (XEN2[j] - XEN2[j - 1])
                B = (XEN2[j - 1] * YEN2[j] - XEN2[j] * YEN2[j - 1]) / (XEN2[j - 1] - XEN2[j])
                object.QION[1][I] = (A * EN + B) * 1e-16
                if object.QION[1][I] < 0:
                    object.QION[1][I] = 0
            else:
                # USE BORN BETHE X-SECTION ABOVE XEN2[NION2] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[1][I] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.04448
            if EN > 2 * object.EION[1]:
                object.PEQION[1][I] = object.PEQEL[1][(I - IOFFION[1])]

        object.QION[2][I] = 0.0
        object.PEQION[2][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[2][I] = 0
        if EN > object.EION[2]:
            if EN <= XEN3[NION3 - 1]:
                j = 0
                for j in range(1, NION3):
                    if EN <= XEN3[j]:
                        break
                A = (YEN3[j] - YEN3[j - 1]) / (XEN3[j] - XEN3[j - 1])
                B = (XEN3[j - 1] * YEN3[j] - XEN3[j] * YEN3[j - 1]) / (XEN3[j - 1] - XEN3[j])
            object.QION[2][I] = (A * EN + B) * 1e-16
            if object.QION[2][I] < 0:
                object.QION[2][I] = 0
        else:
            # USE BORN BETHE X-SECTION ABOVE XEN3[NION3] EV
            X2 = 1 / BETA2
            X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
            object.QION[2][I] = CONST * (AM2 * (X1 - object.DEN[I] / 2) + C * X2) * 0.00987
        if EN > 2 * object.EION[2]:
            object.PEQION[2][I] = object.PEQEL[1][(I - IOFFION[2])]
        # L3 Shell ionisation
        object.QION[3][i] = 0.0
        object.PEQION[3][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[3][i] = 0.0

        if EN > object.EION[3]:
            for j in range(1, NL3S):
                if EN <= XL3S[j]:
                    break
            A = (YL3S[j] - YL3S[j - 1]) / (XL3S[j] - XL3S[j - 1])
            B = (XL3S[j - 1] * YL3S[j] - XL3S[j] * YL3S[j - 1]) / (XL3S[j - 1] - XL3S[j])
            object.QION[3][I] = (A * EN + B) * 1e-16
            object.PEQION[3][I] = object.PEQEL[1][I - IOFFION[3]]

        # L2 Shell ionisation
        object.QION[4][i] = 0.0
        object.PEQION[4][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[4][i] = 0.0

        if EN > object.EION[4]:
            for j in range(1, NL2S):
                if EN <= XL2S[j]:
                    break
            A = (YL2S[j] - YL2S[j - 1]) / (XL2S[j] - XL2S[j - 1])
            B = (XL2S[j - 1] * YL2S[j] - XL2S[j] * YL2S[j - 1]) / (XL2S[j - 1] - XL2S[j])
            object.QION[4][I] = (A * EN + B) * 1e-16
            object.PEQION[4][I] = object.PEQEL[1][I - IOFFION[4]]
        # L1 Shell ionisation
        object.QION[5][i] = 0.0
        object.PEQION[5][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[5][i] = 0.0

        if EN > object.EION[5]:
            for j in range(1, NL1S):
                if EN <= XL1S[j]:
                    break
            A = (YL1S[j] - YL1S[j - 1]) / (XL1S[j] - XL1S[j - 1])
            B = (XL1S[j - 1] * YL1S[j] - XL1S[j] * YL1S[j - 1]) / (XL1S[j - 1] - XL1S[j])
            object.QION[5][I] = (A * EN + B) * 1e-16
            object.PEQION[5][I] = object.PEQEL[1][I - IOFFION[5]]

        # K Shell ionisation
        object.QION[6][i] = 0.0
        object.PEQION[6][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[6][i] = 0.0

        if EN > object.EION[6]:
            for j in range(1, NKSH):
                if EN <= XKSH[j]:
                    break
            A = (YKSH[j] - YKSH[j - 1]) / (XKSH[j] - XKSH[j - 1])
            B = (XKSH[j - 1] * YKSH[j] - XKSH[j] * YKSH[j - 1]) / (XKSH[j - 1] - XKSH[j])
            object.QION[6][I] = (A * EN + B) * 1e-16
            object.PEQION[6][I] = object.PEQEL[1][I - IOFFION[6]]
        # ATTAchment
        object.Q[3][I] = 0.0
        object.QATT[0][I] = 0.0

        # Counting ionisation
        object.Q[4][I] = 0.0
        object.PEQEL[4][I] = 0.5
        if object.NANISO == 2:
            object.PEQEL[4][I] = 0.0
        if EN > object.E[2]:
            if EN <= XENI[NIDATA]:
                for j in range(1, NIDATA):
                    if EN <= XENI[j]:
                        break
                A = (YENC[j] - YENC[j - 1]) / (XENI[j] - XENI[j - 1])
                B = (XENI[j - 1] * YENC[j] - XENI[j] * YENC[j - 1]) / (XENI[j - 1] - XENI[j])
                object.Q[4][I] = (A * EN + B) * 1.0e-16
            else:
                object.Q[4][I] = CONST * (AM2 * (X1 - object.DEN(I) / 2.0) + C * X2)
        QTEMP = object.QION[3][I] + object.QION[4][I] + object.QION[5][I] + object.QION[6][I]
        if object.Q[4][I] == 0.0:
            QCORR = 1.0
        else:
            QCORR = (Q[4][I] - QTEMP) / object.Q[4][I]
        object.QION[0][I] = object.QION[0][I] * QCORR
        object.QION[1][I] = object.QION[1][I] * QCORR
        object.QION[2][I] = object.QION[2][I] * QCORR

        object.Q[5][I] = 0.0
        for NL in range(object.NIN):
            object.QIN[NL][I] = 0.0
            object.PEQIN[NL][I] = 0.5
            if object.NANISO == 2:
                object.PEQIN[NL][I] = 0.0

        # 1S5

        if EN > EIN[0]:
            if EN <= X1S5[N1S5 - 1]:
                for j in range(1, N1S5):
                    if EN <= X1S5[j]:
                        break
                A = (Y1S5[j] - Y1S5[j - 1]) / (X1S5[j] - X1S5[j - 1])
                B = (X1S5[j - 1] * Y1S5[j] - X1S5[j] * Y1S5[j - 1]) / (X1S5[j - 1] - X1S5[j])
                object.QIN[0][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[0][I] = Y1S5[N1S5 - 1] * (X1S5[N1S5 - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * EIN[0]):
                object.PEQIN[0][I] = object.PEQEL[1][I - IOFFN[0]]

        if EN > EIN[1]:
            if EN <= X1S4[N1S4 - 1]:
                for j in range(1, N1S4):
                    if EN <= X1S4[j]:
                        break
                A = (Y1S4[j] - Y1S4[j - 1]) / (X1S4[j] - X1S4[j - 1])
                B = (X1S4[j - 1] * Y1S4[j] - X1S4[j] * Y1S4[j - 1]) / (X1S4[j - 1] - X1S4[j])
                object.QIN[1][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[1][I] = 0.0580 / (EIN[1] * BETA2) * (
                        np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[1])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (
                                           EN + object.E[2] + EIN[1])
            if EN > (2.0 * EIN[1]):
                object.PEQIN[1][I] = object.PEQEL[1][I - IOFFN[1]]

        # 1S3
        if EN > EIN[2]:
            if EN <= X1S3[N1S3 - 1]:
                for j in range(1, N1S3):
                    if EN <= X1S3[j]:
                        break
                A = (Y1S3[j] - Y1S3[j - 1]) / (X1S3[j] - X1S3[j - 1])
                B = (X1S3[j - 1] * Y1S3[j] - X1S3[j] * Y1S3[j - 1]) / (X1S3[j - 1] - X1S3[j])
                object.QIN[2][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[2][I] = Y1S3[N1S3 - 1] * (X1S3[N1S3 - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * EIN[2]):
                object.PEQIN[2][I] = object.PEQEL[1][I - IOFFN[2]]

        # 1S2 F=0.2260
        if EN > EIN[3]:
            if EN <= X1S2[N1S2 - 1]:
                for j in range(1, N1S2):
                    if EN <= X1S2[j]:
                        break
                A = (Y1S2[j] - Y1S2[j - 1]) / (X1S2[j] - X1S2[j - 1])
                B = (X1S2[j - 1] * Y1S2[j] - X1S2[j] * Y1S2[j - 1]) / (X1S2[j - 1] - X1S2[j])
                object.QIN[3][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[3][I] = 0.2260 / (EIN[3] * BETA2) * (
                        np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[3])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (
                                           EN + object.E[2] + EIN[3])
            if EN > (2.0 * EIN[3]):
                object.PEQIN[3][I] = object.PEQEL[1][I - IOFFN[3]]
        # P states, 2P10
        if EN > EIN[4]:
            if EN <= X2P10[N2P10 - 1]:
                for j in range(1, N2P10):
                    if EN <= X2P10[j]:
                        break
                A = (Y2P10[j] - Y2P10[j - 1]) / (X2P10[j] - X2P10[j - 1])
                B = (X2P10[j - 1] * Y2P10[j] - X2P10[j] * Y2P10[j - 1]) / (X2P10[j - 1] - X2P10[j])
                object.QIN[4][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[4][I] = Y2P10[N2P10 - 1] * (X2P10[N2P10 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[4]):
                object.PEQIN[4][I] = object.PEQEL[1][I - IOFFN[4]]

        # P states, 2P9
        if EN > EIN[5]:
            if EN <= X2P9[N2P9 - 1]:
                for j in range(1, N2P9):
                    if EN <= X2P9[j]:
                        break
                A = (Y2P9[j] - Y2P9[j - 1]) / (X2P9[j] - X2P9[j - 1])
                B = (X2P9[j - 1] * Y2P9[j] - X2P9[j] * Y2P9[j - 1]) / (X2P9[j - 1] - X2P9[j])
                object.QIN[5][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[5][I] = Y2P9[N2P9 - 1] * (X2P9[N2P9 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[5]):
                object.PEQIN[5][I] = object.PEQEL[1][I - IOFFN[5]]

        # P states, 2P8
        if EN > EIN[6]:
            if EN <= X2P8[N2P8 - 1]:
                for j in range(1, N2P8):
                    if EN <= X2P8[j]:
                        break
                A = (Y2P8[j] - Y2P8[j - 1]) / (X2P8[j] - X2P8[j - 1])
                B = (X2P8[j - 1] * Y2P8[j] - X2P8[j] * Y2P8[j - 1]) / (X2P8[j - 1] - X2P8[j])
                object.QIN[6][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[6][I] = Y2P8[N2P8 - 1] * (X2P8[N2P8 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[6]):
                object.PEQIN[6][I] = object.PEQEL[1][I - IOFFN[6]]

        # P states, 2P7
        if EN > EIN[7]:
            if EN <= X2P7[N2P7 - 1]:
                for j in range(1, N2P7):
                    if EN <= X2P7[j]:
                        break
                A = (Y2P7[j] - Y2P7[j - 1]) / (X2P7[j] - X2P7[j - 1])
                B = (X2P7[j - 1] * Y2P7[j] - X2P7[j] * Y2P7[j - 1]) / (X2P7[j - 1] - X2P7[j])
                object.QIN[7][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[7][I] = Y2P7[N2P7 - 1] * (X2P7[N2P7 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[7]):
                object.PEQIN[7][I] = object.PEQEL[1][I - IOFFN[7]]

        # P states, 2P6
        if EN > EIN[8]:
            if EN <= X2P6[N2P6 - 1]:
                for j in range(1, N2P6):
                    if EN <= X2P6[j]:
                        break
                A = (Y2P6[j] - Y2P6[j - 1]) / (X2P6[j] - X2P6[j - 1])
                B = (X2P6[j - 1] * Y2P6[j] - X2P6[j] * Y2P6[j - 1]) / (X2P6[j - 1] - X2P6[j])
                object.QIN[8][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[8][I] = Y2P6[N2P6 - 1] * (X2P6[N2P6 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[8]):
                object.PEQIN[8][I] = object.PEQEL[1][I - IOFFN[8]]

        # P states, 2P5
        if EN > EIN[9]:
            if EN <= X2P5[N2P5 - 1]:
                for j in range(1, N2P5):
                    if EN <= X2P5[j]:
                        break
                A = (Y2P5[j] - Y2P5[j - 1]) / (X2P5[j] - X2P5[j - 1])
                B = (X2P5[j - 1] * Y2P5[j] - X2P5[j] * Y2P5[j - 1]) / (X2P5[j - 1] - X2P5[j])
                object.QIN[9][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[9][I] = Y2P5[N2P5 - 1] * (X2P5[N2P5 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[9]):
                object.PEQIN[9][I] = object.PEQEL[1][I - IOFFN[9]]

        # P states, 2P4
        if EN > EIN[10]:
            if EN <= X2P4[N2P4 - 1]:
                for j in range(1, N2P4):
                    if EN <= X2P4[j]:
                        break
                A = (Y2P4[j] - Y2P4[j - 1]) / (X2P4[j] - X2P4[j - 1])
                B = (X2P4[j - 1] * Y2P4[j] - X2P4[j] * Y2P4[j - 1]) / (X2P4[j - 1] - X2P4[j])
                object.QIN[10][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[10][I] = Y2P4[N2P4 - 1] * (X2P4[N2P4 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[10]):
                object.PEQIN[10][I] = object.PEQEL[1][I - IOFFN[10]]

        # P states, 2P3
        if EN > EIN[11]:
            if EN <= X2P3[N2P3 - 1]:
                for j in range(1, N2P3):
                    if EN <= X2P3[j]:
                        break
                A = (Y2P3[j] - Y2P3[j - 1]) / (X2P3[j] - X2P3[j - 1])
                B = (X2P3[j - 1] * Y2P3[j] - X2P3[j] * Y2P3[j - 1]) / (X2P3[j - 1] - X2P3[j])
                object.QIN[11][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[11][I] = Y2P3[N2P3 - 1] * (X2P3[N2P3 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[11]):
                object.PEQIN[11][I] = object.PEQEL[1][I - IOFFN[11]]

        # P states, 2P2
        if EN > EIN[12]:
            if EN <= X2P2[N2P2 - 1]:
                for j in range(1, N2P2):
                    if EN <= X2P2[j]:
                        break
                A = (Y2P2[j] - Y2P2[j - 1]) / (X2P2[j] - X2P2[j - 1])
                B = (X2P2[j - 1] * Y2P2[j] - X2P2[j] * Y2P2[j - 1]) / (X2P2[j - 1] - X2P2[j])
                object.QIN[12][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[12][I] = Y2P2[N2P2 - 1] * (X2P2[N2P2 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[12]):
                object.PEQIN[12][I] = object.PEQEL[1][I - IOFFN[12]]

        # P states, 2P1
        if EN > EIN[13]:
            if EN <= X2P1[N2P1 - 1]:
                for j in range(1, N2P1):
                    if EN <= X2P1[j]:
                        break
                A = (Y2P1[j] - Y2P1[j - 1]) / (X2P1[j] - X2P1[j - 1])
                B = (X2P1[j - 1] * Y2P1[j] - X2P1[j] * Y2P1[j - 1]) / (X2P1[j - 1] - X2P1[j])
                object.QIN[13][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.QIN[13][I] = Y2P1[N2P1 - 1] * (X2P1[N2P1 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * EIN[13]):
                object.PEQIN[13][I] = object.PEQEL[1][I - IOFFN[13]]

        # D states, 3D6
        if EN > EIN[14]:
            if EN <= X3D6[N3D6 - 1]:
                for j in range(1, N3D6):
                    if EN <= X3D6[j]:
                        break
                A = (Y3D6[j] - Y3D6[j - 1]) / (X3D6[j] - X3D6[j - 1])
                B = (X3D6[j - 1] * Y3D6[j] - X3D6[j] * Y3D6[j - 1]) / (X3D6[j - 1] - X3D6[j])
                object.QIN[14][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[14][I] = Y3D6[N3D6 - 1] * (X3D6[N3D6 - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * EIN[14]):
                object.PEQIN[14][I] = object.PEQEL[1][I - IOFFN[14]]

        # D states, 3D5
        if EN > EIN[15]:
            if EN <= X3D5[N3D5 - 1]:
                for j in range(1, N3D5):
                    if EN <= X3D5[j]:
                        break
                A = (Y3D5[j] - Y3D5[j - 1]) / (X3D5[j] - X3D5[j - 1])
                B = (X3D5[j - 1] * Y3D5[j] - X3D5[j] * Y3D5[j - 1]) / (X3D5[j - 1] - X3D5[j])
                object.QIN[15][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[15][I] = 0.0010 / (EIN[15] * BETA2) * (
                        np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[15])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (
                                            EN + object.E[2] + EIN[15])
            if EN > (2.0 * EIN[15]):
                object.PEQIN[15][I] = object.PEQEL[1][I - IOFFN[15]]

        # D states, 3D3
        if EN > EIN[16]:
            if EN <= X3D3[N3D3 - 1]:
                for j in range(1, N3D3):
                    if EN <= X3D3[j]:
                        break
                A = (Y3D3[j] - Y3D3[j - 1]) / (X3D3[j] - X3D3[j - 1])
                B = (X3D3[j - 1] * Y3D3[j] - X3D3[j] * Y3D3[j - 1]) / (X3D3[j - 1] - X3D3[j])
                object.QIN[16][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[16][I] = Y3D3[N3D3 - 1] * (X3D3[N3D3 - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * EIN[16]):
                object.PEQIN[16][I] = object.PEQEL[1][I - IOFFN[16]]

        # D states, 3D4'
        if EN > EIN[17]:
            if EN <= X3D4P[N3D4P - 1]:
                for j in range(1, N3D4P):
                    if EN <= X3D4P[j]:
                        break
                A = (Y3D4P[j] - Y3D4P[j - 1]) / (X3D4P[j] - X3D4P[j - 1])
                B = (X3D4P[j - 1] * Y3D4P[j] - X3D4P[j] * Y3D4P[j - 1]) / (X3D4P[j - 1] - X3D4P[j])
                object.QIN[17][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[17][I] = Y3D4P[N3D4P - 1] * (X3D4P[N3D4P - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * EIN[17]):
                object.PEQIN[17][I] = object.PEQEL[1][I - IOFFN[17]]

        # D states, 3D4
        if EN > EIN[18]:
            if EN <= X3D4[N3D4 - 1]:
                for j in range(1, N3D4):
                    if EN <= X3D4[j]:
                        break
                A = (Y3D4[j] - Y3D4[j - 1]) / (X3D4[j] - X3D4[j - 1])
                B = (X3D4[j - 1] * Y3D4[j] - X3D4[j] * Y3D4[j - 1]) / (X3D4[j - 1] - X3D4[j])
                object.QIN[18][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[18][I] = Y3D4[N3D4 - 1] * (X3D4[N3D4 - 1] / EN) ** 2 * 1.0e-18
            if EN > (2.0 * EIN[18]):
                object.PEQIN[18][I] = object.PEQEL[1][I - IOFFN[18]]

        # D states, 3D1''
        if EN > EIN[19]:
            if EN <= X3D1PP[N3D1PP - 1]:
                for j in range(1, N3D1PP):
                    if EN <= X3D1PP[j]:
                        break
                A = (Y3D1PP[j] - Y3D1PP[j - 1]) / (X3D1PP[j] - X3D1PP[j - 1])
                B = (X3D1PP[j - 1] * Y3D1PP[j] - X3D1PP[j] * Y3D1PP[j - 1]) / (X3D1PP[j - 1] - X3D1PP[j])
                object.QIN[19][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[19][I] = Y3D1PP[N3D1PP - 1] * (X3D1PP[N3D1PP - 1] / EN) ** 2 * 1.0e-18
            if EN > (2.0 * EIN[19]):
                object.PEQIN[19][I] = object.PEQEL[1][I - IOFFN[19]]

        # S states, 2S5
        if EN > EIN[20]:
            if EN <= X2S5[N2S5 - 1]:
                for j in range(1, N2S5):
                    if EN <= X2S5[j]:
                        break
                A = (Y2S5[j] - Y2S5[j - 1]) / (X2S5[j] - X2S5[j - 1])
                B = (X2S5[j - 1] * Y2S5[j] - X2S5[j] * Y2S5[j - 1]) / (X2S5[j - 1] - X2S5[j])
                object.QIN[20][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[20][I] = Y2S5[N2S5 - 1] * (X2S5[N2S5 - 1] / EN) ** 2 * 1.0e-18
            if EN > (2.0 * EIN[20]):
                object.PEQIN[20][I] = object.PEQEL[1][I - IOFFN[20]]

        # S states, 2S4 F=0.0257
        if EN > EIN[21]:
            object.QIN[21][I] = 0.0257 / (EIN[21] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[21])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[21])
            if object.QIN[21][I] < 0:
                object.QIN[21][I] = 0.0
            if EN > (2.0 * EIN[21]):
                object.PEQIN[21][I] = object.PEQEL[1][I - IOFFN[21]]

        # D states, 3D1'
        if EN > EIN[22]:
            if EN <= X3D1P[N3D1P - 1]:
                for j in range(1, N3D1P):
                    if EN <= X3D1P[j]:
                        break
                A = (Y3D1P[j] - Y3D1P[j - 1]) / (X3D1P[j] - X3D1P[j - 1])
                B = (X3D1P[j - 1] * Y3D1P[j] - X3D1P[j] * Y3D1P[j - 1]) / (X3D1P[j - 1] - X3D1P[j])
                object.QIN[22][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[22][I] = Y3D1P[N3D1P - 1] * (X3D1P[N3D1P - 1] / EN) ** 2 * 1.0e-18
            if EN > (2.0 * EIN[22]):
                object.PEQIN[22][I] = object.PEQEL[1][I - IOFFN[22]]

        # D states, 3D2 F=0.074
        if EN > EIN[23]:
            object.QIN[23][I] = 0.074 / (EIN[23] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[23])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[23])
            if object.QIN[23][I] < 0:
                object.QIN[23][I] = 0.0
            if EN > (2.0 * EIN[23]):
                object.PEQIN[23][I] = object.PEQEL[1][I - IOFFN[23]]

        # S states, 3S1''''
        if EN > EIN[24]:
            if EN <= X3S1PPPP[N3S1PPPP - 1]:
                for j in range(1, N3S1PPPP):
                    if EN <= X3S1PPPP[j]:
                        break
                A = (Y3S1PPPP[j] - Y3S1PPPP[j - 1]) / (X3S1PPPP[j] - X3S1PPPP[j - 1])
                B = (X3S1PPPP[j - 1] * Y3S1PPPP[j] - X3S1PPPP[j] * Y3S1PPPP[j - 1]) / (X3S1PPPP[j - 1] - X3S1PPPP[j])
                object.QIN[24][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[24][I] = Y3S1PPPP[N3S1PPPP - 1] * (X3S1PPPP[N3S1PPPP - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * EIN[24]):
                object.PEQIN[24][I] = object.PEQEL[1][I - IOFFN[24]]

        # S states, 3S1''
        if EN > EIN[25]:
            if EN <= X3S1PP[N3S1PP - 1]:
                for j in range(1, N3S1PP):
                    if EN <= X3S1PP[j]:
                        break
                A = (Y3S1PP[j] - Y3S1PP[j - 1]) / (X3S1PP[j] - X3S1PP[j - 1])
                B = (X3S1PP[j - 1] * Y3S1PP[j] - X3S1PP[j] * Y3S1PP[j - 1]) / (X3S1PP[j - 1] - X3S1PP[j])
                object.QIN[25][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[25][I] = Y3S1PP[N3S1PP - 1] * (X3S1PP[N3S1PP - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * EIN[25]):
                object.PEQIN[25][I] = object.PEQEL[1][I - IOFFN[25]]

        # S states, 3S'''
        if EN > EIN[26]:
            if EN <= X3S1PPP[N3S1PPP - 1]:
                for j in range(1, N3S1PPP):
                    if EN <= X3S1PPP[j]:
                        break
                A = (Y3S1PPP[j] - Y3S1PPP[j - 1]) / (X3S1PPP[j] - X3S1PPP[j - 1])
                B = (X3S1PPP[j - 1] * Y3S1PPP[j] - X3S1PPP[j] * Y3S1PPP[j - 1]) / (X3S1PPP[j - 1] - X3S1PPP[j])
                object.QIN[26][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[26][I] = Y3S1PPP[N3S1PPP - 1] * (X3S1PPP[N3S1PPP - 1] / EN) * 1.0e-18
            if EN > (2.0 * EIN[26]):
                object.PEQIN[26][I] = object.PEQEL[1][I - IOFFN[26]]

        # S states, 2S3
        if EN > EIN[27]:
            if EN <= X2S3[N2S3 - 1]:
                for j in range(1, N2S3):
                    if EN <= X2S3[j]:
                        break
                A = (Y2S3[j] - Y2S3[j - 1]) / (X2S3[j] - X2S3[j - 1])
                B = (X2S3[j - 1] * Y2S3[j] - X2S3[j] * Y2S3[j - 1]) / (X2S3[j - 1] - X2S3[j])
                object.QIN[27][I] = (A * EN + B) * 1.0e-18
            else:
                object.QIN[27][I] = Y2S3[N2S3 - 1] * (X2S3[N2S3 - 1] / EN) ** 2 * 1.0e-18
            if EN > (2.0 * EIN[27]):
                object.PEQIN[27][I] = object.PEQEL[1][I - IOFFN[27]]

        # S states, 2S2 F=0.011
        if EN > EIN[28]:
            object.QIN[28][I] = 0.011 / (EIN[28] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[28])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[28])
            if object.QIN[28][I] < 0:
                object.QIN[28][I] = 0.0
            if EN > (2.0 * EIN[28]):
                object.PEQIN[28][I] = object.PEQEL[1][I - IOFFN[28]]

        # S states, 3S1' F=0.092
        if EN > EIN[29]:
            object.QIN[29][I] = 0.092 / (EIN[29] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[29])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[29])
            if object.QIN[29][I] < 0:
                object.QIN[29][I] = 0.0
            if EN > (2.0 * EIN[29]):
                object.PEQIN[29][I] = object.PEQEL[1][I - IOFFN[29]]

        # D states, 4D5 F=0.019
        if EN > EIN[30]:
            object.QIN[30][I] = 0.019 / (EIN[30] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[30])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[30])
            if object.QIN[30][I] < 0:
                object.QIN[30][I] = 0.0
            if EN > (2.0 * EIN[30]):
                object.PEQIN[30][I] = object.PEQEL[1][I - IOFFN[30]]

        # S states, 3S4 F=0.0144
        if EN > EIN[31]:
            object.QIN[31][I] = 0.0144 / (EIN[31] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[31])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[31])
            if object.QIN[31][I] < 0:
                object.QIN[31][I] = 0.0
            if EN > (2.0 * EIN[31]):
                object.PEQIN[31][I] = object.PEQEL[1][I - IOFFN[31]]

        # D states, 4D2 F=0.0484
        if EN > EIN[32]:
            object.QIN[32][I] = 0.0484 / (EIN[32] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[32])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[32])
            if object.QIN[32][I] < 0:
                object.QIN[32][I] = 0.0
            if EN > (2.0 * EIN[32]):
                object.PEQIN[32][I] = object.PEQEL[1][I - IOFFN[32]]

        # S states, 4S1' F=0.0209
        if EN > EIN[33]:
            object.QIN[33][I] = 0.0209 / (EIN[33] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[33])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[33])
            if object.QIN[33][I] < 0:
                object.QIN[33][I] = 0.0
            if EN > (2.0 * EIN[33]):
                object.PEQIN[33][I] = object.PEQEL[1][I - IOFFN[33]]

        # S states, 3S2 F=0.022
        if EN > EIN[34]:
            object.QIN[34][I] = 0.022 / (EIN[34] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[34])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[34])
            if object.QIN[34][I] < 0:
                object.QIN[34][I] = 0.0
            if EN > (2.0 * EIN[34]):
                object.PEQIN[34][I] = object.PEQEL[1][I - IOFFN[34]]

        # D states, 5D5 F=0.0041
        if EN > EIN[35]:
            object.QIN[35][I] = 0.0041 / (EIN[35] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[35])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[35])
            if object.QIN[35][I] < 0:
                object.QIN[35][I] = 0.0
            if EN > (2.0 * EIN[35]):
                object.PEQIN[35][I] = object.PEQEL[1][I - IOFFN[35]]

        # S states, 4S4 F=0.0426
        if EN > EIN[36]:
            object.QIN[36][I] = 0.0426 / (EIN[36] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[36])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[36])
            if object.QIN[36][I] < 0:
                object.QIN[36][I] = 0.0
            if EN > (2.0 * EIN[36]):
                object.PEQIN[36][I] = object.PEQEL[1][I - IOFFN[36]]

        # D states, 5D2 F=0.0426
        if EN > EIN[37]:
            object.QIN[37][I] = 0.0426 / (EIN[37] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[37])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[37])
            if object.QIN[37][I] < 0:
                object.QIN[37][I] = 0.0
            if EN > (2.0 * EIN[37]):
                object.PEQIN[37][I] = object.PEQEL[1][I - IOFFN[37]]

        # D states, 6D5 F=0.00075
        if EN > EIN[38]:
            object.QIN[38][I] = 0.00075 / (EIN[38] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[38])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[38])
            if object.QIN[38][I] < 0:
                object.QIN[38][I] = 0.0
            if EN > (2.0 * EIN[38]):
                object.PEQIN[38][I] = object.PEQEL[1][I - IOFFN[38]]

        # S states, 5S1' F=0.00051
        if EN > EIN[39]:
            object.QIN[39][I] = 0.00051 / (EIN[39] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[39])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[39])
            if object.QIN[39][I] < 0:
                object.QIN[39][I] = 0.0
            if EN > (2.0 * EIN[39]):
                object.PEQIN[39][I] = object.PEQEL[1][I - IOFFN[39]]

        # S states, 4S2 F=0.00074
        if EN > EIN[40]:
            object.QIN[40][I] = 0.00074 / (EIN[40] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[40])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[40])
            if object.QIN[40][I] < 0:
                object.QIN[40][I] = 0.0
            if EN > (2.0 * EIN[40]):
                object.PEQIN[40][I] = object.PEQEL[1][I - IOFFN[40]]

        # S states, 5S4 F=0.013
        if EN > EIN[41]:
            object.QIN[41][I] = 0.013 / (EIN[41] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[41])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[41])
            if object.QIN[41][I] < 0:
                object.QIN[41][I] = 0.0
            if EN > (2.0 * EIN[41]):
                object.PEQIN[41][I] = object.PEQEL[1][I - IOFFN[41]]

        # S states, 6D2 F=0.029
        if EN > EIN[42]:
            object.QIN[42][I] = 0.029 / (EIN[42] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[42])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[42])
            if object.QIN[42][I] < 0:
                object.QIN[42][I] = 0.0
            if EN > (2.0 * EIN[42]):
                object.PEQIN[42][I] = object.PEQEL[1][I - IOFFN[42]]

        # sum higher j=1 states f=0.1315
        if EN > EIN[43]:
            object.QIN[43][I] = 0.1315 / (EIN[43] * BETA2) * (
                    np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 * EIN[43])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + EIN[43])
            if object.QIN[43][I] < 0:
                object.QIN[43][I] = 0.0
            if EN > (2.0 * EIN[43]):
                object.PEQIN[43][I] = object.PEQEL[1][I - IOFFN[43]]

        Q1SSUM = object.QIN[0][I] + object.QIN[1][I] + object.QIN[2][I] + object.QIN[3][I]
        QPSSUM = 0
        QDSSUM = 0
        for i in range(14, 44):
            QDSSUM += object.QIN[i][I]
        for i in range(4, 14):
            QPSSUM += object.QIN[i][I]
        TOTSUM = Q1SSUM + QPSSUM + QDSSUM

        object.Q[1][I] = QELA + Q1SSUM + QPSSUM + QDSSUM
        for i in range(0, 7):
            object.Q[1][I] += object.QION[i][I]

    for J in range(0, object.NIN):
        if object.EFINAL <= EIN[J]:
            object.NIN = J 
            break
    return object
    gd.close()
    print("AR")
