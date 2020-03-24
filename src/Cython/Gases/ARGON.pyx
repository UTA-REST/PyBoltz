from libc.math cimport sin, cos, acos,asin, log,sqrt,exp,atan
import math
import sys
from Gas cimport Gas
import numpy as np
cimport numpy as np

sys.path.append('../hdf5_python')
import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void Gas2(Gas *object):
    '''
    This function is used to calculate the needed momentum cross sections for Argon gas.
    '''
    gd = np.load('gases.npy').item()
    cdef double APOL, AA, DD, FF, A1, ElectronMass2, API, A0, RY, BBCONST, CONST, AM2, C, PSCALE, AUGL3, AUGL2, AUGL1, AUGK
    cdef int N_Ionization, N_Attachment, N_Inelastic, N_Null, NBREM, NDATA, NEPSI, NIDATA, N_Ionization2, N_Ionization3, NKSH, NL1S, NL2S, NL3S, N1S5, NIS4, NIS3, NIS2,N1S4=79,N1S3=70,N1S2=70
    cdef int N2P10, N2P9, N2P8, N2P7, N2P6, N2P5, N2P4, N2P3, N2P2, N2P1, N3D6, N3D5, N3D3, N3D4P, N3D4, N3D1PP, N2S5, N3D1P
    cdef int N3S1PPPP, N3S1PP, N3S1PPP, N2S3
    APOL = <float>(11.08)
    LMAX = 100
    AA = <float>(-1.459)
    DD = <float>(68.93)
    FF = <float>(-97.0)
    A1 = <float>(8.69)
    ElectronMass2 = <float>(1021997.804)
    API = acos(-1)
    A0 = 0.52917720859e-8
    RY = <float>(13.60569193)
    BBCONST = 16.0 * API * A0 * A0 * RY * RY / ElectronMass2
    CONST = 1.873884e-20
    AM2 = <float>(3.593)
    C = <float>(39.70)
    PSCALE = <float>(0.9)
    AUGL3 = 2.0
    AUGL2 = 1.0
    AUGL1 = <float>(2.63)
    AUGK = <float>(3.39)
    object.N_Ionization = 7
    object.N_Attachment = 1
    object.N_Inelastic = 44
    object.N_Null = 0
    NBREM = 25
    cdef int i = 0, j = 0

    for i in range(0, 6):
        object.AngularModel[i] = object.WhichAngularModel
    for i in range(0, object.N_Inelastic):
        object.KIN[i] = object.WhichAngularModel
    NDATA = 117
    NEPSI = 217
    NIDATA = 75
    N_Ionization2 = 47
    N_Ionization3 = 36
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
    ElectronMass = 9.10938291e-31
    AMU = 1.660538921e-27
    object.E = [0.0, 1.0, <float>(15.75961), 0.0, 0.0, <float>(15.0)]
    object.E[1] = <float>(2.0) * ElectronMass / (<float>(39.948) * AMU)
    cdef double ISHELL[30], LEGAS[30], WKLM[30]
    object.EOBY[0:7] = [<float>(9.5), <float>(18.0), <float>(34.0), <float>(110.0), <float>(110.0), <float>(150.0), <float>(1800)]

    object.IonizationEnergy[0:7] = [<float>(15.75961), <float>(43.38928), <float>(84.124), <float>(248.4), <float>(250.6), <float>(326.3), <float>(3205.9)]
    LEGAS[0:7] = [0, 0, 0, 1, 1, 1, 1]
    ISHELL[0:7] = [0, 0, 0, 4, 3, 2, 1]
    object.NC0[0:7] = [0, 1, 2, 2, 2, 3, 4]
    object.EC0[0:7] = [0.0, 6.0, 12.0, 210.5, 202.2, 240.8, 3071]
    WKLM[0:7] = [0.0, 0.0, 0.0, <float>(0.00147), <float>(0.00147), <float>(0.00147), <float>(0.12)]
    object.WK=WKLM
    object.EFL[0:7] = [0.0, 0.0, 0.0, 232, 235, 310, 2957]
    object.NG1[0:7] = [0, 0, 0, 1, 1, 2, 3]
    object.EG1[0:7] = [0.0, 0.0, 0.0, 210.5, 202.2, 240.8, 2850]
    object.NG2[0:7] = [0.0, 0.0, 0.0, 1, 1, 1, 2]
    #TODO: typo in magboltz
    object.EG2[0:7] = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 220]

    cdef int IOFFION[10]
    IOFFION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cdef int IOFFN[44]

    for i in range(44):
        IOFFN[i] = 0

    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if object.EG[i] > object.IonizationEnergy[j]:
                IOFFION[j] = i
                break
    cdef double XEN[117],YSEC[117],YEL[117],XEPS[217],YEPS[217],XENI[75],YENI[75],YENC[75],YEN1[75],XEN2[47]
    cdef double YEN2[47],XEN3[36],YEN3[36],XKSH[89],YKSH[89],XL1S[101],YL1S[101],XL2S[104],YL2S[104],XL3S[104],YL3S[104]
    cdef double X1S5[71],Y1S5[71],YEPS1[71],X1S4[79],Y1S4[79],YEPS2[79],X1S3[70],Y1S3[70],YEPS3[70],X1S2[70],Y1S2[70]
    cdef double YEPS4[70],X2P10[54],Y2P10[54],YEP2P10[54],X2P9[17],Y2P9[17],YEP2P9[17],X2P8[15],Y2P8[15],YEP2P8[15]
    cdef double X2P7[17],Y2P7[17],YEP2P7[17],X2P6[16],Y2P6[16],YEP2P6[16],X2P5[17],Y2P5[17],YEP2P5[17],X2P4[17],Y2P4[17]
    cdef double YEP2P4[17],X2P3[17],Y2P3[17],YEP2P3[17],X2P2[16],Y2P2[16],YEP2P2[16],X2P1[17],Y2P1[17],YEP2P1[17],X3D6[19]
    cdef double Y3D6[19],YEP3D6[19],X3D5[26],Y3D5[26],YEP3D5[26],X3D4P[20],Y3D4P[20],YEP3D4P[20],X3D4[23],Y3D4[23]
    cdef double YEP3D4[23],X3D3[20],Y3D3[20],YEP3D3[20],X3D1PP[19],Y3D1PP[19],YEP3D1PP[19],X3D1P[16],Y3D1P[16],YEP3D1P[16],X3S1PPPP[21]
    cdef double Y3S1PPPP[21],YEP3S1PPPP[21],X3S1PPP[16],Y3S1PPP[16],YEP3S1PPP[16],X3S1PP[21],Y3S1PP[21],YEP3S1PP[21],X2S5[19],Y2S5[19],YEP2S5[19],X2S3[19],Y2S3[19],YEP2S3[19]
    object.EnergyLevels = gd['gas2/EnergyLevels']
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
    for i in range(object.N_Inelastic):
        object.PenningFraction[0][i] = 0.2
        object.PenningFraction[1][i] = 1.0
        object.PenningFraction[2][i] = 1.0
    for i in range(object.N_Inelastic):
        for j in range(4000):
            if object.EG[j] > object.EnergyLevels[i]:
                IOFFN[i] = j
                break
    cdef int I
    cdef double GAMMA1, GAMMA2, BETA, BETA2, ElasticCrossSectionA, QMOM, AK, AK2, AK3, AK4, AN0, AN1, AN2, ANHIGH, Sum, SIFEL, ANLOW, PQ[3], QCORR, QTEMP
    cdef double QPSSum,QDSSum,TOTSum,Q1SSum,PQ1,PQ2,PQ3
    for I in range(4000):
        EN = object.EG[I]

        # EN=EN+object.EnergyStep
        if EN > object.EnergyLevels[0]:
            GAMMA1 = (ElectronMass2 + 2.0 * EN) / ElectronMass2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = sqrt(1.00 - 1.00 / GAMMA2)
            BETA2 = BETA * BETA
        if EN <= 1:
            if EN == 0:
                ElasticCrossSectionA = 7.491e-16
                QMOM = 7.491e-16
            if EN != 0:
                AK = sqrt(EN / object.RhydbergConst)
                AK2 = AK * AK
                AK3 = AK2 * AK
                AK4 = AK3 * AK
                AN0 = -AA * AK * (1.0 + (4.0 * APOL / 3.0) * AK2 * log(AK)) - (
                        API * APOL / 3.0) * AK2 + DD * AK3 + FF * AK4
                AN1 = (API / 15.0) * APOL * AK2 - A1 * AK3
                AN2 = API * APOL * AK2 / 105.0
                AN0 = atan(AN0)
                AN1 = atan(AN1)
                AN2 = atan(AN2)
                ANHIGH = AN2
                Sum = (sin(AN0 - AN1)) ** 2
                Sum = Sum + 2.0 * (sin(AN1 - AN2)) ** 2
                SIGEL = (sin(AN0)) ** 2 + 3.0 * (sin(AN1)) ** 2
                for j in range(2, LMAX):
                    ANLOW = ANHIGH
                    SumI = 6.0 / ((2.0 * j + 5.0) * (2.0 * j + 3.0) * (2.0 * j + 1.0) * (2.0 * j - 1.0))
                    Sum = Sum + (j + 1.0) * (sin(atan(API * APOL * AK2 * SumI))) ** 2
                    ANHIGH = atan(API * APOL * AK2 / ((2.0 * j + 5.0) * (2.0 * j + 3.0) * (2.0 * j + 1.0)))
                    SIGEL = SIGEL + (2.0 * j + 1.0) * (sin(ANLOW)) ** 2
                ElasticCrossSectionA = SIGEL * 4.0 * object.PIR2 / AK2
                QMOM = Sum * 4.0 * object.PIR2 / AK2
        if EN >1:
            for j in range(1, NDATA):
                if EN < XEN[j]:
                    break

            A = (YEL[j] - YEL[j - 1]) / (XEN[j] - XEN[j - 1])
            B = (XEN[j - 1] * YEL[j] - XEN[j] * YEL[j - 1]) / (XEN[j - 1] - XEN[j])
            ElasticCrossSectionA = (A * EN + B) * 1.0e-16
            A = (YSEC[j] - YSEC[j - 1]) / (XEN[j] - XEN[j - 1])
            B = (XEN[j - 1] * YSEC[j] - XEN[j] * YSEC[j - 1]) / (XEN[j - 1] - XEN[j])
            QMOM = (A * EN + B) * 1.0e-16

        PQ1 = 0.5 + (ElasticCrossSectionA - QMOM) / ElasticCrossSectionA
        for j in range(1, NEPSI):
            if EN < XEPS[j]:
                break
        A = (YEPS[j] - YEPS[j - 1]) / (XEPS[j] - XEPS[j - 1])
        B = (XEPS[j - 1] * YEPS[j] - XEPS[j] * YEPS[j - 1]) / (XEPS[j - 1] - XEPS[j])
        PQ2 = A * EN + B

        # EPSILON = 1 - PQ2
        PQ2 = 1.0 - PQ2
        PQ = [0.5, PQ1, PQ2]

        object.PEElasticCrossSection[1][I] = PQ[object.WhichAngularModel]
        object.Q[1][I] = ElasticCrossSectionA
        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM
        object.IonizationCrossSection[0][I] = 0.0
        object.PEIonizationCrossSection[0][I] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[0][I] = 0
        if EN > object.IonizationEnergy[0]:
            if EN <= XENI[NIDATA - 1]:
                j = 0
                for j in range(1, NIDATA):
                    if EN <= XENI[j]:
                        break
                A = (YEN1[j] - YEN1[j - 1]) / (XENI[j] - XENI[j - 1])
                B = (XENI[j - 1] * YEN1[j] - XENI[j] * YEN1[j - 1]) / (XENI[j - 1] - XENI[j])
                object.IonizationCrossSection[0][I] = (A * EN + B) * 1e-16

            else:
                # USE BORN BETHE X-SECTION ABOVE XENI[NIDATA] EV
                X2 = 1 / BETA2
                X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                object.IonizationCrossSection[0][I] = CONST * (AM2 * (X1 - object.DEN[I] / 2) + C * X2) * <float>(0.9466)
            if EN > 2 * object.IonizationEnergy[0]:
                object.PEIonizationCrossSection[0][I] = object.PEElasticCrossSection[1][(I - IOFFION[0])]

        object.IonizationCrossSection[1][I] = 0.0
        object.PEIonizationCrossSection[1][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[1][I] = 0
        if EN > object.IonizationEnergy[1]:
            if EN <= XEN2[N_Ionization2 - 1]:
                j = 0
                for j in range(1, N_Ionization2):
                    if EN <= XEN2[j]:
                        break
                A = (YEN2[j] - YEN2[j - 1]) / (XEN2[j] - XEN2[j - 1])
                B = (XEN2[j - 1] * YEN2[j] - XEN2[j] * YEN2[j - 1]) / (XEN2[j - 1] - XEN2[j])
                object.IonizationCrossSection[1][I] = (A * EN + B) * 1e-16
                if object.IonizationCrossSection[1][I] < 0:
                    object.IonizationCrossSection[1][I] = 0
            else:
                # USE BORN BETHE X-SECTION ABOVE XEN2[N_Ionization2] EV
                X2 = 1.0 / BETA2
                X1 = X2 * log(BETA2 / (1.0 - BETA2)) - 1.0
                object.IonizationCrossSection[1][I] = CONST * (AM2 * (X1 - object.DEN[I] / 2.0) + C * X2) * <float>(0.04448)
            if EN > 2 * object.IonizationEnergy[1]:
                object.PEIonizationCrossSection[1][I] = object.PEElasticCrossSection[1][(I - IOFFION[1])]

        object.IonizationCrossSection[2][I] = 0.0
        object.PEIonizationCrossSection[2][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[2][I] = 0
        if EN > object.IonizationEnergy[2]:
            if EN <= XEN3[N_Ionization3 - 1]:
                j = 0
                for j in range(1, N_Ionization3):
                    if EN <= XEN3[j]:
                        break
                A = (YEN3[j] - YEN3[j - 1]) / (XEN3[j] - XEN3[j - 1])
                B = (XEN3[j - 1] * YEN3[j] - XEN3[j] * YEN3[j - 1]) / (XEN3[j - 1] - XEN3[j])
                object.IonizationCrossSection[2][I] = (A * EN + B) * 1e-16
                if object.IonizationCrossSection[2][I] < 0:
                    object.IonizationCrossSection[2][I] = 0
            else:
                # USE BORN BETHE X-SECTION ABOVE XEN3[N_Ionization3] EV
                X2 = 1 / BETA2
                X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                object.IonizationCrossSection[2][I] = CONST * (AM2 * (X1 - object.DEN[I] / 2.0) + C * X2) * <float>(0.00987)
            if EN > 2 * object.IonizationEnergy[2]:
                object.PEIonizationCrossSection[2][I] = object.PEElasticCrossSection[1][(I - IOFFION[2])]
        # L3 Shell ionisation
        object.IonizationCrossSection[3][I] = 0.0
        object.PEIonizationCrossSection[3][I] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[3][I] = 0.0

        if EN > object.IonizationEnergy[3]:
            for j in range(1, NL3S):
                if EN <= XL3S[j]:
                    break
            A = (YL3S[j] - YL3S[j - 1]) / (XL3S[j] - XL3S[j - 1])
            B = (XL3S[j - 1] * YL3S[j] - XL3S[j] * YL3S[j - 1]) / (XL3S[j - 1] - XL3S[j])
            object.IonizationCrossSection[3][I] = (A * EN + B) * 1e-16
            object.PEIonizationCrossSection[3][I] = object.PEElasticCrossSection[1][I - IOFFION[3]]

        # L2 Shell ionisation
        object.IonizationCrossSection[4][I] = 0.0
        object.PEIonizationCrossSection[4][I] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[4][I] = 0.0

        if EN > object.IonizationEnergy[4]:
            for j in range(1, NL2S):
                if EN <= XL2S[j]:
                    break
            A = (YL2S[j] - YL2S[j - 1]) / (XL2S[j] - XL2S[j - 1])
            B = (XL2S[j - 1] * YL2S[j] - XL2S[j] * YL2S[j - 1]) / (XL2S[j - 1] - XL2S[j])
            object.IonizationCrossSection[4][I] = (A * EN + B) * 1e-16
            object.PEIonizationCrossSection[4][I] = object.PEElasticCrossSection[1][I - IOFFION[4]]
        # L1 Shell ionisation
        object.IonizationCrossSection[5][I] = 0.0
        object.PEIonizationCrossSection[5][I] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[5][I] = 0.0

        if EN > object.IonizationEnergy[5]:
            for j in range(1, NL1S):
                if EN <= XL1S[j]:
                    break
            A = (YL1S[j] - YL1S[j - 1]) / (XL1S[j] - XL1S[j - 1])
            B = (XL1S[j - 1] * YL1S[j] - XL1S[j] * YL1S[j - 1]) / (XL1S[j - 1] - XL1S[j])
            object.IonizationCrossSection[5][I] = (A * EN + B) * 1e-16
            object.PEIonizationCrossSection[5][I] = object.PEElasticCrossSection[1][I - IOFFION[5]]

        # K Shell ionisation
        object.IonizationCrossSection[6][I] = 0.0
        object.PEIonizationCrossSection[6][I] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[6][I] = 0.0

        if EN > object.IonizationEnergy[6]:
            for j in range(1, NKSH):
                if EN <= XKSH[j]:
                    break
            A = (YKSH[j] - YKSH[j - 1]) / (XKSH[j] - XKSH[j - 1])
            B = (XKSH[j - 1] * YKSH[j] - XKSH[j] * YKSH[j - 1]) / (XKSH[j - 1] - XKSH[j])
            object.IonizationCrossSection[6][I] = (A * EN + B) * 1e-16
            object.PEIonizationCrossSection[6][I] = object.PEElasticCrossSection[1][I - IOFFION[6]]
        # ATTAchment
        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = 0.0

        # Counting ionisation
        object.Q[4][I] = 0.0
        object.PEElasticCrossSection[4][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEElasticCrossSection[4][I] = 0.0
        if EN > object.E[2]:
            if EN <= XENI[NIDATA-1]:
                for j in range(1, NIDATA):
                    if EN <= XENI[j]:
                        break
                A = (YENC[j] - YENC[j - 1]) / (XENI[j] - XENI[j - 1])
                B = (XENI[j - 1] * YENC[j] - XENI[j] * YENC[j - 1]) / (XENI[j - 1] - XENI[j])
                object.Q[4][I] = (A * EN + B) * 1.0e-16
            else:
                object.Q[4][I] = CONST * (AM2 * (X1 - object.DEN[I] / 2.0) + C * X2)
        QTEMP = object.IonizationCrossSection[3][I] + object.IonizationCrossSection[4][I] + object.IonizationCrossSection[5][I] + object.IonizationCrossSection[6][I]
        if object.Q[4][I] == 0.0:
            QCORR = 1.0
        else:
            QCORR = (object.Q[4][I] - QTEMP) / object.Q[4][I]
        object.IonizationCrossSection[0][I] = object.IonizationCrossSection[0][I] * QCORR
        object.IonizationCrossSection[1][I] = object.IonizationCrossSection[1][I] * QCORR
        object.IonizationCrossSection[2][I] = object.IonizationCrossSection[2][I] * QCORR

        object.Q[5][I] = 0.0
        for NL in range(object.N_Inelastic):
            object.InelasticCrossSectionPerGas[NL][I] = 0.0
            object.PEInelasticCrossSectionPerGas[NL][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[NL][I] = 0.0

        # 1S5

        if EN > object.EnergyLevels[0]:
            if EN <= X1S5[N1S5 - 1]:
                for j in range(1, N1S5):
                    if EN <= X1S5[j]:
                        break
                A = (Y1S5[j] - Y1S5[j - 1]) / (X1S5[j] - X1S5[j - 1])
                B = (X1S5[j - 1] * Y1S5[j] - X1S5[j] * Y1S5[j - 1]) / (X1S5[j - 1] - X1S5[j])
                object.InelasticCrossSectionPerGas[0][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[0][I] = Y1S5[N1S5 - 1] * (X1S5[N1S5 - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[0]):
                object.PEInelasticCrossSectionPerGas[0][I] = object.PEElasticCrossSection[1][I - IOFFN[0]]

        if EN > object.EnergyLevels[1]:
            if EN <= X1S4[N1S4 - 1]:
                for j in range(1, N1S4):
                    if EN <= X1S4[j]:
                        break
                A = (Y1S4[j] - Y1S4[j - 1]) / (X1S4[j] - X1S4[j - 1])
                B = (X1S4[j - 1] * Y1S4[j] - X1S4[j] * Y1S4[j - 1]) / (X1S4[j - 1] - X1S4[j])
                object.InelasticCrossSectionPerGas[1][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[1][I] = <float>(0.0580) / (object.EnergyLevels[1] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[1])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (
                                           EN + object.E[2] + object.EnergyLevels[1])
            if EN > (2.0 * object.EnergyLevels[1]):
                object.PEInelasticCrossSectionPerGas[1][I] = object.PEElasticCrossSection[1][I - IOFFN[1]]

        # 1S3
        if EN > object.EnergyLevels[2]:
            if EN <= X1S3[N1S3 - 1]:
                for j in range(1, N1S3):
                    if EN <= X1S3[j]:
                        break
                A = (Y1S3[j] - Y1S3[j - 1]) / (X1S3[j] - X1S3[j - 1])
                B = (X1S3[j - 1] * Y1S3[j] - X1S3[j] * Y1S3[j - 1]) / (X1S3[j - 1] - X1S3[j])
                object.InelasticCrossSectionPerGas[2][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[2][I] = Y1S3[N1S3 - 1] * (X1S3[N1S3 - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[2]):
                object.PEInelasticCrossSectionPerGas[2][I] = object.PEElasticCrossSection[1][I - IOFFN[2]]

        # 1S2 F=0.2260
        if EN > object.EnergyLevels[3]:
            if EN <= X1S2[N1S2 - 1]:
                for j in range(1, N1S2):
                    if EN <= X1S2[j]:
                        break
                A = (Y1S2[j] - Y1S2[j - 1]) / (X1S2[j] - X1S2[j - 1])
                B = (X1S2[j - 1] * Y1S2[j] - X1S2[j] * Y1S2[j - 1]) / (X1S2[j - 1] - X1S2[j])
                object.InelasticCrossSectionPerGas[3][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[3][I] = <float>(0.2260) / (object.EnergyLevels[3] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[3])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (
                                           EN + object.E[2] + object.EnergyLevels[3])
            if EN > (2.0 * object.EnergyLevels[3]):
                object.PEInelasticCrossSectionPerGas[3][I] = object.PEElasticCrossSection[1][I - IOFFN[3]]
        # P states, 2P10
        if EN > object.EnergyLevels[4]:
            if EN <= X2P10[N2P10 - 1]:
                for j in range(1, N2P10):
                    if EN <= X2P10[j]:
                        break
                A = (Y2P10[j] - Y2P10[j - 1]) / (X2P10[j] - X2P10[j - 1])
                B = (X2P10[j - 1] * Y2P10[j] - X2P10[j] * Y2P10[j - 1]) / (X2P10[j - 1] - X2P10[j])
                object.InelasticCrossSectionPerGas[4][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[4][I] = Y2P10[N2P10 - 1] * (X2P10[N2P10 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[4]):
                object.PEInelasticCrossSectionPerGas[4][I] = object.PEElasticCrossSection[1][I - IOFFN[4]]

        # P states, 2P9
        if EN > object.EnergyLevels[5]:
            if EN <= X2P9[N2P9 - 1]:
                for j in range(1, N2P9):
                    if EN <= X2P9[j]:
                        break
                A = (Y2P9[j] - Y2P9[j - 1]) / (X2P9[j] - X2P9[j - 1])
                B = (X2P9[j - 1] * Y2P9[j] - X2P9[j] * Y2P9[j - 1]) / (X2P9[j - 1] - X2P9[j])
                object.InelasticCrossSectionPerGas[5][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[5][I] = Y2P9[N2P9 - 1] * (X2P9[N2P9 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[5]):
                object.PEInelasticCrossSectionPerGas[5][I] = object.PEElasticCrossSection[1][I - IOFFN[5]]

        # P states, 2P8
        if EN > object.EnergyLevels[6]:
            if EN <= X2P8[N2P8 - 1]:
                for j in range(1, N2P8):
                    if EN <= X2P8[j]:
                        break
                A = (Y2P8[j] - Y2P8[j - 1]) / (X2P8[j] - X2P8[j - 1])
                B = (X2P8[j - 1] * Y2P8[j] - X2P8[j] * Y2P8[j - 1]) / (X2P8[j - 1] - X2P8[j])
                object.InelasticCrossSectionPerGas[6][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[6][I] = Y2P8[N2P8 - 1] * (X2P8[N2P8 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[6]):
                object.PEInelasticCrossSectionPerGas[6][I] = object.PEElasticCrossSection[1][I - IOFFN[6]]

        # P states, 2P7
        if EN > object.EnergyLevels[7]:
            if EN <= X2P7[N2P7 - 1]:
                for j in range(1, N2P7):
                    if EN <= X2P7[j]:
                        break
                A = (Y2P7[j] - Y2P7[j - 1]) / (X2P7[j] - X2P7[j - 1])
                B = (X2P7[j - 1] * Y2P7[j] - X2P7[j] * Y2P7[j - 1]) / (X2P7[j - 1] - X2P7[j])
                object.InelasticCrossSectionPerGas[7][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[7][I] = Y2P7[N2P7 - 1] * (X2P7[N2P7 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[7]):
                object.PEInelasticCrossSectionPerGas[7][I] = object.PEElasticCrossSection[1][I - IOFFN[7]]

        # P states, 2P6
        if EN > object.EnergyLevels[8]:
            if EN <= X2P6[N2P6 - 1]:
                for j in range(1, N2P6):
                    if EN <= X2P6[j]:
                        break
                A = (Y2P6[j] - Y2P6[j - 1]) / (X2P6[j] - X2P6[j - 1])
                B = (X2P6[j - 1] * Y2P6[j] - X2P6[j] * Y2P6[j - 1]) / (X2P6[j - 1] - X2P6[j])
                object.InelasticCrossSectionPerGas[8][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[8][I] = Y2P6[N2P6 - 1] * (X2P6[N2P6 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[8]):
                object.PEInelasticCrossSectionPerGas[8][I] = object.PEElasticCrossSection[1][I - IOFFN[8]]

        # P states, 2P5
        if EN > object.EnergyLevels[9]:
            if EN <= X2P5[N2P5 - 1]:
                for j in range(1, N2P5):
                    if EN <= X2P5[j]:
                        break
                A = (Y2P5[j] - Y2P5[j - 1]) / (X2P5[j] - X2P5[j - 1])
                B = (X2P5[j - 1] * Y2P5[j] - X2P5[j] * Y2P5[j - 1]) / (X2P5[j - 1] - X2P5[j])
                object.InelasticCrossSectionPerGas[9][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[9][I] = Y2P5[N2P5 - 1] * (X2P5[N2P5 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[9]):
                object.PEInelasticCrossSectionPerGas[9][I] = object.PEElasticCrossSection[1][I - IOFFN[9]]

        # P states, 2P4
        if EN > object.EnergyLevels[10]:
            if EN <= X2P4[N2P4 - 1]:
                for j in range(1, N2P4):
                    if EN <= X2P4[j]:
                        break
                A = (Y2P4[j] - Y2P4[j - 1]) / (X2P4[j] - X2P4[j - 1])
                B = (X2P4[j - 1] * Y2P4[j] - X2P4[j] * Y2P4[j - 1]) / (X2P4[j - 1] - X2P4[j])
                object.InelasticCrossSectionPerGas[10][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[10][I] = Y2P4[N2P4 - 1] * (X2P4[N2P4 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[10]):
                object.PEInelasticCrossSectionPerGas[10][I] = object.PEElasticCrossSection[1][I - IOFFN[10]]

        # P states, 2P3
        if EN > object.EnergyLevels[11]:
            if EN <= X2P3[N2P3 - 1]:
                for j in range(1, N2P3):
                    if EN <= X2P3[j]:
                        break
                A = (Y2P3[j] - Y2P3[j - 1]) / (X2P3[j] - X2P3[j - 1])
                B = (X2P3[j - 1] * Y2P3[j] - X2P3[j] * Y2P3[j - 1]) / (X2P3[j - 1] - X2P3[j])
                object.InelasticCrossSectionPerGas[11][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[11][I] = Y2P3[N2P3 - 1] * (X2P3[N2P3 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[11]):
                object.PEInelasticCrossSectionPerGas[11][I] = object.PEElasticCrossSection[1][I - IOFFN[11]]

        # P states, 2P2
        if EN > object.EnergyLevels[12]:
            if EN <= X2P2[N2P2 - 1]:
                for j in range(1, N2P2):
                    if EN <= X2P2[j]:
                        break
                A = (Y2P2[j] - Y2P2[j - 1]) / (X2P2[j] - X2P2[j - 1])
                B = (X2P2[j - 1] * Y2P2[j] - X2P2[j] * Y2P2[j - 1]) / (X2P2[j - 1] - X2P2[j])
                object.InelasticCrossSectionPerGas[12][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[12][I] = Y2P2[N2P2 - 1] * (X2P2[N2P2 - 1] / EN) ** 2 * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[12]):
                object.PEInelasticCrossSectionPerGas[12][I] = object.PEElasticCrossSection[1][I - IOFFN[12]]

        # P states, 2P1
        if EN > object.EnergyLevels[13]:
            if EN <= X2P1[N2P1 - 1]:
                for j in range(1, N2P1):
                    if EN <= X2P1[j]:
                        break
                A = (Y2P1[j] - Y2P1[j - 1]) / (X2P1[j] - X2P1[j - 1])
                B = (X2P1[j - 1] * Y2P1[j] - X2P1[j] * Y2P1[j - 1]) / (X2P1[j - 1] - X2P1[j])
                object.InelasticCrossSectionPerGas[13][I] = (A * EN + B) * 1.0e-18 * PSCALE
            else:
                object.InelasticCrossSectionPerGas[13][I] = Y2P1[N2P1 - 1] * (X2P1[N2P1 - 1] / EN) * 1.0e-18 * PSCALE
            if EN > (2.0 * object.EnergyLevels[13]):
                object.PEInelasticCrossSectionPerGas[13][I] = object.PEElasticCrossSection[1][I - IOFFN[13]]

        # D states, 3D6
        if EN > object.EnergyLevels[14]:
            if EN <= X3D6[N3D6 - 1]:
                for j in range(1, N3D6):
                    if EN <= X3D6[j]:
                        break
                A = (Y3D6[j] - Y3D6[j - 1]) / (X3D6[j] - X3D6[j - 1])
                B = (X3D6[j - 1] * Y3D6[j] - X3D6[j] * Y3D6[j - 1]) / (X3D6[j - 1] - X3D6[j])
                object.InelasticCrossSectionPerGas[14][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[14][I] = Y3D6[N3D6 - 1] * (X3D6[N3D6 - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[14]):
                object.PEInelasticCrossSectionPerGas[14][I] = object.PEElasticCrossSection[1][I - IOFFN[14]]

        # D states, 3D5
        if EN > object.EnergyLevels[15]:
            if EN <= X3D5[N3D5 - 1]:
                for j in range(1, N3D5):
                    if EN <= X3D5[j]:
                        break
                A = (Y3D5[j] - Y3D5[j - 1]) / (X3D5[j] - X3D5[j - 1])
                B = (X3D5[j - 1] * Y3D5[j] - X3D5[j] * Y3D5[j - 1]) / (X3D5[j - 1] - X3D5[j])
                object.InelasticCrossSectionPerGas[15][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[15][I] =<float>( 0.0010) / (object.EnergyLevels[15] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[15])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (
                                            EN + object.E[2] + object.EnergyLevels[15])
            if EN > (2.0 * object.EnergyLevels[15]):
                object.PEInelasticCrossSectionPerGas[15][I] = object.PEElasticCrossSection[1][I - IOFFN[15]]

        # D states, 3D3
        if EN > object.EnergyLevels[16]:
            if EN <= X3D3[N3D3 - 1]:
                for j in range(1, N3D3):
                    if EN <= X3D3[j]:
                        break
                A = (Y3D3[j] - Y3D3[j - 1]) / (X3D3[j] - X3D3[j - 1])
                B = (X3D3[j - 1] * Y3D3[j] - X3D3[j] * Y3D3[j - 1]) / (X3D3[j - 1] - X3D3[j])
                object.InelasticCrossSectionPerGas[16][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[16][I] = Y3D3[N3D3 - 1] * (X3D3[N3D3 - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[16]):
                object.PEInelasticCrossSectionPerGas[16][I] = object.PEElasticCrossSection[1][I - IOFFN[16]]

        # D states, 3D4'
        if EN > object.EnergyLevels[17]:
            if EN <= X3D4P[N3D4P - 1]:
                for j in range(1, N3D4P):
                    if EN <= X3D4P[j]:
                        break
                A = (Y3D4P[j] - Y3D4P[j - 1]) / (X3D4P[j] - X3D4P[j - 1])
                B = (X3D4P[j - 1] * Y3D4P[j] - X3D4P[j] * Y3D4P[j - 1]) / (X3D4P[j - 1] - X3D4P[j])
                object.InelasticCrossSectionPerGas[17][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[17][I] = Y3D4P[N3D4P - 1] * (X3D4P[N3D4P - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[17]):
                object.PEInelasticCrossSectionPerGas[17][I] = object.PEElasticCrossSection[1][I - IOFFN[17]]

        # D states, 3D4
        if EN > object.EnergyLevels[18]:
            if EN <= X3D4[N3D4 - 1]:
                for j in range(1, N3D4):
                    if EN <= X3D4[j]:
                        break
                A = (Y3D4[j] - Y3D4[j - 1]) / (X3D4[j] - X3D4[j - 1])
                B = (X3D4[j - 1] * Y3D4[j] - X3D4[j] * Y3D4[j - 1]) / (X3D4[j - 1] - X3D4[j])
                object.InelasticCrossSectionPerGas[18][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[18][I] = Y3D4[N3D4 - 1] * (X3D4[N3D4 - 1] / EN) ** 2 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[18]):
                object.PEInelasticCrossSectionPerGas[18][I] = object.PEElasticCrossSection[1][I - IOFFN[18]]

        # D states, 3D1''
        if EN > object.EnergyLevels[19]:
            if EN <= X3D1PP[N3D1PP - 1]:
                for j in range(1, N3D1PP):
                    if EN <= X3D1PP[j]:
                        break
                A = (Y3D1PP[j] - Y3D1PP[j - 1]) / (X3D1PP[j] - X3D1PP[j - 1])
                B = (X3D1PP[j - 1] * Y3D1PP[j] - X3D1PP[j] * Y3D1PP[j - 1]) / (X3D1PP[j - 1] - X3D1PP[j])
                object.InelasticCrossSectionPerGas[19][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[19][I] = Y3D1PP[N3D1PP - 1] * (X3D1PP[N3D1PP - 1] / EN) ** 2 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[19]):
                object.PEInelasticCrossSectionPerGas[19][I] = object.PEElasticCrossSection[1][I - IOFFN[19]]

        # S states, 2S5
        if EN > object.EnergyLevels[20]:
            if EN <= X2S5[N2S5 - 1]:
                for j in range(1, N2S5):
                    if EN <= X2S5[j]:
                        break
                A = (Y2S5[j] - Y2S5[j - 1]) / (X2S5[j] - X2S5[j - 1])
                B = (X2S5[j - 1] * Y2S5[j] - X2S5[j] * Y2S5[j - 1]) / (X2S5[j - 1] - X2S5[j])
                object.InelasticCrossSectionPerGas[20][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[20][I] = Y2S5[N2S5 - 1] * (X2S5[N2S5 - 1] / EN) ** 2 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[20]):
                object.PEInelasticCrossSectionPerGas[20][I] = object.PEElasticCrossSection[1][I - IOFFN[20]]

        # S states, 2S4 F=0.0257
        if EN > object.EnergyLevels[21]:
            object.InelasticCrossSectionPerGas[21][I] = <float>(0.0257) / (object.EnergyLevels[21] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[21])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[21])
            if object.InelasticCrossSectionPerGas[21][I] < 0:
                object.InelasticCrossSectionPerGas[21][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[21]):
                object.PEInelasticCrossSectionPerGas[21][I] = object.PEElasticCrossSection[1][I - IOFFN[21]]

        # D states, 3D1'
        if EN > object.EnergyLevels[22]:
            if EN <= X3D1P[N3D1P - 1]:
                for j in range(1, N3D1P):
                    if EN <= X3D1P[j]:
                        break
                A = (Y3D1P[j] - Y3D1P[j - 1]) / (X3D1P[j] - X3D1P[j - 1])
                B = (X3D1P[j - 1] * Y3D1P[j] - X3D1P[j] * Y3D1P[j - 1]) / (X3D1P[j - 1] - X3D1P[j])
                object.InelasticCrossSectionPerGas[22][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[22][I] = Y3D1P[N3D1P - 1] * (X3D1P[N3D1P - 1] / EN) * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[22]):
                object.PEInelasticCrossSectionPerGas[22][I] = object.PEElasticCrossSection[1][I - IOFFN[22]]

        # D states, 3D2 F=0.074
        if EN > object.EnergyLevels[23]:
            object.InelasticCrossSectionPerGas[23][I] = <float>(0.074) / (object.EnergyLevels[23] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[23])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[23])
            if object.InelasticCrossSectionPerGas[23][I] < 0:
                object.InelasticCrossSectionPerGas[23][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[23]):
                object.PEInelasticCrossSectionPerGas[23][I] = object.PEElasticCrossSection[1][I - IOFFN[23]]

        # S states, 3S1''''
        if EN > object.EnergyLevels[24]:
            if EN <= X3S1PPPP[N3S1PPPP - 1]:
                for j in range(1, N3S1PPPP):
                    if EN <= X3S1PPPP[j]:
                        break
                A = (Y3S1PPPP[j] - Y3S1PPPP[j - 1]) / (X3S1PPPP[j] - X3S1PPPP[j - 1])
                B = (X3S1PPPP[j - 1] * Y3S1PPPP[j] - X3S1PPPP[j] * Y3S1PPPP[j - 1]) / (X3S1PPPP[j - 1] - X3S1PPPP[j])
                object.InelasticCrossSectionPerGas[24][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[24][I] = Y3S1PPPP[N3S1PPPP - 1] * (X3S1PPPP[N3S1PPPP - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[24]):
                object.PEInelasticCrossSectionPerGas[24][I] = object.PEElasticCrossSection[1][I - IOFFN[24]]

        # S states, 3S1''
        if EN > object.EnergyLevels[25]:
            if EN <= X3S1PP[N3S1PP - 1]:
                for j in range(1, N3S1PP):
                    if EN <= X3S1PP[j]:
                        break
                A = (Y3S1PP[j] - Y3S1PP[j - 1]) / (X3S1PP[j] - X3S1PP[j - 1])
                B = (X3S1PP[j - 1] * Y3S1PP[j] - X3S1PP[j] * Y3S1PP[j - 1]) / (X3S1PP[j - 1] - X3S1PP[j])
                object.InelasticCrossSectionPerGas[25][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[25][I] = Y3S1PP[N3S1PP - 1] * (X3S1PP[N3S1PP - 1] / EN) ** 3 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[25]):
                object.PEInelasticCrossSectionPerGas[25][I] = object.PEElasticCrossSection[1][I - IOFFN[25]]

        # S states, 3S'''
        if EN > object.EnergyLevels[26]:
            if EN <= X3S1PPP[N3S1PPP - 1]:
                for j in range(1, N3S1PPP):
                    if EN <= X3S1PPP[j]:
                        break
                A = (Y3S1PPP[j] - Y3S1PPP[j - 1]) / (X3S1PPP[j] - X3S1PPP[j - 1])
                B = (X3S1PPP[j - 1] * Y3S1PPP[j] - X3S1PPP[j] * Y3S1PPP[j - 1]) / (X3S1PPP[j - 1] - X3S1PPP[j])
                object.InelasticCrossSectionPerGas[26][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[26][I] = Y3S1PPP[N3S1PPP - 1] * (X3S1PPP[N3S1PPP - 1] / EN) * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[26]):
                object.PEInelasticCrossSectionPerGas[26][I] = object.PEElasticCrossSection[1][I - IOFFN[26]]

        # S states, 2S3
        if EN > object.EnergyLevels[27]:
            if EN <= X2S3[N2S3 - 1]:
                for j in range(1, N2S3):
                    if EN <= X2S3[j]:
                        break
                A = (Y2S3[j] - Y2S3[j - 1]) / (X2S3[j] - X2S3[j - 1])
                B = (X2S3[j - 1] * Y2S3[j] - X2S3[j] * Y2S3[j - 1]) / (X2S3[j - 1] - X2S3[j])
                object.InelasticCrossSectionPerGas[27][I] = (A * EN + B) * 1.0e-18
            else:
                object.InelasticCrossSectionPerGas[27][I] = Y2S3[N2S3 - 1] * (X2S3[N2S3 - 1] / EN) ** 2 * 1.0e-18
            if EN > (2.0 * object.EnergyLevels[27]):
                object.PEInelasticCrossSectionPerGas[27][I] = object.PEElasticCrossSection[1][I - IOFFN[27]]

        # S states, 2S2 F=0.011
        if EN > object.EnergyLevels[28]:
            object.InelasticCrossSectionPerGas[28][I] = <float>(0.011) / (object.EnergyLevels[28] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[28])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[28])
            if object.InelasticCrossSectionPerGas[28][I] < 0:
                object.InelasticCrossSectionPerGas[28][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[28]):
                object.PEInelasticCrossSectionPerGas[28][I] = object.PEElasticCrossSection[1][I - IOFFN[28]]

        # S states, 3S1' F=0.092
        if EN > object.EnergyLevels[29]:
            object.InelasticCrossSectionPerGas[29][I] = <float>(0.092) / (object.EnergyLevels[29] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[29])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[29])
            if object.InelasticCrossSectionPerGas[29][I] < 0:
                object.InelasticCrossSectionPerGas[29][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[29]):
                object.PEInelasticCrossSectionPerGas[29][I] = object.PEElasticCrossSection[1][I - IOFFN[29]]

        # D states, 4D5 F=0.019
        if EN > object.EnergyLevels[30]:
            object.InelasticCrossSectionPerGas[30][I] = <float>(0.0019) / (object.EnergyLevels[30] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[30])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[30])
            if object.InelasticCrossSectionPerGas[30][I] < 0:
                object.InelasticCrossSectionPerGas[30][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[30]):
                object.PEInelasticCrossSectionPerGas[30][I] = object.PEElasticCrossSection[1][I - IOFFN[30]]

        # S states, 3S4 F=0.0144
        if EN > object.EnergyLevels[31]:
            object.InelasticCrossSectionPerGas[31][I] = <float>(0.0144) / (object.EnergyLevels[31] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[31])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[31])
            if object.InelasticCrossSectionPerGas[31][I] < 0:
                object.InelasticCrossSectionPerGas[31][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[31]):
                object.PEInelasticCrossSectionPerGas[31][I] = object.PEElasticCrossSection[1][I - IOFFN[31]]

        # D states, 4D2 F=0.0484
        if EN > object.EnergyLevels[32]:
            object.InelasticCrossSectionPerGas[32][I] = <float>(0.0484) / (object.EnergyLevels[32] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[32])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[32])
            if object.InelasticCrossSectionPerGas[32][I] < 0:
                object.InelasticCrossSectionPerGas[32][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[32]):
                object.PEInelasticCrossSectionPerGas[32][I] = object.PEElasticCrossSection[1][I - IOFFN[32]]

        # S states, 4S1' F=0.0209
        if EN > object.EnergyLevels[33]:
            object.InelasticCrossSectionPerGas[33][I] = <float>(0.0209 )/ (object.EnergyLevels[33] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[33])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[33])
            if object.InelasticCrossSectionPerGas[33][I] < 0:
                object.InelasticCrossSectionPerGas[33][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[33]):
                object.PEInelasticCrossSectionPerGas[33][I] = object.PEElasticCrossSection[1][I - IOFFN[33]]

        # S states, 3S2 F=0.022
        if EN > object.EnergyLevels[34]:
            object.InelasticCrossSectionPerGas[34][I] = <float>(0.022) / (object.EnergyLevels[34] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[34])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[34])
            if object.InelasticCrossSectionPerGas[34][I] < 0:
                object.InelasticCrossSectionPerGas[34][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[34]):
                object.PEInelasticCrossSectionPerGas[34][I] = object.PEElasticCrossSection[1][I - IOFFN[34]]

        # D states, 5D5 F=0.0041
        if EN > object.EnergyLevels[35]:
            object.InelasticCrossSectionPerGas[35][I] =<float>( 0.0041) / (object.EnergyLevels[35] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[35])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[35])
            if object.InelasticCrossSectionPerGas[35][I] < 0:
                object.InelasticCrossSectionPerGas[35][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[35]):
                object.PEInelasticCrossSectionPerGas[35][I] = object.PEElasticCrossSection[1][I - IOFFN[35]]

        # S states, 4S4 F=0.0426
        if EN > object.EnergyLevels[36]:
            object.InelasticCrossSectionPerGas[36][I] = <float>(0.0426 )/ (object.EnergyLevels[36] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[36])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[36])
            if object.InelasticCrossSectionPerGas[36][I] < 0:
                object.InelasticCrossSectionPerGas[36][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[36]):
                object.PEInelasticCrossSectionPerGas[36][I] = object.PEElasticCrossSection[1][I - IOFFN[36]]

        # D states, 5D2 F=0.0426
        if EN > object.EnergyLevels[37]:
            object.InelasticCrossSectionPerGas[37][I] = <float>(0.0426) / (object.EnergyLevels[37] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[37])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[37])
            if object.InelasticCrossSectionPerGas[37][I] < 0:
                object.InelasticCrossSectionPerGas[37][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[37]):
                object.PEInelasticCrossSectionPerGas[37][I] = object.PEElasticCrossSection[1][I - IOFFN[37]]

        # D states, 6D5 F=0.00075
        if EN > object.EnergyLevels[38]:
            object.InelasticCrossSectionPerGas[38][I] = <float>(0.00075) / (object.EnergyLevels[38] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[38])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[38])
            if object.InelasticCrossSectionPerGas[38][I] < 0:
                object.InelasticCrossSectionPerGas[38][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[38]):
                object.PEInelasticCrossSectionPerGas[38][I] = object.PEElasticCrossSection[1][I - IOFFN[38]]

        # S states, 5S1' F=0.00051
        if EN > object.EnergyLevels[39]:
            object.InelasticCrossSectionPerGas[39][I] = <float>(0.00051) / (object.EnergyLevels[39] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[39])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[39])
            if object.InelasticCrossSectionPerGas[39][I] < 0:
                object.InelasticCrossSectionPerGas[39][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[39]):
                object.PEInelasticCrossSectionPerGas[39][I] = object.PEElasticCrossSection[1][I - IOFFN[39]]

        # S states, 4S2 F=0.00074
        if EN > object.EnergyLevels[40]:
            object.InelasticCrossSectionPerGas[40][I] = <float>(0.00074) / (object.EnergyLevels[40] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[40])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[40])
            if object.InelasticCrossSectionPerGas[40][I] < 0:
                object.InelasticCrossSectionPerGas[40][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[40]):
                object.PEInelasticCrossSectionPerGas[40][I] = object.PEElasticCrossSection[1][I - IOFFN[40]]

        # S states, 5S4 F=0.013
        if EN > object.EnergyLevels[41]:
            object.InelasticCrossSectionPerGas[41][I] = <float>(0.013) / (object.EnergyLevels[41] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[41])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[41])
            if object.InelasticCrossSectionPerGas[41][I] < 0:
                object.InelasticCrossSectionPerGas[41][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[41]):
                object.PEInelasticCrossSectionPerGas[41][I] = object.PEElasticCrossSection[1][I - IOFFN[41]]

        # S states, 6D2 F=0.029
        if EN > object.EnergyLevels[42]:
            object.InelasticCrossSectionPerGas[42][I] = <float>(0.029) / (object.EnergyLevels[42] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[42])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[42])
            if object.InelasticCrossSectionPerGas[42][I] < 0:
                object.InelasticCrossSectionPerGas[42][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[42]):
                object.PEInelasticCrossSectionPerGas[42][I] = object.PEElasticCrossSection[1][I - IOFFN[42]]

        # sum higher j=1 states f=0.1315
        if EN > object.EnergyLevels[43]:
            object.InelasticCrossSectionPerGas[43][I] = <float>(0.1315) / (object.EnergyLevels[43] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[43])) - BETA2 - object.DEN[I] / 2.0) * BBCONST * EN / (
                                        EN + object.E[2] + object.EnergyLevels[43])
            if object.InelasticCrossSectionPerGas[43][I] < 0:
                object.InelasticCrossSectionPerGas[43][I] = 0.0
            if EN > (2.0 * object.EnergyLevels[43]):
                object.PEInelasticCrossSectionPerGas[43][I] = object.PEElasticCrossSection[1][I - IOFFN[43]]

        Q1SSum = object.InelasticCrossSectionPerGas[0][I] + object.InelasticCrossSectionPerGas[1][I] + object.InelasticCrossSectionPerGas[2][I] + object.InelasticCrossSectionPerGas[3][I]
        QPSSum = 0
        QDSSum = 0
        for i in range(14, 44):
            QDSSum += object.InelasticCrossSectionPerGas[i][I]
        for i in range(4, 14):
            QPSSum += object.InelasticCrossSectionPerGas[i][I]
        TOTSum = Q1SSum + QPSSum + QDSSum


        object.Q[0][I] = ElasticCrossSectionA + Q1SSum + QPSSum + QDSSum
        for i in range(0, 7):
            object.Q[0][I] += object.IonizationCrossSection[i][I]


    for j in range(0, object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[j]:
            object.N_Inelastic = j
            break

