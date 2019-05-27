from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow
cimport libc.math
import numpy as np
cimport numpy as np
import sys
from Gas cimport Gas
from cython.parallel import prange
cimport GasUtil

sys.path.append('../hdf5_python')
import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.fast_getattr(True)
cdef void Gas16(Gas*object):
    gd = np.load('gases.npy').item()

    cdef double XELA[216], YELA[216], YMOM[216], YEPS[216], XROT[70], YROT[70], XVB1[87], YVB1[87], XVB2[69], YVB2[69], XVB3[70], YVB3[70],
    cdef double XVB4[50], YVB4[50], XVB5[40], YVB5[40], XVB6[41], YVB6[41], XVB7[42], YVB7[42], XVB8[40], YVB8[40], XVB9[35], YVB9[35],
    cdef double XVB10[35], YVB10[35], XVB11[35], YVB11[35], XVB12[33], YVB12[33], XVB13[31], YVB13[31], XVB14[28], YVB14[28], XVB15[32], YVB15[32],
    cdef double XTRP1[23], YTRP1[23], YTP1M[23], XTRP2[23], YTRP2[23], YTP2M[23], XTRP3[21], YTRP3[21], YTP3M[21], XTRP4[22], YTRP4[22], YTP4M[22],
    cdef double XTRP5[23], YTRP5[23], YTP5M[23], XTRP6[21], YTRP6[21], YTP6M[21], XTRP7[21], YTRP7[21], YTP7M[21], XTRP8[21], YTRP8[21], YTP8M[21],
    cdef double XTRP9[20], YTRP9[20], YTP9M[20], XTRP10[20], YTRP10[20], YTP10M[20], XTRP11[19], YTRP11[19], YTP11M[19], XTRP12[22], YTRP12[22], YTP12M[22],
    cdef double XTRP13[10], YTRP13[10], YTP13M[10], XTRP14[10], YTRP14[10], YTP14M[10], XSNG1[19], YSNG1[19], YSG1M[19], XSNG2[17], YSNG2[17], YSG2M[17],
    cdef double XSNG3[17], YSNG3[17], YSG3M[17], XSNG4[19], YSNG4[19], YSG4M[19], XSNG5[17], YSNG5[17], YSG5M[17], XSNG6[16], YSNG6[16], YSG6M[16],
    cdef double XSNG7[12], YSNG7[12], YSG7M[12], XSNG8[8], YSNG8[8], YSG8M[8], XSNG9[16], YSNG9[16], YSG9M[16], XSNG10[8], YSNG10[8], YSG10M[8],
    cdef double XSNG11[8], YSNG11[8], YSG11M[8], XSNG12[8], YSNG12[8], YSG12M[8], XSNG13[8], YSNG13[8], YSG13M[8], XSNG14[8], YSNG14[8], YSG14M[8],
    cdef double XSNG15[8], YSNG15[8], YSG15M[8], XKSH[89], YKSH[89], XION[87], YION[87], XION1[87], YION1[87], XION2[63], YION2[63],
    cdef double XION3[48], YION3[48], XION4[54], YION4[54], Z7T[25], EBRM[25]
    cdef int IOFFN[127], IOFFION[12]

    XELA = gd['gas16/XELA']
    YELA = gd['gas16/YELA']
    YMOM = gd['gas16/YMOM']
    YEPS = gd['gas16/YEPS']
    XROT = gd['gas16/XROT']
    YROT = gd['gas16/YROT']
    XVB1 = gd['gas16/XVB1']
    YVB1 = gd['gas16/YVB1']
    XVB2 = gd['gas16/XVB2']
    YVB2 = gd['gas16/YVB2']
    XVB3 = gd['gas16/XVB3']
    YVB3 = gd['gas16/YVB3']
    XVB4 = gd['gas16/XVB4']
    YVB4 = gd['gas16/YVB4']
    XVB5 = gd['gas16/XVB5']
    YVB5 = gd['gas16/YVB5']
    XVB6 = gd['gas16/XVB6']
    YVB6 = gd['gas16/YVB6']
    XVB7 = gd['gas16/XVB7']
    YVB7 = gd['gas16/YVB7']
    XVB8 = gd['gas16/XVB8']
    YVB8 = gd['gas16/YVB8']
    XVB9 = gd['gas16/XVB9']
    YVB9 = gd['gas16/YVB9']
    XVB10 = gd['gas16/XVB10']
    YVB10 = gd['gas16/YVB10']
    XVB11 = gd['gas16/XVB11']
    YVB11 = gd['gas16/YVB11']
    XVB12 = gd['gas16/XVB12']
    YVB12 = gd['gas16/YVB12']
    XVB13 = gd['gas16/XVB13']
    YVB13 = gd['gas16/YVB13']
    XVB14 = gd['gas16/XVB14']
    YVB14 = gd['gas16/YVB14']
    XVB15 = gd['gas16/XVB15']
    YVB15 = gd['gas16/YVB15']
    XTRP1 = gd['gas16/XTRP1']
    YTRP1 = gd['gas16/YTRP1']
    YTP1M = gd['gas16/YTP1M']
    XTRP2 = gd['gas16/XTRP2']
    YTRP2 = gd['gas16/YTRP2']
    YTP2M = gd['gas16/YTP2M']
    XTRP3 = gd['gas16/XTRP3']
    YTRP3 = gd['gas16/YTRP3']
    YTP3M = gd['gas16/YTP3M']
    XTRP4 = gd['gas16/XTRP4']
    YTRP4 = gd['gas16/YTRP4']
    YTP4M = gd['gas16/YTP4M']
    XTRP5 = gd['gas16/XTRP5']
    YTRP5 = gd['gas16/YTRP5']
    YTP5M = gd['gas16/YTP5M']
    XTRP6 = gd['gas16/XTRP6']
    YTRP6 = gd['gas16/YTRP6']
    YTP6M = gd['gas16/YTP6M']
    XTRP7 = gd['gas16/XTRP7']
    YTRP7 = gd['gas16/YTRP7']
    YTP7M = gd['gas16/YTP7M']
    XTRP8 = gd['gas16/XTRP8']
    YTRP8 = gd['gas16/YTRP8']
    YTP8M = gd['gas16/YTP8M']
    XTRP9 = gd['gas16/XTRP9']
    YTRP9 = gd['gas16/YTRP9']
    YTP9M = gd['gas16/YTP9M']
    XTRP10 = gd['gas16/XTRP10']
    YTRP10 = gd['gas16/YTRP10']
    YTP10M = gd['gas16/YTP10M']
    XTRP11 = gd['gas16/XTRP11']
    YTRP11 = gd['gas16/YTRP11']
    YTP11M = gd['gas16/YTP11M']
    XTRP12 = gd['gas16/XTRP12']
    YTRP12 = gd['gas16/YTRP12']
    YTP12M = gd['gas16/YTP12M']
    XTRP13 = gd['gas16/XTRP13']
    YTRP13 = gd['gas16/YTRP13']
    YTP13M = gd['gas16/YTP13M']
    XTRP14 = gd['gas16/XTRP14']
    YTRP14 = gd['gas16/YTRP14']
    YTP14M = gd['gas16/YTP14M']
    XSNG1 = gd['gas16/XSNG1']
    YSNG1 = gd['gas16/YSNG1']
    YSG1M = gd['gas16/YSG1M']
    XSNG2 = gd['gas16/XSNG2']
    YSNG2 = gd['gas16/YSNG2']
    YSG2M = gd['gas16/YSG2M']
    XSNG3 = gd['gas16/XSNG3']
    YSNG3 = gd['gas16/YSNG3']
    YSG3M = gd['gas16/YSG3M']
    XSNG4 = gd['gas16/XSNG4']
    YSNG4 = gd['gas16/YSNG4']
    YSG4M = gd['gas16/YSG4M']
    XSNG5 = gd['gas16/XSNG5']
    YSNG5 = gd['gas16/YSNG5']
    YSG5M = gd['gas16/YSG5M']
    XSNG6 = gd['gas16/XSNG6']
    YSNG6 = gd['gas16/YSNG6']
    YSG6M = gd['gas16/YSG6M']
    XSNG7 = gd['gas16/XSNG7']
    YSNG7 = gd['gas16/YSNG7']
    YSG7M = gd['gas16/YSG7M']
    XSNG8 = gd['gas16/XSNG8']
    YSNG8 = gd['gas16/YSNG8']
    YSG8M = gd['gas16/YSG8M']
    XSNG9 = gd['gas16/XSNG9']
    YSNG9 = gd['gas16/YSNG9']
    YSG9M = gd['gas16/YSG9M']
    XSNG10 = gd['gas16/XSNG10']
    YSNG10 = gd['gas16/YSNG10']
    YSG10M = gd['gas16/YSG10M']
    XSNG11 = gd['gas16/XSNG11']
    YSNG11 = gd['gas16/YSNG11']
    YSG11M = gd['gas16/YSG11M']
    XSNG12 = gd['gas16/XSNG12']
    YSNG12 = gd['gas16/YSNG12']
    YSG12M = gd['gas16/YSG12M']
    XSNG13 = gd['gas16/XSNG13']
    YSNG13 = gd['gas16/YSNG13']
    YSG13M = gd['gas16/YSG13M']
    XSNG14 = gd['gas16/XSNG14']
    YSNG14 = gd['gas16/YSNG14']
    YSG14M = gd['gas16/YSG14M']
    XSNG15 = gd['gas16/XSNG15']
    YSNG15 = gd['gas16/YSNG15']
    YSG15M = gd['gas16/YSG15M']
    XKSH = gd['gas16/XKSH']
    YKSH = gd['gas16/YKSH']
    XION = gd['gas16/XION']
    YION = gd['gas16/YION']
    XION1 = gd['gas16/XION1']
    YION1 = gd['gas16/YION1']
    XION2 = gd['gas16/XION2']
    YION2 = gd['gas16/YION2']
    XION3 = gd['gas16/XION3']
    YION3 = gd['gas16/YION3']
    XION4 = gd['gas16/XION4']
    YION4 = gd['gas16/YION4']
    Z7T = gd['gas16/Z7T']
    EBRM = gd['gas16/EBRM']

    cdef double A0, RY, CONST, EMASS2, API, BBCONST, AM2, C, AUGK, B0
    cdef int NBREM, i, j, I, J,
    A0 = 0.52917720859e-08
    RY = 13.60569193
    CONST = 1.873884e-20
    EMASS2 = 1021997.804
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / EMASS2
    #BORN BETHE VALUES FOR IONISATION
    AM2 = 3.35
    C = 38.1
    # AVERAGE AUGER EMISSION FROM KSHELL
    AUGK = 2.0

    object.NION = 12
    object.NATT = 1
    object.NIN = 127
    object.NNULL = 0
    NBREM = 25

    # ANGULAR DISTRIBUTIONS OF ELASTIC AND IONISATION CAN BE EITHER
    # ISOTROPIC (KEL=0) OR
    # CAPITELLI-LONGO (KEL =1)  OR OKHRIMOVSKKY TYPES (KEL=2)
    for J in range(6):
        object.KEL[J] = object.NANISO

    # USE ISOTROPIC SCATTERING FOR ROTATIONAL AND VIBRATIONAL STATES
    for J in range(92):
        object.KIN[J] = 0

    # USE ANISOTROPIC SCATTERING FOR VIBRATIONAL AND EXCITED STATES .
    # ANGULAR DISTRIBUTIONS ARE CAPITELLI-LONGO (FORWARD BACKWARD ASYMMETRY)
    # OR OKRIMOVSKKY

    for J in range(92, object.NIN):
        object.KIN[J] = 1

    cdef int NELA, NROT, NVIB1, NVIB2, NVIB3, NVIB4, NVIB5, NVIB6, NVIB7, NVIB8, NVIB9, NVIB10, NVIB11, NVIB12, NVIB13
    cdef int NVIB14, NVIB15, NTRP1, NTRP2, NTRP3, NTRP4, NTRP5, NTRP6, NTRP7, NTRP8, NTRP9, NTRP10, NTRP11, NTRP12,
    cdef int NTRP13, NTRP14, NSNG1, NSNG2, NSNG3, NSNG4, NSNG5, NSNG6, NSNG7, NSNG8, NSNG9, NSNG10, NSNG11, NSNG12,
    cdef int NSNG13, NSNG14, NSNG15, NIOND, NION1, NION2, NION3, NION4, NKSH,

    NELA = 216
    NROT = 70
    NVIB1 = 87
    NVIB2 = 69
    NVIB3 = 70
    NVIB4 = 50
    NVIB5 = 40
    NVIB6 = 41
    NVIB7 = 42
    NVIB8 = 40
    NVIB9 = 35
    NVIB10 = 35
    NVIB11 = 35
    NVIB12 = 33
    NVIB13 = 31
    NVIB14 = 28
    NVIB15 = 32
    NTRP1 = 23
    NTRP2 = 23
    NTRP3 = 21
    NTRP4 = 22
    NTRP5 = 23
    NTRP6 = 21
    NTRP7 = 21
    NTRP8 = 21
    NTRP9 = 20
    NTRP10 = 20
    NTRP11 = 19
    NTRP12 = 22
    NTRP13 = 10
    NTRP14 = 10
    NSNG1 = 19
    NSNG2 = 17
    NSNG3 = 17
    NSNG4 = 19
    NSNG5 = 17
    NSNG6 = 16
    NSNG7 = 12
    NSNG8 = 8
    NSNG9 = 16
    NSNG10 = 8
    NSNG11 = 8
    NSNG12 = 8
    NSNG13 = 8
    NSNG14 = 8
    NSNG15 = 8
    NIOND = 87
    NION1 = 87
    NION2 = 63
    NION3 = 48
    NION4 = 54
    NKSH = 89

    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, EOBY[12], SUMR, SUMV, SUMEX, SUMEX1

    object.E = [0.0, 1.0, 15.581, 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * EMASS / (27.7940 * AMU)
    object.EION[0:12] = [15.581, 15.855, 16.699, 16.935, 17.171, 18.751, 23.591, 24.294, 24.4, 35.7, 38.8, 401.6]

    for J in range(12):
        EOBY[J] = 13.6
        object.NC0[J] = 0.0
        object.EC0[J] = 0.0
        object.WK[J] = 0.0
        object.EFL[J] = 0.0
        object.NG1[J] = 0.0
        object.EG1[J] = 0.0
        object.NG2[J] = 0.0
        object.EG2[J] = 0.0

    # DOUBLY CHARGED STATES
    object.NC0[10] = 1
    object.EC0[10] = 6.0
    # FLUORESENCE DATA
    object.NC0[11] = 2.0
    object.EC0[11] = 358.6
    object.WK[11] = 0.0044
    object.EFL[11] = 385.
    object.NG1[11] = 1
    object.EG1[11] = 353.
    object.NG2[11] = 1
    object.EG2[11] = 6.

    cdef double QBQA, QBK, SUM, FROT0, PJ[39], RAT
    #CALC FRACTIONAL POPULATION DENSITY FOR ROTATIONAL STATES
    B0 = 2.4668e-4
    #ROTATIONAL QUADRUPOLE MOMENT
    QBQA = 1.045
    QBK = 1.67552 * (QBQA * A0) ** 2
    for J in range(1, 40, 2):
        PJ[J - 1] = 3.0 * (2.0 * J + 1.0) * exp(-1.0 * J * (J + 1.0) * B0 / object.AKT)
    for J in range(2, 39, 2):
        PJ[J - 1] = 6.0 * (2.0 * J + 1.0) * exp(-1.0 * J * (J + 1.0) * B0 / object.AKT)
    SUM = 6.0
    for J in range(39):
        SUM += PJ[J]
    FROT0 = 6.0 / SUM
    for J in range(39):
        PJ[J] /= SUM

    object.EIN = gd['gas16/EIN']

    # OFFSET ENERGY FOR IONISATION ELECTRON ANGULAR DISTRIBUTION
    for j in range(0, object.NION):
        for i in range(0, 4000):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i - 1
                break

    cdef int NL = 0
    for NL in range(object.NIN):
        for i in range(4000):
            if object.EG[i] > abs(object.EIN[NL]):
                IOFFN[NL] = i - 1
                break

    for I in range(106):
        for J in range(3):
            object.PENFRA[I][J] = 0.0

    for J in range(106, 127):
        object.PENFRA[0][J] = 0.0
        object.PENFRA[1][J] = 1.0
        object.PENFRA[2][J] = 1.0

    cdef double APOPV1, APOPGS, APOPSUM, EN, GAMMA1, GAMMA2, BETA, BETA2, QELA, QMOM, PQ[3], QN2PTOT, QNPTOT, RESFAC, ASCALE

    for J in range(NBREM):
        EBRM[J] = exp(EBRM[J])
    # CALC VIBRATIONAL LEVEL V1 POPULATION
    APOPV1 = exp(object.EIN[76] / object.AKT)
    APOPGS = 1.0
    APOPSUM = APOPGS + APOPV1
    APOPV1 = APOPV1 / APOPSUM
    APOPGS = APOPGS / APOPSUM
    #  RENORMALISE GROUND STATE TO ALLOW FOR EXCITATION FROM
    #  THE EXCITED VIBRATIONAL STATE
    APOPGS = 1.0

    for I in range(4000):
        EN = object.EG[I]
        GAMMA1 = (EMASS2 + 2 * EN) / EMASS2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA

        # ELASTIC (+ROTATIONAL)
        QELA = GasUtil.CALQIONREG(EN, NELA, YELA, XELA)
        QMOM = GasUtil.CALQIONREG(EN, NELA, YMOM, XELA)
        PQ[2] = GasUtil.CALPQ3(EN, NELA, YEPS, XELA)

        PQ[2] = 1 - PQ[2]
        PQ[1] = 0.5 + (QELA - QMOM) / QELA
        PQ[0] = 0.5

        object.PEQEL[1][I] = PQ[object.NANISO]

        object.Q[1][I] = QELA
        if object.NANISO == 0:
            object.Q[1][I] = QMOM

        for J in range(12):
            object.PEQION[J][I] = 0.5
            if object.NANISO == 2:
                object.PEQION[J][I] = 0.0
            object.QION[J][I] = 0.0

        # IONISATION TO ALL CHANNELS WITH N2+
        QN2PTOT = 0.0
        if EN > object.EION[0]:
            QN2PTOT = GasUtil.CALQIONX(EN, NION1, YION1, XION1, BETA2, 0.7973, CONST, object.DEN[I], C, AM2)

        object.QION[0][I] = QN2PTOT

        if EN > object.EION[1] and EN <= object.EION[2]:
            object.QION[1][I] = QN2PTOT * 0.2
            object.QION[0][I] = QN2PTOT * 0.8
        elif EN > object.EION[2] and EN <= object.EION[3]:
            object.QION[2][I] = QN2PTOT * 0.1986
            object.QION[1][I] = QN2PTOT * 0.1603
            object.QION[0][I] = QN2PTOT * 0.6411
        elif EN > object.EION[3] and EN <= object.EION[4]:
            object.QION[3][I] = QN2PTOT * 0.2296
            object.QION[2][I] = QN2PTOT * 0.1530
            object.QION[1][I] = QN2PTOT * 0.1235
            object.QION[0][I] = QN2PTOT * 0.4939
        elif EN > object.EION[4] and EN <= object.EION[5]:
            object.QION[4][I] = QN2PTOT * 0.2765
            object.QION[3][I] = QN2PTOT * 0.1659
            object.QION[2][I] = QN2PTOT * 0.1106
            object.QION[1][I] = QN2PTOT * 0.0894
            object.QION[0][I] = QN2PTOT * 0.3576
        elif EN > object.EION[5] and EN <= object.EION[6]:
            object.QION[5][I] = QN2PTOT * 0.1299
            object.QION[4][I] = QN2PTOT * 0.2408
            object.QION[3][I] = QN2PTOT * 0.1445
            object.QION[2][I] = QN2PTOT * 0.0963
            object.QION[1][I] = QN2PTOT * 0.0777
            object.QION[0][I] = QN2PTOT * 0.3108
        elif EN > object.EION[6]:
            object.QION[6][I] = QN2PTOT * 0.022
            object.QION[5][I] = QN2PTOT * 0.127
            object.QION[4][I] = QN2PTOT * 0.2355
            object.QION[3][I] = QN2PTOT * 0.1413
            object.QION[2][I] = QN2PTOT * 0.0942
            object.QION[1][I] = QN2PTOT * 0.076
            object.QION[0][I] = QN2PTOT * 0.304

        # IONISATION TO aLL CHANNELS WITH N +
        QNPTOT = 0.0
        if EN > object.EION[7]:
            GasUtil.CALQIONX(EN, NION2, YION2, XION2, BETA2, 0.197, CONST, object.DEN[I], C, AM2)

        object.QION[7][I] = QNPTOT
        if EN > object.EION[8] and EN <= object.EION[9]:
            if EN < 110:
                object.QION[8][I] = ((EN - object.EION[8]) / (110. - object.EION[8])) * 0.095 * 1.e-16
            else:
                object.QION[8][I] = object.QION[7][I] * 0.1439
            object.QION[7][I] = object.QION[7][I] - object.QION[8][I]

        elif EN > object.EION[9]:
            if EN < 110:
                object.QION[8][I] = ((EN - object.EION[8]) / (110. - object.EION[8])) * 0.095 * 1.e-16
            elif EN >= 110:
                object.QION[8][I] = object.QION[7][I] * 0.1439
            if EN < 120:
                object.QION[9][I] = ((EN - object.EION[9]) / (120. - object.EION[9])) * 0.037 * 1.e-16
            else:
                object.QION[9][I] = object.QION[8][I] * 0.056
            object.QION[7][I] = object.QION[7][I] - object.QION[8][I] - object.QION[9][I]

        if EN > object.EION[10]:
            # SUM OF DOUBLE IONISATION CHANNELS: N+,N+  AND N++,N
            object.QION[10][I] = 0.0
            object.QION[10][I] = GasUtil.CALQIONX(EN, NION3, YION3, XION3, BETA2, 0.0338, CONST, object.DEN[I], C, AM2)
            object.QION[7][I] -= object.QION[10][I]

        if EN > 65.0:
            object.QION[10][I] += GasUtil.CALQIONX(EN, NION4, YION4, XION4, BETA2, 0.0057, CONST, object.DEN[I], C, AM2)

        if EN > object.EION[11]:
            object.QION[11][I] = 2 * GasUtil.CALQIONREG(EN, NKSH, YKSH, XKSH)

        for J in range(12):
            if EN > 2 * object.EION[J]:
                object.PEQION[J][I] = object.PEQEL[1][I - IOFFION[J]]

        # CORRECTION TO IONISATION FOR AUGER EMISSION
        object.QION[0][I] -= AUGK * object.QION[11][I]

        object.Q[3][I] = 0.0
        object.QATT[0][I] = 0.0
        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        #---------------------------------------------------------------------
        #  QUADRUPOLE BORN ROTATIONAL STATES  ( GERJUOY AND STEIN)
        #---------------------------------------------------------------------
        #
        #  SUPERELASTIC ROTATION
        #

        for J in range(1, 39):
            object.QIN[J - 1][I] = 0.0
            object.PEQIN[J - 1][I] = 0.5
            if object.NANISO == 2:
                object.PEQIN[J - 1][I] = 0.0
            if EN > 0.0:
                i = J + 1
                object.QIN[J - 1][I] = PJ[J] * QBK * sqrt(1.0 - object.EIN[J - 1] / EN) * i * (i - 1.0) / (
                        (2.0 * i + 1.0) * (2.0 * i - 1.0))
                #CALCULATE ENHANCEMENT OF ROTATIONAL XSEC IN THE RESONANCE REGION
                RESFAC = GasUtil.CALPQ3(EN - object.EIN[J - 1], NROT, YROT, XROT)
                #USE 30% FOR RESFAC
                RESFAC *= 0.3
                #BORN ROTATIONAL X-SEC SUM IN RESONANCE REGION = 0.249
                RESFAC = 1.0 + RESFAC / 0.249
            object.QIN[J - 1][I] *= RESFAC

        # INELASTIC ROTATION
        #
        # CALCULATE ENHANCEMENT OF ROTATIONAL XSEC IN THE RESONANCE REGION

        for J in range(39, 77):
            object.PEQIN[J - 1][I] = 0.5
            if object.NANISO == 2:
                object.PEQIN[J - 1][I] = 0.0
            object.QIN[J - 1][I] = 0.0
        # CALCULATE ENHANCEMENT OF ROTATIONAL XSEC IN THE RESONANCE REGION
        RESFAC = GasUtil.CALPQ3(EN, NROT, YROT, XROT)
        #USE 30% FOR RESFAC
        RESFAC *= 0.3
        #BORN ROTATIONAL X-SEC SUM IN RESONANCE REGION = 0.249
        RESFAC = 1.0 + RESFAC / 0.249

        # ROT 0-2
        if EN > object.EIN[38]:
            object.QIN[38][I] = FROT0 * QBK * sqrt(1.0 - object.EIN[38] / EN) * 2.0 / 3.0
            object.QIN[38][I] *= RESFAC
            object.PEQIN[38][I] = 0.0
            if object.NANISO == 2:
                object.PEQIN[38][I] = 0.0
            for J in range(40, 77):
                i = J - 39
                if EN > object.EIN[J - 1]:
                    object.QIN[J - 1][I] = PJ[i - 1] * QBK * sqrt(1.0 - object.EIN[J - 1] / EN) * (i + 2.0) * (
                            i + 1.0) / ((2.0 * i + 3.0) * (2.0 * i + 1.0))
                    object.QIN[J - 1][I] *= RESFAC

            if EN >= 5.0:
                ASCALE = QMOM / 8.9e-16
                for J in range(76):
                    object.QIN[J][I] *= ASCALE

        #---------------------------------------------------------------------
        #  VIBRATIONAL AND EXCITATION X-SECTIONS
        #---------------------------------------------------------------------
        #  V1 SUPERELASTIC
        object.QIN[76][I] = 0
        object.PEQIN[76][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[76][I] = 0.0
        if EN > 0.0:
            object.QIN[76][I] = GasUtil.CALQINVISELA(EN, NVIB1, YVB1, XVB1, APOPV1, -1 * object.EIN[76], 1,  object.EIN[76], 0, 0)

        # V1
        object.QIN[77][I] = 0.0
        if EN > object.EIN[77]:
            object.QIN[77][I] = GasUtil.CALQINP(EN, NVIB1, YVB1, XVB1, 1) * 100 * APOPGS

        # V2
        object.QIN[78][I] = 0.0
        if EN > object.EIN[78]:
            object.QIN[78][I] = GasUtil.CALQINP(EN, NVIB2, YVB2, XVB2, 1) * 100 * APOPGS

        # 3V1
        object.QIN[79][I] = 0.0
        if EN > object.EIN[79]:
            object.QIN[79][I] = GasUtil.CALQINP(EN, NVIB3, YVB3, XVB3, 1) * 100 * APOPGS

        # 4V1
        object.QIN[80][I] = 0.0
        if EN > object.EIN[80]:
            object.QIN[80][I] = GasUtil.CALQINP(EN, NVIB4, YVB4, XVB4, 1) * 100 * APOPGS

        # 5V1
        object.QIN[81][I] = 0.0
        if EN > object.EIN[81]:
            object.QIN[81][I] = GasUtil.CALQINP(EN, NVIB5, YVB5, XVB5, 1) * 100 * APOPGS
        # 6V1
        object.QIN[82][I] = 0.0
        if EN > object.EIN[82]:
            object.QIN[82][I] = GasUtil.CALQINP(EN, NVIB6, YVB6, XVB6, 1) * 100 * APOPGS
        # 7V1
        object.QIN[83][I] = 0.0
        if EN > object.EIN[83]:
            object.QIN[83][I] = GasUtil.CALQINP(EN, NVIB7, YVB7, XVB7, 1) * 100 * APOPGS
        # 8V1
        object.QIN[84][I] = 0.0
        if EN > object.EIN[84]:
            object.QIN[84][I] = GasUtil.CALQINP(EN, NVIB8, YVB8, XVB8, 1) * 100 * APOPGS
        # 9V1
        object.QIN[85][I] = 0.0
        if EN > object.EIN[85]:
            object.QIN[85][I] = GasUtil.CALQINP(EN, NVIB9, YVB9, XVB9, 1) * 100 * APOPGS
        # 10V1
        object.QIN[86][I] = 0.0
        if EN > object.EIN[86]:
            object.QIN[86][I] = GasUtil.CALQINP(EN, NVIB10, YVB10, XVB10, 1) * 100 * APOPGS
        # 11V1
        object.QIN[87][I] = 0.0
        if EN > object.EIN[87]:
            object.QIN[87][I] = GasUtil.CALQINP(EN, NVIB11, YVB11, XVB11, 1) * 100 * APOPGS
        # 12V1
        object.QIN[88][I] = 0.0
        if EN > object.EIN[88]:
            object.QIN[88][I] = GasUtil.CALQINP(EN, NVIB12, YVB12, XVB12, 1) * 100 * APOPGS
        # 13V1
        object.QIN[89][I] = 0.0
        if EN > object.EIN[89]:
            object.QIN[89][I] = GasUtil.CALQINP(EN, NVIB13, YVB13, XVB13, 1) * 100 * APOPGS
        # 14V1
        object.QIN[90][I] = 0.0
        if EN > object.EIN[90]:
            object.QIN[90][I] = GasUtil.CALQINP(EN, NVIB14, YVB14, XVB14, 1) * 100 * APOPGS
        # 15V1
        object.QIN[91][I] = 0.0
        if EN > object.EIN[91]:
            object.QIN[91][I] = GasUtil.CALQINP(EN, NVIB15, YVB15, XVB15, 1) * 100 * APOPGS

        # SET ROTATIONAL AND VIBRATIONAL ANGULAR DISTRIBUTIONS ( IF KIN NE 0 )
        for J in range(92):
            object.PEQIN[J][I] = 0.5
            if object.NANISO == 2:
                object.PEQIN[J][I] = 0.0
            if EN > 3 * abs(object.EIN[J]):
                if object.NANISO > 0:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]

        # A3SIGMA (V = 0-4)
        object.QIN[92][I] = 0.0
        object.PEQIN[92][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[92][I] = 0.
        if EN > object.EIN[92]:
            object.QIN[92][I] = GasUtil.CALQINP(EN, NTRP1, YTRP1, XTRP1, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP1, YTP1M, XTRP1, 1) * 1e18
        if EN > 3.0 * object.EIN[92]:
            if object.NANISO == 1:
                object.PEQIN[92][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[92][I] = object.PEQEL[1][I - IOFFN[92]]

        # A3SIGMA (V = 5-9)
        object.QIN[93][I] = 0.0
        object.PEQIN[93][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[93][I] = 0.
        if EN > object.EIN[93]:
            object.QIN[93][I] = GasUtil.CALQINP(EN, NTRP2, YTRP2, XTRP2, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP2, YTP2M, XTRP2, 1) * 1e18
        if EN > 3.0 * object.EIN[93]:
            if object.NANISO == 1:
                object.PEQIN[93][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[93][I] = object.PEQEL[1][I - IOFFN[93]]

        # B3PI (V=0-3)
        object.QIN[94][I] = 0.0
        object.PEQIN[94][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[94][I] = 0.
        if EN > object.EIN[94]:
            object.QIN[94][I] = GasUtil.CALQINP(EN, NTRP3, YTRP3, XTRP3, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP3, YTP3M, XTRP3, 1) * 1e18
        if EN > 3.0 * object.EIN[94]:
            if object.NANISO == 1:
                object.PEQIN[94][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[94][I] = object.PEQEL[1][I - IOFFN[94]]

        # W3DELTA (V = 0-5)
        object.QIN[95][I] = 0.0
        object.PEQIN[95][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[95][I] = 0.
        if EN > object.EIN[95]:
            object.QIN[95][I] = GasUtil.CALQINP(EN, NTRP4, YTRP4, XTRP4, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP4, YTP4M, XTRP4, 1) * 1e18
        if EN > 3.0 * object.EIN[95]:
            if object.NANISO == 1:
                object.PEQIN[95][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[95][I] = object.PEQEL[1][I - IOFFN[95]]

        # A3SIGMA (V = 10-21)
        object.QIN[96][I] = 0.0
        object.PEQIN[96][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[96][I] = 0.
        if EN > object.EIN[96]:
            object.QIN[96][I] = GasUtil.CALQINP(EN, NTRP5, YTRP5, XTRP5, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP5, YTP5M, XTRP5, 1) * 1e18
        if EN > 3.0 * object.EIN[96]:
            if object.NANISO == 1:
                object.PEQIN[96][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[96][I] = object.PEQEL[1][I - IOFFN[96]]

        # B3PI (V=4-16)
        object.QIN[97][I] = 0.0
        object.PEQIN[97][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[97][I] = 0.
        if EN > object.EIN[97]:
            object.QIN[97][I] = GasUtil.CALQINP(EN, NTRP6, YTRP6, XTRP6, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP6, YTP6M, XTRP6, 1) * 1e18
        if EN > 3.0 * object.EIN[97]:
            if object.NANISO == 1:
                object.PEQIN[97][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[97][I] = object.PEQEL[1][I - IOFFN[97]]

        # W3DEL (V=6-10)
        object.QIN[98][I] = 0.0
        object.PEQIN[98][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[98][I] = 0.
        if EN > object.EIN[98]:
            object.QIN[98][I] = GasUtil.CALQINP(EN, NTRP7, YTRP7, XTRP7, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP7, YTP7M, XTRP7, 1) * 1e18
        if EN > 3.0 * object.EIN[98]:
            if object.NANISO == 1:
                object.PEQIN[98][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[98][I] = object.PEQEL[1][I - IOFFN[98]]

        # A1PI (V=0-3)
        object.QIN[99][I] = 0.0
        object.PEQIN[99][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[99][I] = 0.
        if EN > object.EIN[99]:
            object.QIN[99][I] = GasUtil.CALQINP(EN, NSNG1, YSNG1, XSNG1, 1.5) * 100
            RAT = GasUtil.CALQINP(EN, NSNG1, YSG1M, XSNG1, 1) * 1e18
        if EN > 3.0 * object.EIN[99]:
            if object.NANISO == 1:
                object.PEQIN[99][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[99][I] = object.PEQEL[1][I - IOFFN[99]]

        # B!3SIG (V=0-6)
        object.QIN[100][I] = 0.0
        object.PEQIN[100][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[100][I] = 0.
        if EN > object.EIN[100]:
            object.QIN[100][I] = GasUtil.CALQINP(EN, NTRP8, YTRP8, XTRP8, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP8, YTP8M, XTRP8, 1) * 1e18
        if EN > 3.0 * object.EIN[100]:
            if object.NANISO == 1:
                object.PEQIN[100][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[100][I] = object.PEQEL[1][I - IOFFN[100]]

        # A!SIG (V=0-6)
        object.QIN[101][I] = 0.0
        object.PEQIN[101][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[101][I] = 0.
        if EN > object.EIN[101]:
            object.QIN[101][I] = GasUtil.CALQINP(EN, NSNG2, YSNG2, XSNG2, 1.5) * 100
            RAT = GasUtil.CALQINP(EN, NSNG2, YSG2M, XSNG2, 1) * 1e18
        if EN > 3.0 * object.EIN[101]:
            if object.NANISO == 1:
                object.PEQIN[101][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[101][I] = object.PEQEL[1][I - IOFFN[101]]

        # W3DEL(V=11-19)
        object.QIN[102][I] = 0.0
        object.PEQIN[102][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[102][I] = 0.
        if EN > object.EIN[102]:
            object.QIN[102][I] = GasUtil.CALQINP(EN, NTRP9, YTRP9, XTRP9, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP9, YTP9M, XTRP9, 1) * 1e18
        if EN > 3.0 * object.EIN[102]:
            if object.NANISO == 1:
                object.PEQIN[102][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[102][I] = object.PEQEL[1][I - IOFFN[102]]

        # W1DEL (V=0-5)
        object.QIN[103][I] = 0.0
        object.PEQIN[103][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[103][I] = 0.
        if EN > object.EIN[103]:
            object.QIN[103][I] = GasUtil.CALQINP(EN, NSNG3, YSNG3, XSNG3, 1.5) * 100
            RAT = GasUtil.CALQINP(EN, NSNG3, YSG3M, XSNG3, 1) * 1e18
        if EN > 3.0 * object.EIN[103]:
            if object.NANISO == 1:
                object.PEQIN[103][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[103][I] = object.PEQEL[1][I - IOFFN[103]]

        # A1PI (V=4-15)
        object.QIN[104][I] = 0.0
        object.PEQIN[104][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[104][I] = 0.
        if EN > object.EIN[104]:
            object.QIN[104][I] = GasUtil.CALQINP(EN, NSNG4, YSNG4, XSNG4, 1.5) * 100
            RAT = GasUtil.CALQINP(EN, NSNG4, YSG4M, XSNG4, 1) * 1e18
        if EN > 3.0 * object.EIN[104]:
            if object.NANISO == 1:
                object.PEQIN[104][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[104][I] = object.PEQEL[1][I - IOFFN[104]]

        # B!3SIG (V = 7-18)
        object.QIN[105][I] = 0.0
        object.PEQIN[105][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[105][I] = 0.
        if EN > object.EIN[105]:
            object.QIN[105][I] = GasUtil.CALQINP(EN, NTRP10, YTRP10, XTRP10, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP10, YTP10M, XTRP10, 1) * 1e18
        if EN > 3.0 * object.EIN[105]:
            if object.NANISO == 1:
                object.PEQIN[105][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[105][I] = object.PEQEL[1][I - IOFFN[105]]

        # A!1SIG (V=7-19)
        object.QIN[106][I] = 0.0
        object.PEQIN[106][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[106][I] = 0.
        if EN > object.EIN[106]:
            object.QIN[106][I] = GasUtil.CALQINP(EN, NSNG5, YSNG5, XSNG5, 1.5) * 100
            RAT = GasUtil.CALQINP(EN, NSNG5, YSG5M, XSNG5, 1) * 1e18
        if EN > 3.0 * object.EIN[106]:
            if object.NANISO == 1:
                object.PEQIN[106][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[106][I] = object.PEQEL[1][I - IOFFN[106]]

        # W1DEL (V=6-18)
        object.QIN[107][I] = 0.0
        object.PEQIN[107][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[107][I] = 0.
        if EN > object.EIN[107]:
            object.QIN[107][I] = GasUtil.CALQINP(EN, NSNG6, YSNG6, XSNG6, 1.5) * 100
            RAT = GasUtil.CALQINP(EN, NSNG6, YSG6M, XSNG6, 1) * 1e18
        if EN > 3.0 * object.EIN[107]:
            if object.NANISO == 1:
                object.PEQIN[107][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[107][I] = object.PEQEL[1][I - IOFFN[107]]

        # C3PI (V=0-4)
        object.QIN[108][I] = 0.0
        object.PEQIN[108][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[108][I] = 0.
        if EN > object.EIN[108]:
            object.QIN[108][I] = GasUtil.CALQINP(EN, NTRP11, YTRP11, XTRP11, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP11, YTP11M, XTRP11, 1) * 1e18
        if EN > 3.0 * object.EIN[108]:
            if object.NANISO == 1:
                object.PEQIN[108][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[108][I] = object.PEQEL[1][I - IOFFN[108]]

        # E3SIG
        object.QIN[109][I] = 0.0
        object.PEQIN[109][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[109][I] = 0.
        if EN > object.EIN[109]:
            object.QIN[109][I] = GasUtil.CALQINP(EN, NTRP12, YTRP12, XTRP12, 2) * 100
            RAT = GasUtil.CALQINP(EN, NTRP12, YTP12M, XTRP12, 1) * 1e18
        if EN > 3.0 * object.EIN[109]:
            if object.NANISO == 1:
                object.PEQIN[109][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[109][I] = object.PEQEL[1][I - IOFFN[109]]

        # A!!1SIG (V=0-1)
        object.QIN[110][I] = 0.0
        object.PEQIN[110][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[110][I] = 0.
        if EN > object.EIN[110]:
            object.QIN[110][I] = GasUtil.CALQINP(EN, NSNG7, YSNG7, XSNG7, 1.5) * 100
            RAT = GasUtil.CALQINP(EN, NSNG7, YSG7M, XSNG7, 1) * 1e18
        if EN > 3.0 * object.EIN[110]:
            if object.NANISO == 1:
                object.PEQIN[110][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[110][I] = object.PEQEL[1][I - IOFFN[110]]

        # B1PI (V=0-6)   F=0.1855
        object.QIN[111][I] = 0.0
        object.PEQIN[111][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[111][I] = 0.
        if EN > object.EIN[111]:
            object.QIN[111][I] = GasUtil.CALQINBEF(EN, NSNG8, YSNG8, XSNG8, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                   BBCONST, object.EIN[111], object.E[2], 0.1855)
            if EN <= XSNG8[NSNG8 - 1]:
                object.QIN[111][I] *= 100
            RAT = GasUtil.CALQINP(EN, NSNG8, YSG8M, XSNG8, 1) * 1e18
        if EN > 3.0 * object.EIN[111]:
            if object.NANISO == 1:
                object.PEQIN[111][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[111][I] = object.PEQEL[1][I - IOFFN[111]]

        # C!1SIG (V=0-3)   F=0.150
        object.QIN[112][I] = 0.0
        object.PEQIN[112][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[112][I] = 0.
        if EN > object.EIN[112]:
            object.QIN[112][I] = GasUtil.CALQINBEF(EN, NSNG9, YSNG9, XSNG9, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                   BBCONST, object.EIN[112], object.E[2], 0.150)
            if EN <= XSNG9[NSNG9 - 1]:
                object.QIN[112][I] *= 100
            RAT = GasUtil.CALQINP(EN, NSNG9, YSG9M, XSNG9, 1) * 1e18
        if EN > 3.0 * object.EIN[112]:
            if object.NANISO == 1:
                object.PEQIN[112][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[112][I] = object.PEQEL[1][I - IOFFN[112]]

        # G 3PI
        object.QIN[113][I] = 0.0
        object.PEQIN[113][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[113][I] = 0.
        if EN > object.EIN[113]:
            object.QIN[113][I] = GasUtil.CALQINP(EN, NTRP13, YTRP13, XTRP13, 1.5) * 100
            RAT = GasUtil.CALQINP(EN, NTRP13, YTP13M, XTRP13, 1) * 1e18
        if EN > 3.0 * object.EIN[113]:
            if object.NANISO == 1:
                object.PEQIN[113][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[113][I] = object.PEQEL[1][I - IOFFN[113]]

        # C3 1PI (V=0-3)   F=0.150
        object.QIN[114][I] = 0.0
        object.PEQIN[114][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[114][I] = 0.
        if EN > object.EIN[114]:
            object.QIN[114][I] = GasUtil.CALQINBEF(EN, NSNG10, YSNG10, XSNG10, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                   BBCONST, object.EIN[114], object.E[2], 0.150)
            if EN <= XSNG10[NSNG10 - 1]:
                object.QIN[114][I] *= 100
            RAT = GasUtil.CALQINP(EN, NSNG10, YSG10M, XSNG10, 1) * 1e18
        if EN > 3.0 * object.EIN[114]:
            if object.NANISO == 1:
                object.PEQIN[114][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[114][I] = object.PEQEL[1][I - IOFFN[114]]

        # F 3PI (V = 0-3)
        object.QIN[115][I] = 0.0
        object.PEQIN[115][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[115][I] = 0.
        if EN > object.EIN[115]:
            object.QIN[115][I] = GasUtil.CALQINP(EN, NTRP14, YTRP14, XTRP14, 1.5) * 100
            RAT = GasUtil.CALQINP(EN, NTRP14, YTP14M, XTRP14, 1) * 1e18
        if EN > 3.0 * object.EIN[115]:
            if object.NANISO == 1:
                object.PEQIN[115][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[115][I] = object.PEQEL[1][I - IOFFN[115]]

        # B1PI (V=7-14)   F=0.0663
        object.QIN[116][I] = 0.0
        object.PEQIN[116][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[116][I] = 0.
        if EN > object.EIN[116]:
            object.QIN[116][I] = GasUtil.CALQINBEF(EN, NSNG11, YSNG11, XSNG11, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                   BBCONST, object.EIN[116], object.E[2], 0.0663)
            if EN <= XSNG11[NSNG11 - 1]:
                object.QIN[116][I] *= 100
            RAT = GasUtil.CALQINP(EN, NSNG11, YSG11M, XSNG11, 1) * 1e18
        if EN > 3.0 * object.EIN[116]:
            if object.NANISO == 1:
                object.PEQIN[116][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[116][I] = object.PEQEL[1][I - IOFFN[116]]

        # B! SIG (V=0-10)   F=0.0601
        object.QIN[117][I] = 0.0
        object.PEQIN[117][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[117][I] = 0.
        if EN > object.EIN[117]:
            object.QIN[117][I] = GasUtil.CALQINBEF(EN, NSNG12, YSNG12, XSNG12, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                   BBCONST, object.EIN[117], object.E[2], 0.0601)
            if EN <= XSNG12[NSNG12 - 1]:
                object.QIN[117][I] *= 100
            RAT = GasUtil.CALQINP(EN, NSNG12, YSG12M, XSNG12, 1) * 1e18
        if EN > 3.0 * object.EIN[117]:
            if object.NANISO == 1:
                object.PEQIN[117][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[117][I] = object.PEQEL[1][I - IOFFN[117]]

        # O3 1PI (V=0-3)   F=0.0828
        object.QIN[118][I] = 0.0
        object.PEQIN[118][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[118][I] = 0.
        if EN > object.EIN[118]:
            object.QIN[118][I] = GasUtil.CALQINBEF(EN, NSNG13, YSNG13, XSNG13, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                   BBCONST, object.EIN[118], object.E[2], 0.0828)
            if EN <= XSNG13[NSNG13 - 1]:
                object.QIN[118][I] *= 100
            RAT = GasUtil.CALQINP(EN, NSNG13, YSG13M, XSNG13, 1) * 1e18
        if EN > 3.0 * object.EIN[118]:
            if object.NANISO == 1:
                object.PEQIN[118][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[118][I] = object.PEQEL[1][I - IOFFN[118]]

        # C C!  1SIG (SUM V=4-6) (AVERGAE E=14.090)  F=0.139
        object.QIN[119][I] = 0.0
        object.PEQIN[119][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[119][I] = 0.
        if EN > object.EIN[119]:
            object.QIN[119][I] = GasUtil.CALQINBEF(EN, NSNG14, YSNG14, XSNG14, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                   BBCONST, object.EIN[119], object.E[2], 0.1390)
            if EN <= XSNG14[NSNG14 - 1]:
                object.QIN[119][I] *= 100
            RAT = GasUtil.CALQINP(EN, NSNG14, YSG14M, XSNG14, 1) * 1e18
        if EN > 3.0 * object.EIN[119]:
            if object.NANISO == 1:
                object.PEQIN[119][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[119][I] = object.PEQEL[1][I - IOFFN[119]]

        # C C!  1SIG (SUM V=4-6) (AVERGAE E=14.090)  F=0.139
        object.QIN[120][I] = 0.0
        object.PEQIN[120][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[120][I] = 0.
        if EN > object.EIN[120]:
            object.QIN[120][I] = GasUtil.CALQINBEF(EN, NSNG15, YSNG15, XSNG15, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                   BBCONST, object.EIN[120], object.E[2], 0.2650)
            if EN <= XSNG15[NSNG15 - 1]:
                object.QIN[120][I] *= 100
            RAT = GasUtil.CALQINP(EN, NSNG15, YSG15M, XSNG15, 1) * 1e18
        if EN > 3.0 * object.EIN[120]:
            if object.NANISO == 1:
                object.PEQIN[120][I] = 1.5 - RAT
            if object.NANISO == 2:
                object.PEQIN[120][I] = object.PEQEL[1][I - IOFFN[120]]

        # E! 1SIG  ELOSS = 14.36 F = 0.0108
        object.QIN[121][I] = 0.0
        object.PEQIN[121][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[121][I] = 0.0
        if EN > object.EIN[121]:
            object.QIN[121][I] = 0.0108 / (object.EIN[121] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[121])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[121] + object.E[2])
            if object.QIN[121][I] < 0.0:
                object.QIN[121][I] = 0.0
            object.PEQIN[121][I] = object.PEQIN[120][I]

        # E 1PI  ELOSS = 14.45 F = 0.0237
        object.QIN[122][I] = 0.0
        object.PEQIN[122][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[122][I] = 0.0
        if EN > object.EIN[122]:
            object.QIN[122][I] = 0.0237 / (object.EIN[122] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[122])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[122] + object.E[2])
            if object.QIN[122][I] < 0.0:
                object.QIN[122][I] = 0.0
            object.PEQIN[122][I] = object.PEQIN[120][I]

        # SINGLET  ELOSS = 14.839 F = 0.0117
        object.QIN[123][I] = 0.0
        object.PEQIN[123][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[123][I] = 0.0
        if EN > object.EIN[123]:
            object.QIN[123][I] = 0.0117 / (object.EIN[123] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[123])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[123] + object.E[2])
            if object.QIN[123][I] < 0.0:
                object.QIN[123][I] = 0.0
            object.PEQIN[123][I] = object.PEQIN[120][I]

        # SUM  OF HIGH ENERGY SINGLETS ELOSS 15.20EV F = 0.1152
        object.QIN[124][I] = 0.0
        object.PEQIN[124][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[124][I] = 0.0
        if EN > object.EIN[124]:
            object.QIN[124][I] = 0.1152 / (object.EIN[124] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[124])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[124] + object.E[2])
            if object.QIN[124][I] < 0.0:
                object.QIN[124][I] = 0.0
            object.PEQIN[124][I] = object.PEQIN[120][I]

        # SUM NEUTRAL BREAKUP ABOVE IONISATION ENERGY  F=0.160
        object.QIN[125][I] = 0.0
        object.PEQIN[125][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[125][I] = 0.0
        if EN > object.EIN[125]:
            object.QIN[125][I] = 0.1600 / (object.EIN[125] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[125])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * (EN + 2 * object.EIN[125]) / EN
            if object.QIN[125][I] < 0.0:
                object.QIN[125][I] = 0.0
            object.PEQIN[125][I] = object.PEQIN[120][I]

        # SUM NEUTRAL BREAKUP ABOVE IONISATION ENERGY  F=0.160
        object.QIN[126][I] = 0.0
        object.PEQIN[126][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[126][I] = 0.0
        if EN > object.EIN[126]:
            object.QIN[126][I] = 0.090 / (object.EIN[126] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[126])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * (EN + 2 * object.EIN[126]) / EN
            if object.QIN[126][I] < 0.0:
                object.QIN[126][I] = 0.0
            object.PEQIN[126][I] = object.PEQIN[120][I]

        #LOAD BREMSSTRAHLUNG X-SECTIONS
        object.QIN[127][I] = 0.0
        if EN > 1000:
            object.QIN[127][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z7T, EBRM) * 2e-8

        # ROTATIONAL SUM
        SUMR = 0.0
        for J in range(76):
            SUMR += object.QIN[J][I]

        # VIBRATIONAL SUM
        SUMV = 0.0
        for J in range(76, 92):
            SUMV += object.QIN[J][I]

        #EXCITATION SUM
        SUMEX = 0.0
        for J in range(92, 111):
            SUMEX += object.QIN[J][I]

        #EXCITATION SUM
        SUMEX1 = 0.0
        for J in range(111, 127):
            SUMEX1 += object.QIN[J][I]
        # GET CORRECT ELASTIC XSECTION BY SUBTRACTION OF ROTATION
        object.Q[1][I] -= SUMR

        if object.Q[1][I] < 0.0:
            # FOR VERY HIGH TEMPERATURES SOMETIMES SUMR BECOMES LARGER THAN
            # THE ELASTIC+ROT (ONLY IN FIRST TWO ENERGY BINS) FIX GT 0
            object.Q[1][I] = 0.95e-16

        object.Q[0][I] = object.Q[1][I] + object.Q[4][I] + object.QION[1][I] + SUMR + SUMV + SUMEX + SUMEX1

    for I in range(1,128):
        J = 128 - I - 1
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
    if object.NIN < 77:
        object.NIN = 77
    return
