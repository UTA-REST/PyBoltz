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
cdef void Gas5(Gas* object):
    """
    This function is used to calculate the needed momentum cross sections for Neon gas.
    """
    gd = np.load('gases.npy').item()
    cdef double XEN[125], YXSEC[125], XEL[120], YEL[120], XEPS[196], YEPS[196], XION[74], YION[74], YINC[74], YIN1[74]
    cdef double XIN2[49], YIN2[49], XIN3[41], YIN3[41], XKSH[99], YKSH[99], X1S5[111], Y1S5[111], X1S4[137], Y1S4[137], X1S3[117], Y1S3[117]
    cdef double X1S2[119], Y1S2[119], X2P10[73], Y2P10[73], X2P9[70], Y2P9[70], X2P8[72], Y2P8[72], X2P7[65], Y2P7[65], X2P6[59], Y2P6[59]
    cdef double X2P5[63], Y2P5[63], X2P4[66], Y2P4[66], X2P3[62], Y2P3[62], X2P2[62], Y2P2[62], X2P1[59], Y2P1[59], X2S5[19], Y2S5[19]
    cdef double X2S3[19], Y2S3[19], X3D6[12], Y3D6[12], X3D4P[12], Y3D4P[12], X3D4[12], Y3D4[12], X3D3[12], Y3D3[12], X3D1PP[12], Y3D1PP[12]
    cdef double X3D1P[12], Y3D1P[12], X3S1PPPP[12], Y3S1PPPP[12], X3S1PPP[12], Y3S1PPP[12], X3S1PP[12], Y3S1PP[12], X3P106[16]
    cdef double Y3P106[16], X3P52[16], Y3P52[16], X3P1[16], Y3P1[16], Z10T[25], EBRM[25]
    cdef int IOFFN[45], IOFFION[10], i, j, I, J, NL, NBREM
    cdef double CONST, EMASS2, API, A0, RY, BBCONST, AM2, C, AUGK
    # BORN BETHE VALUES FOR IONISATION
    CONST = 1.873884e-20
    EMASS2 = <float>(1021997.804)
    API = acos(-1)
    A0 = 0.52917720859e-8
    RY = <float>(13.60569193)
    BBCONST = 16.0 * API * A0 * A0 * RY * RY / EMASS2
    AM2 = <float>(1.69)
    C = <float>(17.80)
    #AVERAGE AUGER EMISSION FOR EACH SHELL
    AUGK = <float>(1.99)

    object.NION = 4
    object.NATT = 1
    object.NIN = 45
    object.NNULL = 0

    NBREM = 25
    XEN = gd['gas5/XEN']
    YXSEC = gd['gas5/YXSEC']
    XEL = gd['gas5/XEL']
    YEL = gd['gas5/YEL']
    XEPS = gd['gas5/XEPS']
    YEPS = gd['gas5/YEPS']
    XION = gd['gas5/XION']
    YION = gd['gas5/YION']
    YINC = gd['gas5/YINC']
    YIN1 = gd['gas5/YIN1']
    XIN2 = gd['gas5/XIN2']
    YIN2 = gd['gas5/YIN2']
    XIN3 = gd['gas5/XIN3']
    YIN3 = gd['gas5/YIN3']
    XKSH = gd['gas5/XKSH']
    YKSH = gd['gas5/YKSH']
    X1S5 = gd['gas5/X1S5']
    Y1S5 = gd['gas5/Y1S5']
    X1S4 = gd['gas5/X1S4']
    Y1S4 = gd['gas5/Y1S4']
    X1S3 = gd['gas5/X1S3']
    Y1S3 = gd['gas5/Y1S3']
    X1S2 = gd['gas5/X1S2']
    Y1S2 = gd['gas5/Y1S2']
    X2P10 = gd['gas5/X2P10']
    Y2P10 = gd['gas5/Y2P10']
    X2P9 = gd['gas5/X2P9']
    Y2P9 = gd['gas5/Y2P9']
    X2P8 = gd['gas5/X2P8']
    Y2P8 = gd['gas5/Y2P8']
    X2P7 = gd['gas5/X2P7']
    Y2P7 = gd['gas5/Y2P7']
    X2P6 = gd['gas5/X2P6']
    Y2P6 = gd['gas5/Y2P6']
    X2P5 = gd['gas5/X2P5']
    Y2P5 = gd['gas5/Y2P5']
    X2P4 = gd['gas5/X2P4']
    Y2P4 = gd['gas5/Y2P4']
    X2P3 = gd['gas5/X2P3']
    Y2P3 = gd['gas5/Y2P3']
    X2P2 = gd['gas5/X2P2']
    Y2P2 = gd['gas5/Y2P2']
    X2P1 = gd['gas5/X2P1']
    Y2P1 = gd['gas5/Y2P1']
    X2S5 = gd['gas5/X2S5']
    Y2S5 = gd['gas5/Y2S5']
    X2S3 = gd['gas5/X2S3']
    Y2S3 = gd['gas5/Y2S3']
    X3D6 = gd['gas5/X3D6']
    Y3D6 = gd['gas5/Y3D6']
    X3D4P = gd['gas5/X3D4P']
    Y3D4P = gd['gas5/Y3D4P']
    X3D4 = gd['gas5/X3D4']
    Y3D4 = gd['gas5/Y3D4']
    X3D3 = gd['gas5/X3D3']
    Y3D3 = gd['gas5/Y3D3']
    X3D1PP = gd['gas5/X3D1PP']
    Y3D1PP = gd['gas5/Y3D1PP']
    X3D1P = gd['gas5/X3D1P']
    Y3D1P = gd['gas5/Y3D1P']
    X3S1PPPP = gd['gas5/X3S1PPPP']
    Y3S1PPPP = gd['gas5/Y3S1PPPP']
    X3S1PPP = gd['gas5/X3S1PPP']
    Y3S1PPP = gd['gas5/Y3S1PPP']
    X3S1PP = gd['gas5/X3S1PP']
    Y3S1PP = gd['gas5/Y3S1PP']
    X3P106 = gd['gas5/X3P106']
    Y3P106 = gd['gas5/Y3P106']
    X3P52 = gd['gas5/X3P52']
    Y3P52 = gd['gas5/Y3P52']
    X3P1 = gd['gas5/X3P1']
    Y3P1 = gd['gas5/Y3P1']
    Z10T = gd['gas5/Z10T']
    EBRM = gd['gas5/EBRM']
    object.EIN = gd['gas5/EIN']
    for J in range(6):
        object.KEL[J] = object.WhichAngularModel
    for J in range(object.NIN):
        object.KIN[J] = object.WhichAngularModel
    cdef int NEL, NDATA, NEPSI, NIOND, NION2, NION3, NKSH, N1S5, N1S4, N1S3, N1S2, N2P10, N2P9, N2P8, N2P7, N2P6, N2P5, N2P4, N2P3, N2P2
    cdef int N2P1, N2S5, N2S3, N3D6, N3D4P, N3D4, N3D3, N3D1PP, N3D1P, N3S1PPPP, N3S1PPP, N3S1PP, N3P106, N3P52, N3P1

    NEL = 120
    NDATA = 125
    NEPSI = 196
    NIOND = 74
    NION2 = 49
    NION3 = 41
    NKSH = 99
    N1S5 = 111
    N1S4 = 137
    N1S3 = 117
    N1S2 = 119
    N2P10 = 73
    N2P9 = 70
    N2P8 = 72
    N2P7 = 65
    N2P6 = 59
    N2P5 = 63
    N2P4 = 66
    N2P3 = 62
    N2P2 = 62
    N2P1 = 59
    N2S5 = 19
    N2S3 = 19
    N3D6 = 12
    N3D4P = 12
    N3D4 = 12
    N3D3 = 12
    N3D1PP = 12
    N3D1P = 12
    N3S1PPPP = 12
    N3S1PPP = 12
    N3S1PP = 12
    N3P106 = 16
    N3P52 = 16
    N3P1 = 16

    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, EOBY[5]

    object.E = [0.0, 1.0, <float>(21.56454), 0.0, 0.0, <float>(19.5)]
    object.E[1] = <float>(2.0) * EMASS / (<float>(20.1797) * AMU)

    EOBY[0:4] = [17.4, 36, 73, 500]
    object.EION[0:4] = [<float>(21.56454), <float>(62.5275), <float>(125.9508), <float>(870.2)]
    object.NC0[0:4] = [0, 1, 2, 2]
    object.EC0[0:4] = [0.0, 5.0, 10.0, <float>(806.6)]
    object.WK[0:4] = [0.0, 0.0, 0.0, 0.015]
    object.EFL[0:4] = [0.0, 0.0, 0.0, 849]
    object.NG1[0:4] = [0, 0, 0, 2]
    object.NG2[0:4] = [0, 0, 0, 1]
    object.EG1[0:4] = [0.0, 0.0, 0.0, 801]
    object.EG2[0:4] = [0.0, 0.0, 0.0, 5.0]

    for j in range(0, object.NION):
        for i in range(0, 4000):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i
                break
    for NL in range(object.NIN):
        object.PenningFraction[0][NL] = 0.5
        object.PenningFraction[1][NL] = 1
        object.PenningFraction[2][NL] = 1

    for NL in range(object.NIN):
        for i in range(4000):
            if object.EG[i] > object.EIN[NL]:
                IOFFN[NL] = i
                break
    cdef int LMAX
    cdef double APOL, AA, DD, FF, A1, B1, A2, EN, GAMMA1, GAMMA2, BETA, BETA2, AK, AK2, AK3, AK4
    cdef double AK5, AN0, AN1, AN2,  ANHIGH, SUM, SIGEL, ANLOW, SUMI, QELA, QMOM, PQ[3], QCORR, QINEL,
    # PARAMETERS OF PHASE SHIFT ANALYSIS
    APOL = <float>(2.672)
    LMAX = 100
    AA = <float>(0.2135)
    DD = <float>(3.86)
    FF = <float>(-2.656)
    A1 = <float>(1.846)
    B1 = <float>(3.29)
    A2 = <float>(-0.037)
    object.EnergySteps = 4000

    for I in range(object.EnergySteps):
        EN = object.EG[I]
        if EN > object.EIN[0]:
            GAMMA1 = (EMASS2 + 2 * EN) / EMASS2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = sqrt(1.0 - 1.0 / GAMMA2)
            BETA2 = BETA * BETA
        if EN == 0:
            QELA = 0.161e-16
            QMOM = 0.161e-16
        elif EN <= 1:
            AK = sqrt(EN / object.ARY)
            AK2 = AK * AK
            AK3 = AK2 * AK
            AK4 = AK3 * AK
            AK5 = AK4 * AK
            AN0 = -AA * AK * (1.0 + (4.0 * APOL / 3.0) * AK2 * log(AK)) - (API * APOL / 3.0) * AK2 + DD * AK3 + FF * AK4
            AN1 = ((API / 15.0) * APOL * AK2 - A1 * AK3) / (1.0 + B1 * AK2)
            AN2 = API * APOL * AK2 / 105.0 - A2 * AK5
            ANHIGH = AN2
            SUM = (sin(AN0 - AN1)) ** 2
            SUM = SUM + 2.0 * (sin(AN1 - AN2)) ** 2
            SIGEL = (sin(AN0)) ** 2 + 3.0 * (sin(AN1)) ** 2
            for J in range(2, LMAX):
                ANLOW = ANHIGH
                ANHIGH = API * APOL * AK2 / ((2. * J + 5.0) * (2. * J + 3.0) * (2. * J + 1.0))
                SUMI = 6.0 / ((2.0 * J + 5.0) * (2.0 * J + 3.0) * (2.0 * J + 1.0) * (2.0 * J - 1.0))
                SUM = SUM + (J + 1.0) * (sin(API * APOL * AK2 * SUMI)) ** 2
                SIGEL = SIGEL + (2.0 * J + 1.0) * (sin(ANLOW)) ** 2
            QELA = SIGEL * 4.0 * object.PIR2 / AK2
            QMOM = SUM * 4.0 * object.PIR2 / AK2
        else:
            QELA = GasUtil.CALQIONREG(EN, NEL, YEL, XEL)
            QMOM = GasUtil.CALQIONREG(EN, NDATA, YXSEC, XEN)
        PQ[1] = 0.5 + (QELA - QMOM) / QELA
        PQ[0] = 0.5
        PQ[2] = GasUtil.CALPQ3(EN, NEPSI, YEPS, XEPS)
        PQ[2] = 1-PQ[2]
        object.PEQEL[1][I] = PQ[object.WhichAngularModel]
        object.Q[1][I] = QELA
        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM

        #IONISATION FOR CHARGE STATE =1
        object.QION[0][I] = 0.0
        object.PEQION[0][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[0][I] = 0.0
        if EN > object.EION[0]:
            object.QION[0][I] = GasUtil.CALQIONX(EN, NIOND, YIN1, XION, BETA2, <float>(0.9594), CONST, object.DEN[I], C, AM2)
        #USE AAnisotropicDetectedTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
        #ENERGIES ABOVE 2 * IONISATION ENERGY
        # AAnisotropicDetectedTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFF SET BY
        # THE IONISATION ENERGY
            if EN > 2 * object.EION[0]:
                object.PEQION[0][I] = object.PEQEL[1][I - IOFFION[0]]

        #IONISATION FOR CHARGE STATE =2
        object.QION[1][I] = 0.0
        object.PEQION[1][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[1][I] = 0.0
        if EN > object.EION[1]:
            object.QION[1][I] = GasUtil.CALQIONX(EN, NION2, YIN2, XIN2, BETA2, <float>(0.0388), CONST, object.DEN[I], C, AM2)
            if EN > 2 * object.EION[1]:
                object.PEQION[1][I] = object.PEQEL[1][I - IOFFION[1]]

        #IONISATION FOR CHARGE STATE =3
        object.QION[2][I] = 0.0
        object.PEQION[2][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[2][I] = 0.0
        if EN > object.EION[2]:
            object.QION[2][I] = GasUtil.CALQIONX(EN, NION3, YIN3, XIN3, BETA2, <float>(0.00215), CONST, object.DEN[I], C, AM2)
            if EN > 2 * object.EION[2]:
                object.PEQION[2][I] = object.PEQEL[1][I - IOFFION[2]]

        # K-SHELL IONISATION
        object.QION[3][I] = 0.0
        object.PEQION[3][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[3][I] = 0.0
        if EN > object.EION[3]:
            object.QION[3][I] = GasUtil.CALQIONREG(EN, NKSH, YKSH, XKSH)
            object.PEQION[3][I] = object.PEQEL[1][I - IOFFION[3]]

        # ATTACHMENT
        object.Q[3][I] = 0.0
        object.QATT[0][I] = 0.0

        #COUNTING IONISATION
        object.Q[4][I] = 0.0
        object.PEQEL[4][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQEL[4][I] = 0.0
            if EN > object.E[2]:
                object.Q[4][I] = GasUtil.CALQIONX(EN, NIOND, YINC, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)

        # CORRECTION TO CHARGE STATE 1 2 AND 3 X-SECTION FOR K SHELL
        # GIVES TOTAL IONISATION EQUAL TO OSCILLATOR SUM

        if object.Q[4][I] == 0.0:
            QCORR = 1.0
        else:
            QCORR = (object.Q[4][I] - object.QION[3][I]) / object.Q[4][I]
        object.QION[0][I] *= QCORR
        object.QION[1][I] *= QCORR
        object.QION[2][I] *= QCORR

        object.Q[5][I] = 0.0

        for NL in range(object.NIN + 1):
            object.QIN[NL][I] = 0.0
            object.PEQIN[NL][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEQIN[NL][I] = 0.0

        #1S5 METASTABLE LEVEL
        if EN > object.EIN[0]:
            object.QIN[0][I] = GasUtil.CALQINP(EN, N1S5, Y1S5, X1S5, 3)
            if EN > 2 * object.EIN[0]:
                object.PEQIN[0][I] = object.PEQEL[1][I - IOFFN[0]]

        #1S4 RESONANCE LEVEL  F=0.0118
        if EN > object.EIN[1]:
            object.QIN[1][I] = GasUtil.CALQINBEF(EN, EN,N1S4, Y1S4, X1S4, BETA2, GAMMA2, EMASS2, object.DEN[I], BBCONST,
                                                 object.EIN[1], object.E[2], <float>(0.0118))
            object.QIN[1][I] = abs(object.QIN[1][I])
            if EN > 2 * object.EIN[1]:
                object.PEQIN[1][I] = object.PEQEL[1][I - IOFFN[1]]

        #1S3 METASTABLE LEVEL
        if EN > object.EIN[2]:
            object.QIN[2][I] = GasUtil.CALQINP(EN, N1S3, Y1S3, X1S3, 3)
            if EN > 2 * object.EIN[2]:
                object.PEQIN[2][I] = object.PEQEL[1][I - IOFFN[2]]

        #1S2 RESONANCE LEVEL  F=0.159
        if EN > object.EIN[3]:
            object.QIN[3][I] = GasUtil.CALQINBEF(EN, EN,N1S2, Y1S2, X1S2, BETA2, GAMMA2, EMASS2, object.DEN[I], BBCONST,
                                                 object.EIN[3], object.E[2], <float>(0.159))
            object.QIN[3][I] = abs(object.QIN[3][I])
            if EN > 2 * object.EIN[3]:
                object.PEQIN[3][I] = object.PEQEL[1][I - IOFFN[3]]

        #2P10
        if EN > object.EIN[4]:
            object.QIN[4][I] = GasUtil.CALQINP(EN, N2P10, Y2P10, X2P10, 2)
            if EN > 2 * object.EIN[4]:
                object.PEQIN[4][I] = object.PEQEL[1][I - IOFFN[4]]

        #2P9
        if EN > object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQINP(EN, N2P9, Y2P9, X2P9, 2)
            if EN > 2 * object.EIN[5]:
                object.PEQIN[5][I] = object.PEQEL[1][I - IOFFN[5]]

        #2P8
        if EN > object.EIN[6]:
            object.QIN[6][I] = GasUtil.CALQINP(EN, N2P8, Y2P8, X2P8, 1)
            if EN > 2 * object.EIN[6]:
                object.PEQIN[6][I] = object.PEQEL[1][I - IOFFN[6]]

        #2P7
        if EN > object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQINP(EN, N2P7, Y2P7, X2P7, 2)
            if EN > 2 * object.EIN[7]:
                object.PEQIN[7][I] = object.PEQEL[1][I - IOFFN[7]]

        #2P6
        if EN > object.EIN[8]:
            object.QIN[8][I] = GasUtil.CALQINP(EN, N2P6, Y2P6, X2P6, 1)
            if EN > 2 * object.EIN[8]:
                object.PEQIN[8][I] = object.PEQEL[1][I - IOFFN[8]]

        #2P5
        if EN > object.EIN[9]:
            object.QIN[9][I] = GasUtil.CALQINP(EN, N2P5, Y2P5, X2P5, 2)
            if EN > 2 * object.EIN[9]:
                object.PEQIN[9][I] = object.PEQEL[1][I - IOFFN[9]]

        #2P4
        if EN > object.EIN[10]:
            object.QIN[10][I] = GasUtil.CALQINP(EN, N2P4, Y2P4, X2P4, 1)
            if EN > 2 * object.EIN[10]:
                object.PEQIN[10][I] = object.PEQEL[1][I - IOFFN[10]]

        #2P3
        if EN > object.EIN[11]:
            object.QIN[11][I] = GasUtil.CALQINP(EN, N2P3, Y2P3, X2P3, 1)
            if EN > 2 * object.EIN[11]:
                object.PEQIN[11][I] = object.PEQEL[1][I - IOFFN[11]]

        #2P2
        if EN > object.EIN[12]:
            object.QIN[12][I] = GasUtil.CALQINP(EN, N2P2, Y2P2, X2P2, 2)
            if EN > 2 * object.EIN[12]:
                object.PEQIN[12][I] = object.PEQEL[1][I - IOFFN[12]]

        #2P1
        if EN > object.EIN[13]:
            object.QIN[13][I] = GasUtil.CALQINP(EN, N2P1, Y2P1, X2P1, 1)
            if EN > 2 * object.EIN[13]:
                object.PEQIN[13][I] = object.PEQEL[1][I - IOFFN[13]]

        #2S5
        if EN > object.EIN[14]:
            object.QIN[14][I] = GasUtil.CALQINP(EN, N2S5, Y2S5, X2S5, 2)
            if EN > 2 * object.EIN[14]:
                object.PEQIN[14][I] = object.PEQEL[1][I - IOFFN[14]]

        #2S4  BEF SCALING
        if EN > object.EIN[15]:
            object.QIN[15][I] = <float>(0.0128) / (object.EIN[15] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[15])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[15] + object.E[2])
            object.QIN[15][I] = abs(object.QIN[15][I])
            if EN > 2 * object.EIN[15]:
                object.PEQIN[15][I] = object.PEQEL[1][I - IOFFN[15]]

        #2S3
        if EN > object.EIN[16]:
            object.QIN[16][I] = GasUtil.CALQINP(EN, N2S3, Y2S3, X2S3, 2)
            if EN > 2 * object.EIN[16]:
                object.PEQIN[16][I] = object.PEQEL[1][I - IOFFN[16]]

        #2S2  BEF SCALING
        if EN > object.EIN[17]:
            object.QIN[17][I] = <float>(0.0166) / (object.EIN[17] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[17])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[17] + object.E[2])
            object.QIN[17][I] = abs(object.QIN[17][I])
            if EN > 2 * object.EIN[17]:
                object.PEQIN[17][I] = object.PEQEL[1][I - IOFFN[17]]

        #3D6
        if EN > object.EIN[18]:
            object.QIN[18][I] = GasUtil.CALQINP(EN, N3D6, Y3D6, X3D6, 2)
            if EN > 2 * object.EIN[18]:
                object.PEQIN[18][I] = object.PEQEL[1][I - IOFFN[18]]

        #3D5  BEF SCALING
        if EN > object.EIN[19]:
            object.QIN[19][I] = <float>(0.0048) / (object.EIN[19] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[19])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[19] + object.E[2])
            object.QIN[19][I] = abs(object.QIN[19][I])
            if EN > 2 * object.EIN[19]:
                object.PEQIN[19][I] = object.PEQEL[1][I - IOFFN[19]]

        #3D4!
        if EN > object.EIN[20]:
            object.QIN[20][I] = GasUtil.CALQINP(EN, N3D4P, Y3D4P, X3D4P, 2)
            if EN > 2 * object.EIN[20]:
                object.PEQIN[20][I] = object.PEQEL[1][I - IOFFN[20]]

        #3D4
        if EN > object.EIN[21]:
            object.QIN[21][I] = GasUtil.CALQINP(EN, N3D4, Y3D4, X3D4, 2)
            if EN > 2 * object.EIN[21]:
                object.PEQIN[21][I] = object.PEQEL[1][I - IOFFN[21]]

        #3D3
        if EN > object.EIN[22]:
            object.QIN[22][I] = GasUtil.CALQINP(EN, N3D3, Y3D3, X3D3, 2)
            if EN > 2 * object.EIN[22]:
                object.PEQIN[22][I] = object.PEQEL[1][I - IOFFN[22]]

        #3D2  BEF SCALING
        if EN > object.EIN[23]:
            object.QIN[23][I] = <float>(0.0146) / (object.EIN[23] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[23])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[23] + object.E[2])
            object.QIN[23][I] = abs(object.QIN[23][I])
            if EN > 2 * object.EIN[23]:
                object.PEQIN[23][I] = object.PEQEL[1][I - IOFFN[23]]

        #3D1!!
        if EN > object.EIN[24]:
            object.QIN[24][I] = GasUtil.CALQINP(EN, N3D1PP, Y3D1PP, X3D1PP, 2)
            if EN > 2 * object.EIN[24]:
                object.PEQIN[24][I] = object.PEQEL[1][I - IOFFN[24]]

        #3D1!
        if EN > object.EIN[25]:
            object.QIN[25][I] = GasUtil.CALQINP(EN, N3D1P, Y3D1P, X3D1P, 2)
            if EN > 2 * object.EIN[25]:
                object.PEQIN[25][I] = object.PEQEL[1][I - IOFFN[25]]

        #3S1!!!!
        if EN > object.EIN[26]:
            object.QIN[26][I] = GasUtil.CALQINP(EN, N3S1PPPP, Y3S1PPPP, X3S1PPPP, 2)
            if EN > 2 * object.EIN[26]:
                object.PEQIN[26][I] = object.PEQEL[1][I - IOFFN[26]]

        #3S1!!!
        if EN > object.EIN[27]:
            object.QIN[27][I] = GasUtil.CALQINP(EN, N3S1PPP, Y3S1PPP, X3S1PPP, 2)
            if EN > 2 * object.EIN[27]:
                object.PEQIN[27][I] = object.PEQEL[1][I - IOFFN[27]]

        #3S1!!
        if EN > object.EIN[28]:
            object.QIN[28][I] = GasUtil.CALQINP(EN, N3S1PP, Y3S1PP, X3S1PP, 2)
            if EN > 2 * object.EIN[28]:
                object.PEQIN[28][I] = object.PEQEL[1][I - IOFFN[28]]

        #3S1!  BEF SCALING
        if EN > object.EIN[29]:
            object.QIN[29][I] = <float>(0.00676) / (object.EIN[29] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[29])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[29] + object.E[2])
            object.QIN[29][I] = abs(object.QIN[29][I])
            if EN > 2 * object.EIN[29]:
                object.PEQIN[29][I] = object.PEQEL[1][I - IOFFN[29]]

        #SUM 3P10 -- 3P6
        if EN > object.EIN[30]:
            object.QIN[30][I] = GasUtil.CALQINP(EN, N3P106, Y3P106, X3P106, 1.5)
            if EN > 2 * object.EIN[30]:
                object.PEQIN[30][I] = object.PEQEL[1][I - IOFFN[30]]

        #SUM 3P5 -- 3P2
        if EN > object.EIN[31]:
            object.QIN[31][I] = GasUtil.CALQINP(EN, N3P52, Y3P52, X3P52, 1.5)
            if EN > 2 * object.EIN[31]:
                object.PEQIN[31][I] = object.PEQEL[1][I - IOFFN[31]]

        #3P1
        if EN > object.EIN[32]:
            object.QIN[32][I] = GasUtil.CALQINP(EN, N3P1, Y3P1, X3P1, 1)
            if EN > 2 * object.EIN[32]:
                object.PEQIN[32][I] = object.PEQEL[1][I - IOFFN[32]]

        #3S4  BEF SCALING
        if EN > object.EIN[33]:
            object.QIN[33][I] = <float>(0.00635) / (object.EIN[33] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[33])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[33] + object.E[2])
            object.QIN[33][I] = abs(object.QIN[33][I])
            if EN > 2 * object.EIN[33]:
                object.PEQIN[33][I] = object.PEQEL[1][I - IOFFN[33]]

        #3S2  BEF SCALING
        if EN > object.EIN[34]:
            object.QIN[34][I] = <float>(0.00440) / (object.EIN[34] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[34])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[34] + object.E[2])
            object.QIN[34][I] = abs(object.QIN[34][I])
            if EN > 2 * object.EIN[34]:
                object.PEQIN[34][I] = object.PEQEL[1][I - IOFFN[34]]

        #4D5  BEF SCALING
        if EN > object.EIN[35]:
            object.QIN[35][I] = <float>(0.00705) / (object.EIN[35] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[35])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[35] + object.E[2])
            object.QIN[35][I] = abs(object.QIN[35][I])
            if EN > 2 * object.EIN[35]:
                object.PEQIN[35][I] = object.PEQEL[1][I - IOFFN[35]]

        #4D2  BEF SCALING
        if EN > object.EIN[36]:
            object.QIN[36][I] = <float>(0.00235) / (object.EIN[36] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[36])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[36] + object.E[2])
            object.QIN[36][I] = abs(object.QIN[36][I])
            if EN > 2 * object.EIN[36]:
                object.PEQIN[36][I] = object.PEQEL[1][I - IOFFN[36]]

        #4S1!  BEF SCALING
        if EN > object.EIN[37]:
            object.QIN[37][I] = <float>(0.00435) / (object.EIN[37] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[37])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[37] + object.E[2])
            object.QIN[37][I] = abs(object.QIN[37][I])
            if EN > 2 * object.EIN[37]:
                object.PEQIN[37][I] = object.PEQEL[1][I - IOFFN[37]]

        #4S4  BEF SCALING
        if EN > object.EIN[38]:
            object.QIN[38][I] = <float>(0.00325) / (object.EIN[38] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[38])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[38] + object.E[2])
            object.QIN[38][I] = abs(object.QIN[38][I])
            if EN > 2 * object.EIN[38]:
                object.PEQIN[38][I] = object.PEQEL[1][I - IOFFN[38]]

        #5D5 BEF SCALING
        if EN > object.EIN[39]:
            object.QIN[39][I] = <float>(0.00383) / (object.EIN[39] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[39])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[39] + object.E[2])
            object.QIN[39][I] = abs(object.QIN[39][I])
            if EN > 2 * object.EIN[39]:
                object.PEQIN[39][I] = object.PEQEL[1][I - IOFFN[39]]

        #5D2 BEF SCALING
        if EN > object.EIN[40]:
            object.QIN[40][I] = <float>(0.00127) / (object.EIN[40] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[40])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[40] + object.E[2])
            object.QIN[40][I] = abs(object.QIN[40][I])
            if EN > 2 * object.EIN[40]:
                object.PEQIN[40][I] = object.PEQEL[1][I - IOFFN[40]]

        #4S2 BEF SCALING
        if EN > object.EIN[41]:
            object.QIN[41][I] = <float>(0.00165) / (object.EIN[41] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[41])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[41] + object.E[2])
            object.QIN[41][I] = abs(object.QIN[41][I])
            if EN > 2 * object.EIN[41]:
                object.PEQIN[41][I] = object.PEQEL[1][I - IOFFN[41]]

        #5S1! BEF SCALING
        if EN > object.EIN[42]:
            object.QIN[42][I] =<float>(0.00250) / (object.EIN[42] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[42])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[42] + object.E[2])
            object.QIN[42][I] = abs(object.QIN[42][I])
            if EN > 2 * object.EIN[42]:
                object.PEQIN[42][I] = object.PEQEL[1][I - IOFFN[42]]

        #SUM HIGHER RESONANCE S STATES
        if EN > object.EIN[43]:
            object.QIN[43][I] = <float>(0.00962) / (object.EIN[43] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[43])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[43] + object.E[2])
            object.QIN[43][I] = abs(object.QIN[43][I])
            if EN > 2 * object.EIN[43]:
                object.PEQIN[43][I] = object.PEQEL[1][I - IOFFN[43]]

        #SUM HIGHER RESONANCE S STATES
        if EN > object.EIN[44]:
            object.QIN[44][I] = <float>(0.01695) / (object.EIN[44] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[44])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[44] + object.E[2])
            object.QIN[44][I] = abs(object.QIN[44][I])
            if EN > 2 * object.EIN[44]:
                object.PEQIN[44][I] = object.PEQEL[1][I - IOFFN[44]]

        QINEL = 0
        for J in range(object.NIN):
            QINEL += object.QIN[J][I]

        object.Q[0][I] = QELA + object.QION[0][I] + object.QION[1][I] + object.QION[2][I] + object.QION[3][I] + QINEL

    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
    return
