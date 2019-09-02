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
cdef void Gas8(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for CH4 gas.
    """
    gd = np.load('gases.npy').item()
    cdef double XEN[153], YELM[153], YELT[153], YEPS[153], XATT[6], YATT[6], XVBV4[26], YVBV4[26], XVBV2[29], YVBV2[29], XVBV1[30], YVBV1[30],
    cdef double XVBV3[25], YVBV3[25], XVBH1[14], YVBH1[14], XVBH2[14], YVBH2[14], XION[70], YION[70], YINC[70], XINF[70], YINF[70], XINF1[68], YINF1[68],
    cdef double XINF2[66], YINF2[66], XINF3[53], YINF3[53], XINF4[51], YINF4[51], XINF5[50], YINF5[50], XINF6[48], YINF6[48], XINPP[49], YINPP[49],
    cdef double XDET[9], YDET[9], XTR1[12], YTR1[12], XTR2[11], YTR2[11], XTR3[11], YTR3[11], XCHD[32], YCHD[32], XCHB[35], YCHB[35], XHAL[34], YHAL[34],
    cdef double XHBE[34], YHBE[34], XKSH[83], YKSH[83], Z1T[25], Z6T[25], EBRM[25],
    cdef int IOFFN[34], IOFFION[10]
    XEN = gd['gas8/XEN']
    YELM = gd['gas8/YELM']
    YELT = gd['gas8/YELT']
    YEPS = gd['gas8/YEPS']
    XATT = gd['gas8/XATT']
    YATT = gd['gas8/YATT']
    XVBV4 = gd['gas8/XVBV4']
    YVBV4 = gd['gas8/YVBV4']
    XVBV2 = gd['gas8/XVBV2']
    YVBV2 = gd['gas8/YVBV2']
    XVBV1 = gd['gas8/XVBV1']
    YVBV1 = gd['gas8/YVBV1']
    XVBV3 = gd['gas8/XVBV3']
    YVBV3 = gd['gas8/YVBV3']
    XVBH1 = gd['gas8/XVBH1']
    YVBH1 = gd['gas8/YVBH1']
    XVBH2 = gd['gas8/XVBH2']
    YVBH2 = gd['gas8/YVBH2']
    XION = gd['gas8/XION']
    YION = gd['gas8/YION']
    YINC = gd['gas8/YINC']
    XINF = gd['gas8/XINF']
    YINF = gd['gas8/YINF']
    XINF1 = gd['gas8/XINF1']
    YINF1 = gd['gas8/YINF1']
    XINF2 = gd['gas8/XINF2']
    YINF2 = gd['gas8/YINF2']
    XINF3 = gd['gas8/XINF3']
    YINF3 = gd['gas8/YINF3']
    XINF4 = gd['gas8/XINF4']
    YINF4 = gd['gas8/YINF4']
    XINF5 = gd['gas8/XINF5']
    YINF5 = gd['gas8/YINF5']
    XINF6 = gd['gas8/XINF6']
    YINF6 = gd['gas8/YINF6']
    XINPP = gd['gas8/XINPP']
    YINPP = gd['gas8/YINPP']
    XDET = gd['gas8/XDET']
    YDET = gd['gas8/YDET']
    XTR1 = gd['gas8/XTR1']
    YTR1 = gd['gas8/YTR1']
    XTR2 = gd['gas8/XTR2']
    YTR2 = gd['gas8/YTR2']
    XTR3 = gd['gas8/XTR3']
    YTR3 = gd['gas8/YTR3']
    XCHD = gd['gas8/XCHD']
    YCHD = gd['gas8/YCHD']
    XCHB = gd['gas8/XCHB']
    YCHB = gd['gas8/YCHB']
    XHAL = gd['gas8/XHAL']
    YHAL = gd['gas8/YHAL']
    XHBE = gd['gas8/XHBE']
    YHBE = gd['gas8/YHBE']
    XKSH = gd['gas8/XKSH']
    YKSH = gd['gas8/YKSH']
    Z1T = gd['gas8/Z1T']
    Z6T = gd['gas8/Z6T']
    EBRM = gd['gas8/EBRM']

    cdef double A0, RY, CONST, EMASS2, API, BBCONST, AM2, C, AM2EXC, CEXC, RAT, DEGV4, DEGV3, DEGV2, DEGV1
    cdef int J, I, i, j, NBREM, NDATA, NVIBV4, NVIBV2, NVIBV1, NVIBV3, NVIBH1, NVIBH2, NIOND, NIONF, NIONF1, NIONF2
    cdef int NIONF3, NIONF4, NIONF5, NIONF6, NIONPP, NKSH, NATT1, NDET, NTRP1, NTRP2, NTRP3, NCHD, NCHB, NHAL, NHBE
    cdef int NASIZE = 4000
    for J in range(6):
        object.KEL[J] = object.WhichAngularModel
        #SUPERELASTIC, V2 V1 AND HARMONIC VIBRATIONS ASSUMED ISOTROPIC
        object.KIN[J] = 0
    object.KIN[6]=0.0
    object.KIN[7]=0.0
    #V4 AND V3 VIBRATIONS ANISOTROPIC ( CAPITELLI-LONGO)
    object.KIN[1] = 1
    object.KIN[5] = 1
    for J in range(NBREM):
        EBRM[J] = exp(EBRM[J])
    # ANGULAR DISTRIBUTION FOR DISSOCIATIVE EXCITATION IS OKHRIMOVSKYY TYPE
    for J in range(8, object.NIN):
        object.KIN[J] = 2

    #RAT IS MOMENTUM TRANSFER TO TOTAL RATIO FOR VIBRATIONS IN THE
    #RESONANCE REGION AND ALSO FOR THE VIBRATIONS V1 AND V2 .
    #USED DIPOLE ANGULAR DISTRIBUTION FOR V3 AND V4 NEAR THRESHOLD.
    RAT = 1.0
    #BORN BETHE CONSTANTS
    A0 = 0.52917720859e-08
    RY = <float> (13.60569193)
    CONST = 1.873884e-20
    EMASS2 = <float> (1021997.804)
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / EMASS2
    #BORN BETHE VALUES FOR IONISATION
    AM2 = <float> (3.75)
    C = <float> (42.5)
    #BORN BETHE FOR EXCITATION
    AM2EXC = <float> (1.40)
    CEXC = <float> (19.0)
    #ARRAY SIZE
    NASIZE = 4000
    object.NION = 9
    object.NATT = 1
    object.NIN = 34
    object.NNULL = 0
    NBREM = 25

    NDATA = 153
    NVIBV4 = 26
    NVIBV2 = 29
    NVIBV1 = 30
    NVIBV3 = 25
    NVIBH1 = 14
    NVIBH2 = 14
    NIOND = 70
    NIONF = 70
    NIONF1 = 68
    NIONF2 = 66
    NIONF3 = 53
    NIONF4 = 51
    NIONF5 = 50
    NIONF6 = 48
    NIONPP = 49
    NKSH = 83
    NATT1 = 6
    NDET = 9
    NTRP1 = 12
    NTRP2 = 11
    NTRP3 = 11
    NCHD = 32
    NCHB = 35
    NHAL = 34
    NHBE = 34

    #VIBRATIONAL DEGENERACY
    DEGV4 = 3.0
    DEGV2 = 2.0
    DEGV1 = 1.0
    DEGV3 = 3.0
    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27

    object.E = [0.0, 1.0, <float> (12.65), 0.0, 0.0, 0.0]
    object.E[1] = <float>(2.0) * EMASS / (<float> (16.0426) * AMU)
    object.EION[0:9] = [<float> (12.65), <float> (14.25), <float> (15.2), <float> (22.2), <float> (23.5),
                        <float> (25.2), <float> (27.0), <float> (27.9), <float> (285.0)]
    #OPAL BEATY
    cdef double SCLOBY = <float> (0.475), EOBY[9]
    for j in range(9):
        EOBY[J] = object.EION[J] * SCLOBY
    EOBY[8] = object.EION[8] * <float> (0.63)
    for J in range(8):
        object.NC0[J] = 0.0
        object.EC0[J] = 0.0
        object.WK[J] = 0.0
        object.EFL[J] = 0.0
        object.NG1[J] = 0.0
        object.EG1[J] = 0.0
        object.NG2[J] = 0.0
        object.EG2[J] = 0.0
    #DOUBLE CHARGED, 2+ ION STATES (EXTRA ELECTRON)
    object.NC0[6] = 1
    object.EC0[6] = 6.0
    #FLUORESCENCE DATA
    object.NC0[8] = 2
    object.EC0[8] = 253
    object.WK[8] = <float> (0.0026)
    object.EFL[8] = 273
    object.NG1[8] = 1
    object.EG1[8] = 253
    object.NG2[8] = 2
    object.EG2[8] = 5

    # OFFSET ENERGY FOR IONISATION ELECTRON ANGULAR DISTRIBUTION
    for j in range(0, object.NION):
        for i in range(0, NASIZE):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i
                break
    object.EIN = gd['gas8/EIN']

    cdef int NL = 0
    for NL in range(object.NIN):
        for i in range(4000):
            if object.EG[i] > abs(object.EIN[NL]):
                IOFFN[NL] = i
                break
    for NL in range(3):
        for i in range(46):
            object.PENFRA[NL][i] = 0.0

    cdef double APOPV4, APOPV2, APOPGS, APOPSUM, EN, GAMMA1, GAMMA2, BETA, BETA2, QELA, QMOM, PQ[3], YXJ, XNJ, YXJ1, XNJ1, A, B, X2, X1, QSUM, FAC
    cdef double XMT, CON[18], F[18], QSUP, QVIB, QDATT, QSING, QTRIP, QEXC, QTTT, QWINT, QINEL, QIONS
    cdef int FI, CONI
    F[0:18] = [<float> (0.0271), <float> (0.0442), <float> (0.0859), <float> (0.0906), <float> (0.0841),
               <float> (0.1036), <float> (0.1460), <float> (0.1548), <float> (0.1927), <float> (0.1981),
               <float> (0.1628), <float> (0.10930), <float> (0.0628), <float> (0.0297), <float> (0.0074), <float> (0.5),
               <float> (0.0045), <float> (0.0045) ]
    CON[0:18] = [<float> (1.029), <float> (1.027), <float> (1.026), <float> (1.024), <float> (1.023), <float> (1.022),
                 <float> (1.021), <float> (1.020), <float> (1.020), <float> (1.019), <float> (1.018), <float> (1.018),
                 <float> (1.017), <float> (1.016), <float> (1.016), <float> (1), <float> (1.037), <float> (1.034) ]
    #CALC LEVEL POPULATIONS
    APOPV4 = DEGV4 * exp(object.EIN[0] / object.AKT)
    APOPV2 = DEGV2 * exp(object.EIN[2] / object.AKT)
    APOPGS = 1.0
    APOPSUM = APOPGS + APOPV4 + APOPV2
    APOPGS = 1.0 / APOPSUM
    APOPV4 = APOPV4 / APOPSUM
    APOPV2 = APOPV2 / APOPSUM
    #  RENORMALISE GROUND STATE TO ALLOW FOR INCREASED EXCITATION X-SEC
    #  FROM EXCITED VIBRATIONAL STATE ( EXACT FOR TWICE GROUND STATE XSEC)
    APOPGS = 1.0
    object.NSTEP = 4000
    for I in range(object.NSTEP):
        EN = object.EG[I]
        GAMMA1 = (EMASS2 + 2 * EN) / EMASS2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA

        #USE LOG INTERPOLATION FOR ELASTIC
        if EN <= XEN[1]:
            QELA = 26.7e-16
            QMOM = 26.7e-16
            PQ[2] = 0.0
        else:
            for J in range(1, NDATA):
                if EN <= XEN[J]:
                    break
            YXJ = log(YELT[J])
            YXJ1 = log(YELT[J - 1])
            XNJ = log(XEN[J])
            XNJ1 = log(XEN[J - 1])
            A = (YXJ - YXJ1) / (XNJ - XNJ1)
            B = (XNJ1 * YXJ - XNJ * YXJ1) / (XNJ1 - XNJ)
            QELA = exp(A * log(EN) + B) * 1.e-16
            YXJ = log(YELM[J])
            YXJ1 = log(YELM[J - 1])
            A = (YXJ - YXJ1) / (XNJ - XNJ1)
            B = (XNJ1 * YXJ - XNJ * YXJ1) / (XNJ1 - XNJ)
            QMOM = exp(A * log(EN) + B) * 1.e-16
            YXJ = log(YEPS[J])
            YXJ1 = log(YEPS[J - 1])
            A = (YXJ - YXJ1) / (XNJ - XNJ1)
            B = (XNJ1 * YXJ - XNJ * YXJ1) / (XNJ1 - XNJ)
            PQ[2] = exp(A * log(EN) + B)
            #  EPSILON =1-YEPS
            PQ[2] = 1.0e0 - PQ[2]
        PQ[1] = 0.5 + (QELA - QMOM) / QELA
        PQ[0] = 0.5
        object.PEQEL[1][I] = PQ[object.WhichAngularModel]
        object.Q[1][I] = QELA
        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM

        # IONISATION TO CH4 +
        object.QION[0][I] = 0.0
        object.PEQION[0][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[0][I] = 0
        if EN >= object.EION[0]:
            object.QION[0][I] = GasUtil.CALQIONX(EN, NIONF, YINF, XINF, BETA2, <float> (0.4594), CONST, object.DEN[I],
                                                 C, AM2)
            if EN > 2 * object.EION[0]:
                object.PEQION[0][I] = object.PEQEL[1][I - IOFFION[0]]

        # IONISATION TO CH3 +
        object.QION[1][I] = 0.0
        object.PEQION[1][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[1][I] = 0
        if EN >= object.EION[1]:
            object.QION[1][I] = GasUtil.CALQIONX(EN, NIONF1, YINF1, XINF1, BETA2, <float> (0.3716), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.EION[1]:
                object.PEQION[1][I] = object.PEQEL[1][I - IOFFION[1]]

        # IONISATION TO CH2 +
        object.QION[2][I] = 0.0
        object.PEQION[2][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[2][I] = 0
        if EN >= object.EION[2]:
            object.QION[2][I] = GasUtil.CALQIONX(EN, NIONF2, YINF2, XINF2, BETA2, <float> (0.06312), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.EION[2]:
                object.PEQION[2][I] = object.PEQEL[1][I - IOFFION[2]]

        # IONISATION TO H +
        object.QION[3][I] = 0.0
        object.PEQION[3][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[3][I] = 0
        if EN >= object.EION[3]:
            object.QION[3][I] = GasUtil.CALQIONX(EN, NIONF3, YINF3, XINF3, BETA2, <float> (0.0664), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.EION[3]:
                object.PEQION[3][I] = object.PEQEL[1][I - IOFFION[3]]

        # IONISATION TO CH +
        object.QION[4][I] = 0.0
        object.PEQION[4][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[4][I] = 0
        if EN >= object.EION[4]:
            object.QION[4][I] = GasUtil.CALQIONX(EN, NIONF4, YINF4, XINF4, BETA2, <float> (0.02625), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.EION[4]:
                object.PEQION[4][I] = object.PEQEL[1][I - IOFFION[4]]

        # IONISATION TO C +
        object.QION[5][I] = 0.0
        object.PEQION[5][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[5][I] = 0
        if EN >= object.EION[5]:
            object.QION[5][I] = GasUtil.CALQIONX(EN, NIONF5, YINF5, XINF5, BETA2, <float> (0.00798), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.EION[5]:
                object.PEQION[5][I] = object.PEQEL[1][I - IOFFION[5]]

        # IONISATION TO DOUBLY POSITIVE CHARGED FINAL STATES
        object.QION[6][I] = 0.0
        object.PEQION[6][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[6][I] = 0
        if EN >= object.EION[6]:
            object.QION[6][I] = GasUtil.CALQIONX(EN, NIONPP, YINPP, XINPP, BETA2, <float> (0.0095969), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.EION[6]:
                object.PEQION[6][I] = object.PEQEL[1][I - IOFFION[6]]

        # IONISATION TO H2+
        object.QION[7][I] = 0.0
        object.PEQION[7][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[7][I] = 0
        if EN >= object.EION[7]:
            object.QION[7][I] = GasUtil.CALQIONX(EN, NIONF6, YINF6, XINF6, BETA2, <float> (0.00523), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.EION[7]:
                object.PEQION[7][I] = object.PEQEL[1][I - IOFFION[7]]

        # CALCULATE K-SHELL IONISATION
        object.QION[8][I] = 0.0
        object.PEQION[8][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[8][I] = 0
        if EN >= object.EION[8]:
            object.QION[8][I] = GasUtil.CALQIONREG(EN, NKSH, YKSH, XKSH)
            if EN > 2 * object.EION[8]:
                object.PEQION[8][I] = object.PEQEL[1][I - IOFFION[8]]

        # CORRECT IONISATION FOR SPLIT INTO K-SHELL
        QSUN = 0.0
        for i in range(9):
            QSUM += object.QION[i][I]
        if QSUM != 0:
            FAC = (QSUM - object.QION[8][I])/ QSUM
            for i in range(9):
                object.QION[i][I] = object.QION[i][I] * FAC

        #ATTACHMENT
        object.Q[3][I] = 0.0
        object.QATT[0][I] = 0.0
        if EN >= XATT[0]:
            object.Q[3][I] = GasUtil.CALQION(EN, NATT1, YATT, XATT)
            object.QATT[0][I] = object.Q[3][I]
        # COUNTING IONISATION
        object.Q[4][I] = 0.0

        object.Q[5][I] = 0.0

        #V4  SUPERELASTIC ISOTROPIC
        object.QIN[0][I] = 0.0
        object.PEQIN[0][I] = 0.5
        if EN > 0.0:
            object.QIN[0][I] = GasUtil.CALQINVISO(EN, NVIBV4, YVBV4, XVBV4, APOPV4, object.EIN[1], DEGV4, object.EIN[0],
                                                   <float> (0.076))
        #V4 ANISOTROPIC
        object.QIN[1][I] = 0.0
        object.PEQIN[1][I] = 0.5
        if EN > object.EIN[1]:
            object.QIN[1][I] = GasUtil.CALQINVANISO(EN, NVIBV4, YVBV4, XVBV4, object.EIN[1], APOPGS, RAT,
                                                    <float> (0.076))
            #RATIO OF MT TO TOTAL X-SECT FOR RESONANCE PART =RAT
            XMT = GasUtil.CALXMTVANISO(EN, NVIBV4, YVBV4, XVBV4, object.EIN[1], APOPGS, RAT, <float>(0.076))
            object.PEQIN[1][I] = 0.5 + (object.QIN[1][I] - XMT) / object.QIN[1][I]

        #V2  SUPERELASTIC ISOTROPIC
        object.QIN[2][I] = 0.0
        object.PEQIN[2][I] = 0.5
        if EN > 0.0:
            object.QIN[2][I] = GasUtil.CALQINVISO(EN, NVIBV2, YVBV2, XVBV2, APOPV2, object.EIN[3], DEGV2, object.EIN[0],
                                                  0.0)

        #V2  ISOTROPIC
        object.QIN[3][I] = 0.0
        object.PEQIN[3][I] = 0.5
        if EN > object.EIN[3]:
            object.QIN[3][I] = GasUtil.CALQINVISO(EN, NVIBV2, YVBV2, XVBV2, APOPGS, 0, 1, object.EIN[0], 0.0)

        #V1  ISOTROPIC
        object.QIN[4][I] = 0.0
        object.PEQIN[4][I] = 0.5
        if EN > object.EIN[4]:
            object.QIN[4][I] = GasUtil.CALQINVISO(EN, NVIBV1, YVBV1, XVBV1, 1, 0, 1, object.EIN[0], 0.0)

        #V3 ANISOTROPIC
        object.QIN[5][I] = 0.0
        object.PEQIN[5][I] = 0.5
        if EN > object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQINVANISO(EN, NVIBV3, YVBV3, XVBV3, object.EIN[5], 1, RAT, <float> (0.076))
            #RATIO OF MT TO TOTAL X-SECT FOR RESONANCE PART =RAT
            XMT = GasUtil.CALXMTVANISO(EN, NVIBV3, YVBV3, XVBV3, object.EIN[5], 1, RAT, <float> (0.076))
            object.PEQIN[5][I] = 0.5 + (object.QIN[5][I] - XMT) / object.QIN[5][I]

        #VIBRATION HARMONICS 1 ISOTROPIC
        object.QIN[6][I] = 0.0
        object.PEQIN[6][I] = 0.5
        if EN > object.EIN[6]:
            object.QIN[6][I] = GasUtil.CALQINVISO(EN, NVIBH1, YVBH1, XVBH1, 1, 0, 1, object.EIN[0], 0.0)

        #VIBRATION HARMONICS 2 ISOTROPIC
        object.QIN[7][I] = 0.0
        object.PEQIN[7][I] = 0.5
        if EN > object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQINVISO(EN, NVIBH2, YVBH2, XVBH2, 1, 0, 1, object.EIN[0], 0.0)

        #TRIPLET DISSOCIATION 7.5EV
        object.QIN[8][I] = 0.0
        object.PEQIN[8][I] = 0.0
        if EN > object.EIN[8]:
            object.QIN[8][I] = GasUtil.CALQINVISO(EN, NTRP1, YTR1, XTR1, 1, 0, 1, object.EIN[0], 0.0)
            if EN > 3 * object.EIN[8]:
                object.PEQIN[8][I] = object.PEQEL[1][I - IOFFN[8]]

        #ATTACHMENT - DEATTACHMENT RESONANCE VIA H- AT 9.8EV RESONANCE
        object.QIN[9][I] = 0.0
        object.PEQIN[9][I] = 0.0
        if EN > object.EIN[9]:
            object.QIN[9][I] = GasUtil.CALQION(EN, NDET, YDET, XDET)
            if EN > 3 * object.EIN[9] and object.QIN[9][I] != 0.0:
                object.PEQIN[9][I] = object.PEQEL[1][I - IOFFN[9]]

        #TRIPLET DISSOCIATION 8.5EV
        object.QIN[10][I] = 0.0
        object.PEQIN[10][I] = 0.0
        if EN > object.EIN[10]:
            object.QIN[10][I] = GasUtil.CALQINVISO(EN, NTRP2, YTR2, XTR2, 1, 0, 1, object.EIN[0], 0.0)
            if EN > 3 * object.EIN[10]:
                object.PEQIN[10][I] = object.PEQEL[1][I - IOFFN[10]]

        #SINGLET DISSOCIATION AT 8.75+FI*0.5 EV USE BEF SCALING WITH F[J]
        FI = 0
        CONI = 0
        for J in range(11, 14):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.0
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (<float>(4.0) * object.EIN[J])) - BETA2 - object.DEN[
                    I] / <float>(2.0)) * BBCONST * EN / (EN + object.EIN[J] + object.E[2]) * CON[CONI]
                if EN > 3 * object.EIN[J]:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        #TRIPLET DISSOCIATION 10.0EV
        object.QIN[14][I] = 0.0
        object.PEQIN[14][I] = 0.0
        if EN > object.EIN[14]:
            object.QIN[14][I] = GasUtil.CALQINVISO(EN, NTRP3, YTR3, XTR3, 1, 0, 1, object.EIN[0], 0.0)
            if EN > 3 * object.EIN[14]:
                object.PEQIN[14][I] = object.PEQEL[1][I - IOFFN[14]]

        #SINGLET DISSOCIATION AT 8.75+FI*0.5 EV USE BEF SCALING WITH F[J]
        for J in range(15, 22):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.0
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (<float>(4.0) * object.EIN[J])) - BETA2 - object.DEN[
                    I] / <float>(2.0)) * BBCONST * EN / (EN + object.EIN[J] + object.E[2]) * CON[CONI]
                if EN > 3 * object.EIN[J]:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        #DISSOCIATIVE EXC TO STATES DECAYING VIA CH(A2DELTA TO G.S.)
        object.QIN[22][I] = 0.0
        object.PEQIN[22][I] = 0.0
        if EN > object.EIN[22]:
            object.QIN[22][I] = GasUtil.CALQINP(EN, NCHD, YCHD, XCHD, 1) * 100
            if EN > 3 * object.EIN[22]:
                object.PEQIN[22][I] = object.PEQEL[1][I - IOFFN[22]]

        #DISSOCIATIVE EXC TO STATES DECAYING VIA CH(B2SIGMA- TO G.S.)
        object.QIN[23][I] = 0.0
        object.PEQIN[23][I] = 0.0
        if EN > object.EIN[23]:
            object.QIN[23][I] = GasUtil.CALQINP(EN, NCHB, YCHB, XCHB, 1) * 100
            if EN > 3 * object.EIN[23]:
                object.PEQIN[23][I] = object.PEQEL[1][I - IOFFN[23]]

        #SINGLET DISSOCIATION AT 8.75+FI*0.5 EV USE BEF SCALING WITH F[J]
        for J in range(24, 30):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.0
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (<float>(4.0) * object.EIN[J])) - BETA2 - object.DEN[
                    I] / <float>(2.0)) * BBCONST * EN / (EN + object.EIN[J] + object.E[2]) * CON[CONI]
                if EN > 3 * object.EIN[J]:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        #DISSOCIATIVE EXC TO STATES DECAYING VIA H(ALPHA)
        object.QIN[30][I] = 0.0
        object.PEQIN[30][I] = 0.0
        if EN > object.EIN[30]:
            object.QIN[30][I] = GasUtil.CALQINP(EN, NHAL, YHAL, XHAL, 1) * 100
            if EN > 3 * object.EIN[30]:
                object.PEQIN[30][I] = object.PEQEL[1][I - IOFFN[30]]

        #DISSOCIATIVE EXC TO STATES DECAYING VIA H(BETA)
        object.QIN[31][I] = 0.0
        object.PEQIN[31][I] = 0.0
        if EN > object.EIN[31]:
            object.QIN[31][I] = GasUtil.CALQINP(EN, NHBE, YHBE, XHBE, 1) * 100
            if EN > 3 * object.EIN[31]:
                object.PEQIN[31][I] = object.PEQEL[1][I - IOFFN[31]]

        #SINGLET DISSOCIATION AT 8.75+FI*0.5 EV USE BEF SCALING WITH F[J]
        for J in range(32, 34):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.0
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (<float>(4.0) * object.EIN[J])) - BETA2 - object.DEN[
                    I] / <float>(2.0)) * BBCONST * EN / (EN + object.EIN[J] + object.E[2]) * CON[CONI]
                if EN > 3 * object.EIN[J]:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        #LOAD BREMSSTRAHLUNG X-SECTIONS
        object.QIN[34][I] = 0.0
        object.QIN[35][I] = 0.0
        if EN > 1000:
            object.QIN[34][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z6T, EBRM) * 1e-8
            object.QIN[35][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z1T, EBRM) * 4e-8

        #skipped the QSUP,QVIB,QDATT,QSING,QTRIP,QEXC,QTTT,QWINT,QINEL,QIONS as they are not used later one.

    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
    return
