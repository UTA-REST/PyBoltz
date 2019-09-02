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
cdef void Gas21(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Hydrogen gas.
    """
    gd = np.load('gases.npy').item()

    cdef double XELM[172], YELM[172], YELT[172], YEPS[172], XROT0[53], YROT0[53], XROT1[43], YROT1[43], XROT2[28], YROT2[28],
    cdef double XROT3[28], YROT3[28], XVIB1[43], YVIB1[43], XVIB2[42], YVIB2[42], XVIB3[13], YVIB3[13], XVIB4[12], YVIB4[12],
    cdef double XB3S1[3], YB3S1[3], XB3S2[6], YB3S2[6], XB3S3[5], YB3S3[5], DISLY[37], DISWR[14], DISD1P[16], DISB1S[9]
    cdef double XB3S4[8], YB3S4[8], XC3PI[5], YC3PI[5], XA3SG[5], YA3SG[5], XE3SG[5], YE3SG[5], XEFSG[34], YEFSG[34],
    cdef double XATT[18], YATT[18], XION[92], YION[92], XIOND[61], YIOND[61], PJ[7], ERLVL[7], BEF[10], Z1T[25], EBRM[25]
    cdef int IOFFN[107], IOFFION[2]

    XELM = gd['gas21/XELM']
    YELM = gd['gas21/YELM']
    YELT = gd['gas21/YELT']
    YEPS = gd['gas21/YEPS']
    XROT0 = gd['gas21/XROT0']
    YROT0 = gd['gas21/YROT0']
    XROT1 = gd['gas21/XROT1']
    YROT1 = gd['gas21/YROT1']
    XROT2 = gd['gas21/XROT2']
    YROT2 = gd['gas21/YROT2']
    XROT3 = gd['gas21/XROT3']
    YROT3 = gd['gas21/YROT3']
    XVIB1 = gd['gas21/XVIB1']
    YVIB1 = gd['gas21/YVIB1']
    XVIB2 = gd['gas21/XVIB2']
    YVIB2 = gd['gas21/YVIB2']
    XVIB3 = gd['gas21/XVIB3']
    YVIB3 = gd['gas21/YVIB3']
    XVIB4 = gd['gas21/XVIB4']
    YVIB4 = gd['gas21/YVIB4']
    XB3S1 = gd['gas21/XB3S1']
    YB3S1 = gd['gas21/YB3S1']
    XB3S2 = gd['gas21/XB3S2']
    YB3S2 = gd['gas21/YB3S2']
    XB3S3 = gd['gas21/XB3S3']
    YB3S3 = gd['gas21/YB3S3']
    XB3S4 = gd['gas21/XB3S4']
    YB3S4 = gd['gas21/YB3S4']
    XC3PI = gd['gas21/XC3PI']
    YC3PI = gd['gas21/YC3PI']
    XA3SG = gd['gas21/XA3SG']
    YA3SG = gd['gas21/YA3SG']
    XE3SG = gd['gas21/XE3SG']
    YE3SG = gd['gas21/YE3SG']
    XEFSG = gd['gas21/XEFSG']
    YEFSG = gd['gas21/YEFSG']
    XATT = gd['gas21/XATT']
    YATT = gd['gas21/YATT']
    XION = gd['gas21/XION']
    YION = gd['gas21/YION']
    XIOND = gd['gas21/XIOND']
    YIOND = gd['gas21/YIOND']
    DISLY = gd['gas21/DISLY']
    DISWR = gd['gas21/DISWR']
    DISD1P = gd['gas21/DISD1P']
    DISB1S = gd['gas21/DISB1S']
    Z1T = gd['gas21/Z1T']
    EBRM = gd['gas21/EBRM']

    cdef double A0, RY, CONST, EMASS2, API, BBCONST, AM2, C,
    cdef int NBREM, i, j, I, J
    A0 = 0.52917720859e-08
    RY = 13.60569193
    CONST = 1.873884e-20
    EMASS2 = 1021997.804
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / EMASS2
    #BORN BETHE VALUES FOR IONISATION
    AM2 = 0.642
    C = 8.30

    object.NION = 2
    object.NATT = 1
    object.NIN = 107
    object.NNULL = 0

    for i in range(6):
        object.KEL[i] = object.WhichAngularModel
    for i in range(4, object.NIN):
        object.KIN[i] = object.WhichAngularModel

    for i in range(4):
        object.KIN[i] = 0

    cdef int NELM, NROT0, NROT1, NROT2, NROT3, NVIB1, NVIB2, NVIB3, NVIB4, NB3S1, NB3S2, NB3S3, NB3S4, NC3PI, NA3SG, NE3SG, NEFSG, NIONG
    cdef int NIOND, NATT1
    NELM = 172
    NROT0 = 53
    NROT1 = 43
    NROT2 = 28
    NROT3 = 28
    NVIB1 = 43
    NVIB2 = 42
    NVIB3 = 13
    NVIB4 = 12
    NB3S1 = 3
    NB3S2 = 6
    NB3S3 = 5
    NB3S4 = 8
    NC3PI = 5
    NA3SG = 5
    NE3SG = 5
    NEFSG = 34
    NIONG = 92
    NIOND = 61
    NATT1 = 18
    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, ESCOBY, EOBY[2], EATTTH, EATTWD, AMPATT, EATTTH1, EATTWD1, AMPATT1

    object.E = [0.0, 1.0, 15.418, 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * EMASS / (2.01565 * AMU)
    # IONISATION ENERGY FOR PARA =15.42580155 EV
    # IONISATION ENERGY FOR ORTHO=15.41833111 EV
    # USE ORTHO ENERGY FOR ROOM TEMPERATURE GAS
    object.EION[0:2] = [object.E[2], 18.076]

    # OPAL BEATY FOR LOW ENERGY
    ESCOBY = 0.5
    EOBY[0] = object.EION[0] * ESCOBY
    EOBY[1] = object.EION[1] * ESCOBY

    #FLUORESENCE DATA
    for J in range(2):
        object.NC0[J] = 0
        object.EC0[J] = 0.0
        object.WK[J] = 0.0
        object.EFL[J] = 0.0
        object.NG1[J] = 0
        object.EG1[J] = 0.0
        object.NG2[J] = 0
        object.EG2[J] = 0.0

    for j in range(0, object.NION):
        for i in range(0, 4000):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i - 1
                break
    #TODO: add EIN comments

    object.EIN = gd['gas21/EIN']

    for J in range(5):
        BEF[J] = object.E[2]

    # ATTACHMENT THRESHOLD EV FOR 2 SIGMAu
    EATTTH = 3.723
    # ATTACHMENT WIDTH FOR 2 SIGMAu
    EATTWD = 0.45
    # ATTACHMENT AMPLITUDE FOR 2 SIGMAu
    AMPATT = 3.0e-21
    # ATTACHMENT THRESHOLD EV FOR 2 SIGMAg
    EATTTH1 = 13.922
    # ATTACHMENT WIDTH FOR 2 SIGMAg
    EATTWD1 = 0.95
    # ATTACHMENT AMPLITUDE FOR 2 SIGMAg
    AMPATT1 = 3.0e-20

    # ROTATIONAL ENERGY LEVELS: ERLVL(N)
    #  PARA - ORTHO ENERGY DIFFERENCE ( J=0 - J=1 ROT LEVEL) = 0.01469049 EV
    #  REF :ASTROPHYS J.  282(1984)L85
    ERLVL[0] = 0.01469049
    ERLVL[1] = object.EIN[4]
    ERLVL[2] = 0.01469049 + object.EIN[5]
    ERLVL[3] = object.EIN[4] + object.EIN[6]
    ERLVL[4] = 0.01469049 + object.EIN[5] + object.EIN[7]
    ERLVL[5] = object.EIN[4] + object.EIN[6] + 0.15381
    ERLVL[6] = 0.01469049 + object.EIN[5] + object.EIN[7] + 0.1794

    for J in range(object.NIN):
        object.PENFRA[0][J] = 0.0
        object.PENFRA[1][J] = 1.0
        object.PENFRA[2][J] = 1.0

    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION
    cdef int NL = 0, FI
    for NL in range(object.NIN):
        for i in range(4000):
            if object.EG[i] > abs(object.EIN[NL]):
                IOFFN[NL] = i - 1
                break

    cdef double SUM, FROT[8], GAMMA1, GAMMA2, BETA, BETA2, QMOM, QELA, PQ[3], EN, F[84]

    F = [.0016884, .005782, .011536, .017531, .022477, .025688, .027021, .026731, .025233, .022980, .020362, .017653,
         .015054, .012678, .010567, .008746, .007201, .005909, .004838, .003956, .003233, .002644, .002165, .001775,
         .001457, .001199, .0009882, .0008153, .0006738, .0005561, .0004573, .0003731, .0002992, .0002309, .0001627,
         8.652e-5, 2.256e-5, .0476000, .0728400, .0698200, .0547200, .0387400, .0259800, .0170000, .0109900, .0070980,
         .0045920, .0029760, .0019090, .0011710, .0005590, .003970, .008150, .009980, .009520, .007550, .004230,
         .000460, .000450, .000300, .007750, .013100, .013670, .011560, .008730, .006190, .004280, .002920, .001960,
         .001330, .000910, .000630, .000430, .000290, .000200, .000120, .02230, .01450, .01450, .01010, .00500, .02680,
         .01700, .00927]

    SUM = 1.0
    #ROTATIONAL POPULATIONS
    for I in range(1, 8, 2):
        PJ[I - 1] = 3 * (2 * I + 1) * exp(-1 * ERLVL[I - 1] / object.AKT)

    for I in range(2, 7, 2):
        PJ[I - 1] = (2 * I + 1) * exp(-1 * ERLVL[I - 1] / object.AKT)

    for I in range(7):
        SUM+=PJ[I]
    FROT[0] = 1.0 / SUM

    for I in range(1, 8):
        FROT[I] = PJ[I - 1] / SUM

    for I in range(4000):
        EN = object.EG[I]
        if EN > object.EIN[0]:
            GAMMA1 = (EMASS2 + 2 * EN) / EMASS2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = sqrt(1.0 - 1.0 / GAMMA2)
            BETA2 = BETA * BETA

        QMOM = GasUtil.CALQIONREG(EN, NELM, YELM, XELM)
        QELA = GasUtil.CALQIONREG(EN, NELM, YELT, XELM)
        PQ[2] = GasUtil.CALPQ3(EN, NELM, YEPS, XELM)

        PQ = [0.5, 0.5 + (QELA - QMOM) / QELA, 1 - PQ[2]]

        object.PEQEL[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = QELA
        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM

        # GROSS IONISATION
        object.QION[0][I] = 0.0
        object.PEQION[0][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[0][I] = 0
        if EN >= object.EION[0]:
            object.QION[0][I] = GasUtil.CALQIONX(EN, NIONG, YION, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[0]:
            object.PEQION[0][I] = object.PEQEL[1][I - IOFFION[0]]

        # DISSOCIATIVE IONISATION
        object.QION[0][I] = 0.0
        object.PEQION[0][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[0][I] = 0
        if EN >= object.EION[1]:
            object.QION[1][I] = GasUtil.CALQIONX(EN, NIOND, YIOND, XIOND, BETA2, 0.05481, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[1]:
            object.PEQION[1][I] = object.PEQEL[1][I - IOFFION[1]]

        # CALCULATE NON_DISSOCIATIVE IONISATION
        if object.QION[0][I] != 0.0:
            object.QION[0][I] -= object.QION[1][I]

        #ATTACHMENT
        object.Q[3][I] = 0.0
        object.QATT[0][I] = 0.0
        object.PEQEL[3][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQEL[3][I] = 0.0

        #ROTATIONAL DEPENDANCE OF ATTACHMENT TO 2 SIGMAu

        if EN >= (EATTTH - ERLVL[6]):
            object.Q[3][I] = AMPATT * 5.0 * FROT[7] * exp(-1 * (EN - EATTTH + ERLVL[6]) / EATTWD)
        if EN >= (EATTTH - ERLVL[5]):
            object.Q[3][I] += AMPATT * 3.96 * FROT[6] * exp(-1 * (EN - EATTTH + ERLVL[5]) / EATTWD)
        if EN >= (EATTTH - ERLVL[4]):
            object.Q[3][I] += AMPATT * 3.15 * FROT[5] * exp(-1 * (EN - EATTTH + ERLVL[4]) / EATTWD)
        if EN >= (EATTTH - ERLVL[3]):
            object.Q[3][I] += AMPATT * 2.50 * FROT[4] * exp(-1 * (EN - EATTTH + ERLVL[3]) / EATTWD)
        if EN >= (EATTTH - ERLVL[2]):
            object.Q[3][I] += AMPATT * 1.99 * FROT[3] * exp(-1 * (EN - EATTTH + ERLVL[2]) / EATTWD)
        if EN >= (EATTTH - ERLVL[1]):
            object.Q[3][I] += AMPATT * 1.58 * FROT[2] * exp(-1 * (EN - EATTTH + ERLVL[1]) / EATTWD)
        if EN >= (EATTTH - ERLVL[0]):
            object.Q[3][I] += AMPATT * 1.26 * FROT[1] * exp(-1 * (EN - EATTTH + ERLVL[0]) / EATTWD)
        if EN >= EATTTH:
            object.Q[3][I] += AMPATT * FROT[0] * exp(-1 * (EN - EATTTH) / EATTWD)

        if EN > XATT[0]:
            # ATTACHMENT TO 2 SIGMAg
            object.Q[3][I] += GasUtil.CALQIONREG(EN, NATT1, YATT, XATT)
        if EN > EATTTH1:
            object.Q[3][I] += AMPATT1 * exp(-1 * (EN - EATTTH1) / EATTWD1)
        object.QATT[0][I] = object.Q[3][I]

        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        for J in range(object.NIN + 1):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEQIN[J][I] = 0.0

        # SUPERELASTIC 2-0
        if EN > 0.0:
            object.QIN[0][I] = GasUtil.CALQIONREG(EN + object.EIN[4], NROT0, YROT0, XROT0)
            object.QIN[0][I] *= ((object.EIN[4] + EN) / EN) * FROT[2] * 0.2

        # SUPERELASTIC 3-1
        if EN > 0.0:
            object.QIN[1][I] = GasUtil.CALQIONREG(EN + object.EIN[5], NROT1, YROT1, XROT1)
            object.QIN[1][I] *= ((object.EIN[5] + EN) / EN) * FROT[3] * (3.0 / 7.0)

        # SUPERELASTIC 4-2
        if EN > 0.0:
            object.QIN[2][I] = GasUtil.CALQIONREG(EN + object.EIN[6], NROT2, YROT2, XROT2)
            object.QIN[2][I] *= ((object.EIN[6] + EN) / EN) * FROT[4] * (5.0 / 9.0)

        # SUPERELASTIC 5-3
        if EN > 0.0:
            object.QIN[3][I] = GasUtil.CALQIONREG(EN + object.EIN[7], NROT3, YROT3, XROT3)
            object.QIN[3][I] *= ((object.EIN[7] + EN) / EN) * FROT[5] * (7.0 / 11.0)

        # ROTATION 0-2
        if EN > object.EIN[4]:
            object.QIN[4][I] = GasUtil.CALQINP(EN, NROT0, YROT0, XROT0, 1) * 100 * FROT[0]
        if EN > 2 * object.EIN[4]:
            object.PEQIN[4][I] = object.PEQEL[1][I - IOFFN[4]]

        # ROTATION 1-3
        if EN > object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQINP(EN, NROT1, YROT1, XROT1, 1) * 100 * FROT[1]
        if EN > 2 * object.EIN[5]:
            object.PEQIN[5][I] = object.PEQEL[1][I - IOFFN[5]]

        #                      ROTATION 2-4 + 4-6 + 6-8
        # USED SCALED 2-4 XSECTION FOR 4-6 AND 6-8
        # ALSO SCALED FOR ENERGY LOSS BY 1.5 FOR 4-6 AND BY 2.0 FOR 6-8
        if EN > object.EIN[6]:
            object.QIN[6][I] = GasUtil.CALQINP(EN, NROT2, YROT2, XROT2, 1) * 100 * (
                    FROT[2] + FROT[4] * 0.8 * 1.5 + FROT[6] * 0.5 * 2.0)
        if EN > 2 * object.EIN[6]:
            object.PEQIN[6][I] = object.PEQEL[1][I - IOFFN[6]]

        #                        ROTATION 3-5 + 5-7 + 7-9
        # USED SCALED 3-5 XSECTION FOR 5-7 AND 7-9
        # ALSO SCALED FOR ENERGY LOSS BY 1.4 FOR 5-7 AND 1.8 FOR 7-9
        if EN > object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQINP(EN, NROT3, YROT3, XROT3, 1) * 100 * (
                    FROT[3] + FROT[5] * 0.8 * 1.5 + FROT[7] * 0.5 * 2.0)
        if EN > 2 * object.EIN[7]:
            object.PEQIN[7][I] = object.PEQEL[1][I - IOFFN[7]]

        #VIBRATION V1 with DJ = 0
        if EN > object.EIN[8]:
            object.QIN[8][I] = GasUtil.CALQINP(EN, NVIB1, YVIB1, XVIB1, 1) * 100
        if EN > 2 * object.EIN[8]:
            object.PEQIN[8][I] = object.PEQEL[1][I - IOFFN[8]]

        #VIBRATION V1 with DJ = 2
        if EN > object.EIN[9]:
            object.QIN[9][I] = GasUtil.CALQINP(EN, NVIB2, YVIB2, XVIB2, 1) * 100
        if EN > 2 * object.EIN[9]:
            object.PEQIN[9][I] = object.PEQEL[1][I - IOFFN[9]]

        #VIBRATION V2
        if EN > object.EIN[10]:
            object.QIN[10][I] = GasUtil.CALQINP(EN, NVIB3, YVIB3, XVIB3, 1) * 100
        if EN > 2 * object.EIN[10]:
            object.PEQIN[10][I] = object.PEQEL[1][I - IOFFN[10]]

        #VIBRATION V3
        if EN > object.EIN[11]:
            object.QIN[11][I] = GasUtil.CALQINP(EN, NVIB4, YVIB4, XVIB4, 1) * 100
        if EN > 2 * object.EIN[11]:
            object.PEQIN[11][I] = object.PEQEL[1][I - IOFFN[11]]

        # B3 SIGMA DISSOCIATION ELOSS=8.0EV
        if EN > object.EIN[12]:
            object.QIN[12][I] = GasUtil.CALQION(EN, NB3S1, YB3S1, XB3S1)
        if EN > 2 * object.EIN[12]:
            object.PEQIN[12][I] = object.PEQEL[1][I - IOFFN[12]]

        # B3 SIGMA DISSOCIATION ELOSS=9.0EV
        if EN > object.EIN[13]:
            object.QIN[13][I] = GasUtil.CALQION(EN, NB3S2, YB3S2, XB3S2)
        if EN > 2 * object.EIN[13]:
            object.PEQIN[13][I] = object.PEQEL[1][I - IOFFN[13]]

        # B3 SIGMA DISSOCIATION ELOSS=9.5EV
        if EN > object.EIN[14]:
            object.QIN[14][I] = GasUtil.CALQION(EN, NB3S3, YB3S3, XB3S3)
        if EN > 2 * object.EIN[14]:
            object.PEQIN[14][I] = object.PEQEL[1][I - IOFFN[14]]

        # B3 SIGMA DISSOCIATION ELOSS=10.0EV
        # SCALED BY 1/E**3 ABOVE XB3S4(NB3S4) EV
        if EN > object.EIN[15]:
            object.QIN[15][I] = GasUtil.CALQINP(EN, NB3S4, YB3S4, XB3S4, 3) * 100
        if EN > 2 * object.EIN[15]:
            object.PEQIN[15][I] = object.PEQEL[1][I - IOFFN[15]]

        #LYMAN BANDS FOR VIB=0 TO 36    B1 SIGMA--- GROUND STATE
        #   DIPOLE ALLOWED
        # V=FI B1 SIGMA
        FI = 0
        for J in range(16, 53):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + BEF[0])
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1

        #V =FI C1 PI
        for J in range(53, 67):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + BEF[1])
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1

        # C3PI V=0-4  METASTABLE LEVEL     FRANCK-CONDON FAC=0.6967
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EIN[67]:
            object.QIN[67][I] = GasUtil.CALQINP(EN, NC3PI, YC3PI, XC3PI, 3) * 100 * 0.6967
        if EN > 2 * object.EIN[67]:
            object.PEQIN[67][I] = object.PEQEL[1][I - IOFFN[67]]

        # C3PI V=5-18  METASTABLE LEVEL     FRANCK-CONDON FAC=0.3033
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EIN[68]:
            object.QIN[68][I] = GasUtil.CALQINP(EN, NC3PI, YC3PI, XC3PI, 3) * 100 * 0.3033
        if EN > 2 * object.EIN[68]:
            object.PEQIN[68][I] = object.PEQEL[1][I - IOFFN[68]]

        # A3SG V=0-2  METASTABLE LEVEL     FRANCK-CONDON FAC=0.6668
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EIN[69]:
            object.QIN[69][I] = GasUtil.CALQINP(EN, NA3SG, YA3SG, XA3SG, 3) * 100 * 0.6668
        if EN > 2 * object.EIN[69]:
            object.PEQIN[69][I] = object.PEQEL[1][I - IOFFN[69]]

        # A3SG V=3-17  METASTABLE LEVEL     FRANCK-CONDON FAC=0.3332
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EIN[70]:
            object.QIN[70][I] = GasUtil.CALQINP(EN, NA3SG, YA3SG, XA3SG, 3) * 100 * 0.3332
        if EN > 2 * object.EIN[70]:
            object.PEQIN[70][I] = object.PEQEL[1][I - IOFFN[70]]

        # E3SG V=0-9
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EIN[71]:
            object.QIN[71][I] = GasUtil.CALQINP(EN, NE3SG, YE3SG, XE3SG, 3) * 100
        if EN > 2 * object.EIN[71]:
            object.PEQIN[71][I] = object.PEQEL[1][I - IOFFN[71]]

        # EF1 SIGMA V=0-5           FRANCK-CONDON FACTOR=0.4
        # USE BORN SCALING ABOVE XEFSG(NEFSG)  EV
        if EN > object.EIN[72]:
            object.QIN[72][I] = GasUtil.CALQINBEF(EN,EN, NEFSG, YEFSG, XEFSG, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                  BBCONST, object.EIN[72], BEF[2], 0.0089000)
            if EN <= XEFSG[NEFSG - 1]:
                object.QIN[72][I] * 100 * 0.4
        if EN > 2 * object.EIN[72]:
            object.PEQIN[72][I] = object.PEQEL[1][I - IOFFN[72]]

        # EF1 SIGMA V=0-5           FRANCK-CONDON FACTOR=0.6
        # USE BORN SCALING ABOVE XEFSG(NEFSG)  EV
        if EN > object.EIN[73]:
            object.QIN[73][I] = GasUtil.CALQINBEF(EN, EN,NEFSG, YEFSG, XEFSG, BETA2, GAMMA2, EMASS2, object.DEN[I],
                                                  BBCONST, object.EIN[73], BEF[2], 0.0133000)
            if EN <= XEFSG[NEFSG - 1]:
                object.QIN[73][I] * 100 * 0.6
        if EN > 2 * object.EIN[73]:
            object.PEQIN[73][I] = object.PEQEL[1][I - IOFFN[73]]

        #B!1 SIGMA V=FI
        for J in range(74, 83):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + BEF[3])
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1

        #D1 PI  V=FI
        for J in range(83, 99):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + BEF[4])
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1

        #TODO: add comments
        for J in range(99, 104):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + BEF[4])*1.08
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1

        for J in range(104, 106):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + BEF[4])*1.20
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1
        J=106
        if EN > object.EIN[J]:
            object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + BEF[4])
            if object.QIN[J][I] < 0.0:
                object.QIN[J][I] = 0.0
        if EN > 2 * object.EIN[J]:
            object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
        FI += 1

        object.Q[0][I] = 0.0
        for J in range(107):
            object.Q[0][I]+=object.QIN[J][I]

        object.Q[0][I]+=object.Q[1][I] +object.Q[3][I]+object.QION[0][I]+object.QION[1][I]

    object.NIN =12
    if object.EFINAL>8.0 and object.EFINAL<=10.0:
        object.NIN = 16
    if object.EFINAL>10.0:
        object.NIN = 107
    return
