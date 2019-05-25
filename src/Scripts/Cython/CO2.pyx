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
cdef void Gas12(Gas*object):
    gd = np.load('gases.npy').item()
    cdef double XEN[158], YMOM[158], YEL[158], YVBMOM[158], YVBEL[158], YEPS[158], XION1[63], YION1[63], XION2[66], YION2[66], XION3[66],
    cdef double YION3[66], XION4[41], YION4[41], XION5[41], YION5[41], XION6[40], YION6[40], XION7[37], YION7[37], XION8[30], YION8[30], XION9[27],
    cdef double YION9[27], XATT[68], YATT[68], XV2[17], YV2[17], X2V2[19], Y2V2[19], XV1[26], YV1[26], X3V2[11], Y3V2[11], XV3[11]
    cdef double YV3[11], XVPD3[14], YVPD3[14], XV130[12], YV130[12], XVPD4[14], YVPD4[14], XVPD5[11], YVPD5[11], XVPD6[11], YVPD6[11]
    cdef double XVPD7[11], YVPD7[11], XVPD8[11], YVPD8[11], XVPD9[11], YVPD9[11], XVPDH[9], YVPDH[9], XTRP1[11], YTRP1[11], XTRP2[11], YTRP2[11],
    cdef double XKSHC[83], YKSHC[83], XKSHO[81], YKSHO[81], Z6T[25], Z8T[25], EBRM[25]
    cdef int IOFFN[144], IOFFION[11], i, j, I, J, NL

    XEN = gd['gas12/XEN']
    YMOM = gd['gas12/YMOM']
    YEL = gd['gas12/YEL']
    YVBMOM = gd['gas12/YVBMOM']
    YVBEL = gd['gas12/YVBEL']
    YEPS = gd['gas12/YEPS']
    XION1 = gd['gas12/XION1']
    YION1 = gd['gas12/YION1']
    XION2 = gd['gas12/XION2']
    YION2 = gd['gas12/YION2']
    XION3 = gd['gas12/XION3']
    YION3 = gd['gas12/YION3']
    XION4 = gd['gas12/XION4']
    YION4 = gd['gas12/YION4']
    XION5 = gd['gas12/XION5']
    YION5 = gd['gas12/YION5']
    XION6 = gd['gas12/XION6']
    YION6 = gd['gas12/YION6']
    XION7 = gd['gas12/XION7']
    YION7 = gd['gas12/YION7']
    XION8 = gd['gas12/XION8']
    YION8 = gd['gas12/YION8']
    XION9 = gd['gas12/XION9']
    YION9 = gd['gas12/YION9']
    XATT = gd['gas12/XATT']
    YATT = gd['gas12/YATT']
    XV2 = gd['gas12/XV2']
    YV2 = gd['gas12/YV2']
    X2V2 = gd['gas12/X2V2']
    Y2V2 = gd['gas12/Y2V2']
    XV1 = gd['gas12/XV1']
    YV1 = gd['gas12/YV1']
    X3V2 = gd['gas12/X3V2']
    Y3V2 = gd['gas12/Y3V2']
    XV3 = gd['gas12/XV3']
    YV3 = gd['gas12/YV3']
    XVPD3 = gd['gas12/XVPD3']
    YVPD3 = gd['gas12/YVPD3']
    XV130 = gd['gas12/XV130']
    YV130 = gd['gas12/YV130']
    XVPD4 = gd['gas12/XVPD4']
    YVPD4 = gd['gas12/YVPD4']
    XVPD5 = gd['gas12/XVPD5']
    YVPD5 = gd['gas12/YVPD5']
    XVPD6 = gd['gas12/XVPD6']
    YVPD6 = gd['gas12/YVPD6']
    XVPD7 = gd['gas12/XVPD7']
    YVPD7 = gd['gas12/YVPD7']
    XVPD8 = gd['gas12/XVPD8']
    YVPD8 = gd['gas12/YVPD8']
    XVPD9 = gd['gas12/XVPD9']
    YVPD9 = gd['gas12/YVPD9']
    XVPDH = gd['gas12/XVPDH']
    YVPDH = gd['gas12/YVPDH']
    XTRP1 = gd['gas12/XTRP1']
    YTRP1 = gd['gas12/YTRP1']
    XTRP2 = gd['gas12/XTRP2']
    YTRP2 = gd['gas12/YTRP2']
    XKSHC = gd['gas12/XKSHC']
    YKSHC = gd['gas12/YKSHC']
    XKSHO = gd['gas12/XKSHO']
    YKSHO = gd['gas12/YKSHO']
    Z6T = gd['gas12/Z6T']
    Z8T = gd['gas12/Z8T']
    EBRM = gd['gas12/EBRM']

    #---------------------------------------------------------------------
    # 2018 UPDATE :  SCALED V(001) X-SECTION BY 0.975
    # ---------------------------------------------------------------------
    # 2015: UPGRADE INCLUDES :
    #      1) OSCILLATOR STRENGTH FROM ANALYSIS OF DATA FROM
    #         BRION GROUP AND SHAW ET AL . OSCILLATOR SUM S(0)=21.9856
    #         S(-1)i=5.372
    #      2) USED STRAUB DATA FOR DISSOCIATIVE IONISATION ABOVE 30EV
    #         AND   RAP AND ENGLADER-GOLDEN  AT LOW ENERGY
    #      3) IONISATION-EXCITATION FROM ITIKAWA REVIEW
    #
    # ANGULAR DISTRIBUTION ONLY ALLOWED FOR ELASTIC , IONISATION AND
    # EXCITATION ABOVE 10EV.
    #
    # ---------------------------------------------------------------------

    cdef double A0, RY, CONST, EMASS2, API, BBCONST, AM2, C, EOBFAC, AUGKC, AUGK0
    cdef int NBREM

    # BORN-BETHE CONSTANTS
    A0 = 0.52917720859e-08
    RY = 13.60569193
    CONST = 1.873884e-20
    EMASS2 = 1021997.804
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / EMASS2
    # BORN BETHE VALUES FOR IONISATION
    AM2 = 5.38
    EOBFAC = 0.56
    C = 57.0

    # AVERAGE SUGER EMISSIONS FROM EACH SHELL
    AUGKC = 2.0
    AUGK0 = 1.99

    object.NION = 11
    object.NATT = 1
    object.NIN = 144
    object.NNULL = 0

    NBREM = 25

    for J in range(6):
        object.KEL[J] = object.NANISO
    for J in range(object.NIN):
        object.KIN[J] = object.NANISO

    cdef int NEL, NV2, N2V2, NV1, N3V2, NV3, NPD3, NV130, NPD4, NPD5, NPD6, NPD7, NPD8, NPD9, NPDH, NATT1, NTRP1, NTRP2, NION1, NION2, NION3
    cdef int NION4, NION5, NION6, NION7, NION8, NION9, NKSHC, NKSHO

    NEL = 158
    NV2 = 17
    N2V2 = 19
    NV1 = 26
    N3V2 = 11
    NV3 = 11
    NPD3 = 14
    NV130 = 12
    NPD4 = 14
    NPD5 = 11
    NPD6 = 11
    NPD7 = 11
    NPD8 = 11
    NPD9 = 11
    NPDH = 9
    NATT1 = 68
    NTRP1 = 11
    NTRP2 = 11
    NION1 = 63
    NION2 = 66
    NION3 = 66
    NION4 = 41
    NION5 = 41
    NION6 = 40
    NION7 = 37
    NION8 = 30
    NION9 = 27
    NKSHC = 83
    NKSHO = 81
    cdef double EMASS = 9.10938291e-31, PENSUM
    cdef double AMU = 1.660538921e-27, EOBY[11]

    object.E = [0.0, 1.0, 13.776, 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * EMASS / (44.0095 * AMU)

    object.EION[0:11] = [13.776, 17.314, 18.077, 19.07, 19.47, 27.82, 37.4, 72.0, 74.0, 285.0, 532.0]

    # OPAL AND BEATY
    for J in range(11):
        EOBY[J] = 13.8

    for J in range(9):
        object.NC0[J] = 0
        object.EC0[J] = 0.0
        object.WK[J] = 0.0
        object.EFL[J] = 0.0
        object.NG1[J] = 0
        object.EG1[J] = 0.0
        object.NG2[J] = 0
        object.EG2[J] = 0.0
    #DOUBLE CHARGE ++ ION STATES (EXTRA ELECTRON)
    object.NC0[6] = 1
    object.EC0[6] = 1.0
    object.NC0[7] = 1
    object.EC0[7] = 1.0
    object.NC0[8] = 1
    object.EC0[8] = 1.0

    # FLUORESCENCE DATA K SHELLS

    object.NC0[9] = 2
    object.EC0[9] = 253.0
    object.WK[9] = 0.0026
    object.EFL[9] = 273.0
    object.NG1[9] = 1
    object.EG1[9] = 253.0
    object.NG2[9] = 2
    object.EG2[9] = 5.0
    object.NC0[10] = 3
    object.EC0[10] = 485.0
    object.WK[10] = 0.0069
    object.EFL[10] = 518.0
    object.NG1[10] = 1
    object.EG1[10] = 480
    object.NG2[10] = 2
    object.EG2[10] = 5.0

    cdef int L
    for j in range(0, object.NION):
        for i in range(0, 4000):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i - 1
                break
    cdef double AMPV2, AMPV3, B0, QBQA, QBK, PJ[220], SUM = 0.0
    # DIPOLE TRANSITION STRENGTH FOR VIBRATIONS V010 AND V001
    AMPV2 = 0.1703
    AMPV3 = 0.3824
    #-----------------------------------------------------------------------
    #  B0 IS ROTATIONAL CONSTANT
    #  QBQA IS QUADRUPOLE MOMENT
    B0 = 4.838e-5
    A0 = 0.5291772083e-8
    QBQA = 3.24
    QBK = 1.67552 * (QBQA * A0) ** 2

    #CALC FRACTIONAL POPULATION DENSITY OF ROTATIONAL STATES
    PJ[0] = 1.0
    for J in range(2, 32):
        i = (2 * J) - 2
        PJ[J - 1] = (2 * i + 1) * exp(-1 * i * (i + 1) * B0 / object.AKT)
    for J in range(31):
        SUM += PJ[J]
    for J in range(31):
        PJ[J] = PJ[J] / SUM
    object.EIN = gd['gas12/EIN']

    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION
    for NL in range(object.NIN):
        for i in range(4000):
            if object.EG[i] > object.EIN[NL]:
                IOFFN[NL] = i - 1
                break

    # ENTER PENNING TRANSFER FRACTION FOR EACH LEVEL
    # FIRST 81 LEVELS UNLIKELY TO HAVE ENOUGH ENERGY
    for I in range(3):
        for J in range(81):
            object.PENFRA[I][J] = 0.0

    # PENNING TRANSFER FRACTION FOR LEVELS
    for J in range(81, object.NIN):
        object.PENFRA[0][J] = 0.0
        #PENNING TRANSFER DISTANCE IN MICRONS
        object.PENFRA[1][J] = 1.0
        #PENNING TRANSFER TIME IN PICOSECONDS
        object.PENFRA[2][J] = 1.0

    for J in range(NBREM):
        EBRM[J] = exp(EBRM[J])

    cdef double DEGV1, DEGV2, DEGV3, DEG2V2, DEG3V2, APOPV2, APOP2V2, APOPV1, APOP3V2, APOPV3, APOPGS, APBEND, AEXT20, AGST20
    DEGV1 = 1.0
    DEGV2 = 2.0
    DEGV3 = 1.0
    DEG2V2 = 3.0
    # 3V2 === SUM (3V2 + V12) =   4+2
    DEG3V2 = 6.0
    #----------------------------------------------------
    # CALC POPULATION OF VIBRATIONAL STATES
    SUM = 0.0
    APOPV2 = DEGV2 * exp(object.EIN[60] / object.AKT)
    APOP2V2 = DEG2V2 * exp(object.EIN[62] / object.AKT)
    APOPV1 = DEGV1 * exp(object.EIN[64] / object.AKT)
    APOP3V2 = DEG3V2 * exp(object.EIN[66] / object.AKT)
    APOPV3 = DEGV3 * exp(object.EIN[68] / object.AKT)
    SUM = 1.0 + APOPV2 + APOP2V2 + APOPV1 + APOP3V2 + APOPV3
    APOPGS = 1.0 / SUM
    APOPV2 = APOPV2 / SUM
    APOP2V2 = APOP2V2 / SUM
    APOPV1 = APOPV1 / SUM
    APOP3V2 = APOP3V2 / SUM
    APOPV3 = APOPV3 / SUM
    APBEND = APOPV2 + APOP2V2 + APOP3V2

    # RENORMALISE VIBRATIONAL GROUND STATE POPULATION IN ORDER TO ACCOUNT
    # FOR EXCITATION FROM VIBRATIONALLY EXCITED STATES
    APOPGS = 1.0
    # BEND MODE AND EFFECTIVE GROUND STATE POPULATION AT 293.15 KELVIN
    AEXT20 = 7.51373753e-2
    AGST20 = 1.0 - AEXT20
    cdef double EN, GAMMA1, GAMMA2, BETA, BETA2, QMT, QEL, PQ[3], X1, X2, QBB = 0.0, QMOM, QELA, QBMOM, QBELA, F[62], CONS[62]
    cdef double SUMR, SUMV, SUME, SUMTRP, SUMEXC, SUMION
    F = [0.0000698, 0.0000630, 0.0000758, 0.0001638, 0.0003356, 0.0007378, 0.001145, 0.001409, 0.001481, 0.000859,
         0.001687, 0.002115, 0.001920, 0.001180, 0.000683, 0.000456, 0.004361, 0.1718, 0.06242, 0.01852, 0.01125,
         0.01535, 0.01009, 0.01940, 0.03817, 0.05814, 0.04769, 0.09315, 0.06305, 0.02477, 0.06231, 0.06696, 0.09451,
         0.04986, 0.09029, 0.07431, 0.15625, 0.08084, 0.02662, 0.01062, 0.00644, 0.00485, 0.00880, 0.01522, 0.01683,
         0.02135, 0.03232, 0.02534, 0.01433, 0.00965, 0.01481, 0.01148, 0.00885, 0.00931, 0.00666, 0.00443, 0.00371,
         0.00344, 0.00356, 0.00530, 0.00621, 0.00619]

    CONS = [1.0192, 1.0185, 1.0179, 1.0172, 1.0167, 1.0161, 1.0156, 1.0152, 1.0147, 1.0143, 1.014, 1.0137, 1.0133,
            1.0130, 1.0126, 1.0123, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0075, 1.0089,
            1.0088,
            1.0086, 1.0085, 1.0083, 1.0082, 1.0081, 1.0079, 1.0078, 1.0077, 1.0076, 1.0075, 1.0074, 1.0072, 1.0071,
            1.0070, 1.0069, 1.0068, 1.0068, 1.0067, 1.0066, 1.0065, 1.0064, 1.0070]
    cdef int FI = 0, CONI = 0

    for I in range(4000):
        EN = object.EG[I]
        ENLG = log(EN)
        GAMMA1 = (EMASS2 + 2 * EN) / EMASS2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA
        # ELASTIC USE LOG INTERPOLATION
        if EN <= XEN[0]:
            QMOM = YMOM[0] * 1e-16
            QELA = YEL[0] * 1e-16
            QBMOM = QMOM
            QBELA = QELA
            PQ[2] = 0.0
        else:
            QMOM = GasUtil.QLSCALE(EN, NEL, YMOM, XEN)
            QELA = GasUtil.QLSCALE(EN, NEL, YEL, XEN)
            QBMOM = GasUtil.QLSCALE(EN, NEL, YVBMOM, XEN)
            QBELA = GasUtil.QLSCALE(EN, NEL, YVBEL, XEN)
            PQ[2] = GasUtil.QLSCALE(EN, NEL, YEPS, XEN) * 1.0e16
            PQ[2] = 1 - PQ[2]

        # CALC CHANGE IN ELASTIC CROSS SECTION DUE TO CHANGE IN ELASTIC
        # SCATTERING FROM BEND MODES ( CHANGE RELATIVE TO X-SECTION AT 293.15K)
        # BEND MODE POPULATION AT 293.15K == AEXT20,GROUND STATE POP. == AGST20

        QMOM = (1.0 - APBEND) * (QMOM - AEXT20 * QBMOM) / AGST20 + APBEND * QBMOM
        QELA = (1.0 - APBEND) * (QELA - AEXT20 * QBELA) / AGST20 + APBEND * QBELA
        PQ[1] = 0.5 + (QELA - QMOM) / (QELA)
        if object.NANISO == 2:
            object.Q[1][I] = QELA
            object.PEQEL[1][I] = PQ[2]
            if EN < 10:
                object.PEQEL[1][I] = 0.0
                object.Q[1][I] = QMOM

        if object.NANISO == 1:
            object.Q[1][I] = QELA
            object.PEQEL[1][I] = PQ[1]
            if EN < 10:
                object.PEQEL[1][I] = 0.5
                object.Q[1][I] = QMOM
        if object.NANISO == 0:
            object.PEQEL[1][I] = 0.5
            object.Q[1][I] = QMOM

        for J in range(11):
            object.QION[J][I] = 0.0
            object.PEQION[J][I] = 0.5
            if object.NANISO == 2:
                object.PEQION[J][I] = 0.0

        # IONISATION CO2+
        if EN > object.EION[0]:
            object.QION[0][I] = GasUtil.CALQIONX(EN, NION1, YION1, XION1, BETA2, 0.67716, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[0]:
            object.PEQION[0][I] = object.PEQEL[1][I - IOFFION[0]]

        # IONISATION CO2+ (A2PIu)
        if EN > object.EION[1]:
            object.QION[1][I] = GasUtil.CALQIONX(EN, NION2, YION2, XION2, BETA2, 0.67716 * 0.385, CONST, object.DEN[I],
                                                 C, AM2)
        if EN > 2 * object.EION[1]:
            object.PEQION[1][I] = object.PEQEL[1][I - IOFFION[1]]

        # IONISATION CO2+ (B2SIGMA+u)
        if EN > object.EION[2]:
            object.QION[2][I] = GasUtil.CALQIONX(EN, NION3, YION3, XION3, BETA2, 0.67716 * 0.220, CONST, object.DEN[I],
                                                 C, AM2)
        if EN > 2 * object.EION[2]:
            object.PEQION[2][I] = object.PEQEL[1][I - IOFFION[2]]

        # DISSOCIATIVE IONISATION O+
        if EN > object.EION[3]:
            object.QION[3][I] = GasUtil.CALQIONX(EN, NION4, YION4, XION4, BETA2, 0.16156, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[3]:
            object.PEQION[3][I] = object.PEQEL[1][I - IOFFION[3]]

        # DISSOCIATIVE IONISATION CO+
        if EN > object.EION[4]:
            object.QION[4][I] = GasUtil.CALQIONX(EN, NION5, YION5, XION5, BETA2, 0.07962, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[4]:
            object.PEQION[4][I] = object.PEQEL[1][I - IOFFION[4]]

        # DISSOCIATIVE IONISATION C+
        if EN > object.EION[5]:
            object.QION[5][I] = GasUtil.CALQIONX(EN, NION6, YION6, XION6, BETA2, 0.07452, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[5]:
            object.PEQION[5][I] = object.PEQEL[1][I - IOFFION[5]]

        # IONISATION CO2++
        if EN > object.EION[6]:
            object.QION[6][I] = GasUtil.CALQIONX(EN, NION7, YION7, XION7, BETA2, 0.00559, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[6]:
            object.PEQION[6][I] = object.PEQEL[1][I - IOFFION[6]]

        # DISSOCIATIVE IONISATION C++
        if EN > object.EION[7]:
            object.QION[7][I] = GasUtil.CALQIONX(EN, NION8, YION8, XION8, BETA2, 0.00076, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[7]:
            object.PEQION[7][I] = object.PEQEL[1][I - IOFFION[7]]

        # DISSOCIATIVE IONISATION O++
        if EN > object.EION[8]:
            object.QION[8][I] = GasUtil.CALQIONX(EN, NION9, YION9, XION9, BETA2, 0.00080, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[8]:
            object.PEQION[8][I] = object.PEQEL[1][I - IOFFION[8]]

        # CARBON K-SHELL IONISATION
        if EN > object.EION[9]:
            object.QION[9][I] = GasUtil.CALQIONREG(EN, NKSHC, YKSHC, XKSHC)
        if EN > 2 * object.EION[9]:
            object.PEQION[9][I] = object.PEQEL[1][I - IOFFION[9]]

        #OXYGEN K-SHELL IONISATION
        if EN > object.EION[10]:
            object.QION[10][I] = GasUtil.CALQIONREG(EN, NKSHO, YKSHO, XKSHO) * 2
        if EN > 2 * object.EION[10]:
            object.PEQION[10][I] = object.PEQEL[1][I - IOFFION[10]]

        #FIX CO2+ X-SECTION FOR SPLIT INTO CO2+ EXCITED STATES
        object.QION[0][I] -= object.QION[1][I] + object.QION[2][I]

        # ATTACHMENT

        object.Q[3][I] = 0.0
        object.QATT[0][I] = 0.0

        if EN > XATT[0]:
            object.Q[3][I] = GasUtil.CALQINP(EN, NATT1, YATT, XATT, 3) * 100

        object.QATT[0][I] = object.Q[3][I]
        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        # ----------------------------------------------------------------------
        #  QUADRUPOLE BORN ROTATIONAL STATES (GERJUOY AND STEIN)
        # ----------------------------------------------------------------------
        # SUPERELASTIC ROTATION

        for J in range(2, 61, 2):
            L = (J / 2)
            object.PEQIN[J - 1][I] = 0.5
            if object.NANISO == 2:
                object.PEQIN[J - 1][I] = 0.0
            if EN >= 4 * abs(object.EIN[J - 1]):
                if object.NANISO == 0:
                    object.PEQIN[J - 1][I] = object.PEQEL[1][I - IOFFN[J - 1]]
            object.QIN[J - 1][I] = PJ[L] * QBK * sqrt(1.0 - object.EIN[J - 1] / EN) * J * (J - 1) / (
                    (2 * J + 1.0) * (2 * J - 1.0))

        # ROTATION
        for J in range(1, 61, 2):
            object.QIN[J - 1][I] = 0.0
            object.PEQIN[J - 1][I] = 0.5
            if object.NANISO == 2:
                object.PEQIN[J - 1][I] = 0.0
            if EN > object.EIN[J - 1]:
                L = (J + 1) / 2
                object.QIN[J - 1][I] = PJ[L - 1] * QBK * sqrt(1.0 - object.EIN[J - 1] / EN) * ((J - 1) + 2.0) * (
                        (J - 1) + 1.0) / ((2 * (J - 1) + 3.0) * (2 * (J - 1) + 1.0))
            if EN >= 4.0 * abs(object.EIN[J - 1]):
                if object.NANISO > 0:
                    object.PEQIN[J - 1][I] = object.PEQEL[1][I - IOFFN[J - 1]]
        # BORN (1/E) FALL OFF IN ROTATONAL X-SEC ABOVE 6.0 EV .
        if EN >= 6.0:
            for J in range(60):
                object.QIN[J][I] *= (6.0 / EN)

        # SUPERELASTIC V2 BEND MODE
        object.QIN[60][I] = 0.0
        object.PEQIN[60][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[60][I] = 0.0
        if EN > 0.0:
            object.QIN[60][I] = GasUtil.CALQINVISELA(EN, NV2, YV2, XV2, APOPV2, object.EIN[61], DEGV2, object.EIN[60],
                                                     AMPV2,0)
        if EN > 3 * abs(object.EIN[60]):
            if object.NANISO > 0:
                object.PEQIN[60][I] = object.PEQEL[1][I - IOFFN[60]]

        # V2 BEND MODE
        object.QIN[61][I] = 0.0
        object.PEQIN[61][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[61][I] = 0.0
        if EN > object.EIN[61]:
            object.QIN[61][I] = GasUtil.CALQINVISELA(EN, NV2, YV2, XV2, APOPGS, 0, 1, object.EIN[61], AMPV2,1)
        if EN > 3 * abs(object.EIN[61]):
            if object.NANISO > 0:
                object.PEQIN[61][I] = object.PEQEL[1][I - IOFFN[61]]

        # SUPERELASTIC 2V2 BEND MODE HARMONIC
        object.QIN[62][I] = 0.0
        object.PEQIN[62][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[62][I] = 0.0
        if EN > 0.0:
            object.QIN[62][I] = GasUtil.CALQINVISELA(EN, N2V2, Y2V2, X2V2, APOP2V2, object.EIN[63], DEG2V2,
                                                     object.EIN[60], 0,0)
        if EN > 3 * abs(object.EIN[62]):
            if object.NANISO > 0:
                object.PEQIN[62][I] = object.PEQEL[1][I - IOFFN[62]]

        # 2V2 BEND MODE HARMONIC
        object.QIN[63][I] = 0.0
        object.PEQIN[63][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[63][I] = 0.0
        if EN > object.EIN[63]:
            object.QIN[63][I] = GasUtil.CALQINVISELA(EN, N2V2, Y2V2, X2V2, APOPGS, 0, 1, object.EIN[63], 0,1)
        if EN > 3 * abs(object.EIN[63]):
            if object.NANISO > 0:
                object.PEQIN[63][I] = object.PEQEL[1][I - IOFFN[63]]

        # SUPERELASTIC V1 SYMMETRIC STRETCH
        object.QIN[64][I] = 0.0
        object.PEQIN[64][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[64][I] = 0.0
        if EN > 0.0:
            object.QIN[64][I] = GasUtil.CALQINVISELA(EN, NV1, YV1, XV1, APOPV1, object.EIN[65], DEGV1, object.EIN[63],
                                                     0,1)
        if EN > 3 * abs(object.EIN[64]):
            if object.NANISO > 0:
                object.PEQIN[64][I] = object.PEQEL[1][I - IOFFN[64]]

        # V1 SYMMETRIC STRETCH
        object.QIN[65][I] = 0.0
        object.PEQIN[65][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[65][I] = 0.0
        if EN > object.EIN[65]:
            object.QIN[65][I] = GasUtil.CALQINVISELA(EN, NV1, YV1, XV1, APOPGS, 0, 1, object.EIN[63], 0,1)
        if EN > 3 * abs(object.EIN[65]):
            if object.NANISO > 0:
                object.PEQIN[65][I] = object.PEQEL[1][I - IOFFN[65]]

        # SUPERELASTIC 3V2 + V12
        object.QIN[66][I] = 0.0
        object.PEQIN[66][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[66][I] = 0.0
        if EN > 0.0:
            object.QIN[66][I] = GasUtil.CALQINVISELA(EN, N3V2, Y3V2, X3V2, APOP3V2, object.EIN[67], DEG3V2,
                                                     object.EIN[63], 0,1)
        if EN > 3 * abs(object.EIN[67]):
            if object.NANISO > 0:
                object.PEQIN[66][I] = object.PEQEL[1][I - IOFFN[66]]

        # 3V2 + V12
        object.QIN[67][I] = 0.0
        object.PEQIN[67][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[67][I] = 0.0
        if EN > object.EIN[67]:
            object.QIN[67][I] = GasUtil.CALQINVISELA(EN, N3V2, Y3V2, X3V2, APOPGS, 0, 1, object.EIN[63], 0,1)
        if EN > 3 * abs(object.EIN[67]):
            if object.NANISO > 0:
                object.PEQIN[67][I] = object.PEQEL[1][I - IOFFN[67]]

        # SUPERELASTIC V3 ASYMMETRIC STRETCH
        object.QIN[68][I] = 0.0
        object.PEQIN[68][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[68][I] = 0.0
        if EN > 0.0:
            object.QIN[68][I] = GasUtil.CALQINVISELA(EN, NV3, YV3, XV3, APOPV3, object.EIN[69], DEGV3, object.EIN[68],
                                                     AMPV3,0)
        if EN > 3 * abs(object.EIN[68]):
            if object.NANISO > 0:
                object.PEQIN[68][I] = object.PEQEL[1][I - IOFFN[68]]

        # V3  ASYMMETRIC STRETCH
        object.QIN[69][I] = 0.0
        object.PEQIN[69][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[69][I] = 0.0
        if EN > object.EIN[69]:
            object.QIN[69][I] = GasUtil.CALQINVISELA(EN, NV3, YV3, XV3, APOPGS, 0, 1, object.EIN[69], AMPV3,1)
        if EN > 3 * abs(object.EIN[69]):
            if object.NANISO > 0:
                object.PEQIN[69][I] = object.PEQEL[1][I - IOFFN[69]]

        # 4V2 + 2V1 + V12V2 POLYAD 3
        object.QIN[70][I] = 0.0
        object.PEQIN[70][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[70][I] = 0.0
        if EN > object.EIN[70]:
            object.QIN[70][I] = GasUtil.CALQINP(EN, NPD3, YVPD3, XVPD3, 1) * 100
        if EN > 3 * abs(object.EIN[70]):
            if object.NANISO > 0:
                object.PEQIN[70][I] = object.PEQEL[1][I - IOFFN[70]]

        # 3V2V1 + 2V1V2
        object.QIN[71][I] = 0.0
        object.PEQIN[71][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[71][I] = 0.0
        if EN > object.EIN[71]:
            object.QIN[71][I] = GasUtil.CALQINP(EN, NV130, YV130, XV130, 1) * 100
        if EN > 3 * abs(object.EIN[71]):
            if object.NANISO > 0:
                object.PEQIN[71][I] = object.PEQEL[1][I - IOFFN[71]]

        # POLYAD 4
        object.QIN[72][I] = 0.0
        object.PEQIN[72][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[72][I] = 0.0
        if EN > object.EIN[72]:
            object.QIN[72][I] = GasUtil.CALQINP(EN, NPD4, YVPD4, XVPD4, 1) * 100
        if EN > 3 * abs(object.EIN[72]):
            if object.NANISO > 0:
                object.PEQIN[72][I] = object.PEQEL[1][I - IOFFN[72]]

        # POLYAD 5
        object.QIN[73][I] = 0.0
        object.PEQIN[73][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[73][I] = 0.0
        if EN > object.EIN[73]:
            object.QIN[73][I] = GasUtil.CALQINP(EN, NPD5, YVPD5, XVPD5, 1) * 100
        if EN > 3 * abs(object.EIN[73]):
            if object.NANISO > 0:
                object.PEQIN[73][I] = object.PEQEL[1][I - IOFFN[73]]

        # POLYAD 6
        object.QIN[74][I] = 0.0
        object.PEQIN[74][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[74][I] = 0.0
        if EN > object.EIN[74]:
            object.QIN[74][I] = GasUtil.CALQINP(EN, NPD6, YVPD6, XVPD6, 1) * 100
        if EN > 3 * abs(object.EIN[74]):
            if object.NANISO > 0:
                object.PEQIN[74][I] = object.PEQEL[1][I - IOFFN[74]]

        # POLYAD 7
        object.QIN[75][I] = 0.0
        object.PEQIN[75][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[75][I] = 0.0
        if EN > object.EIN[75]:
            object.QIN[75][I] = GasUtil.CALQINP(EN, NPD7, YVPD7, XVPD7, 1) * 100
        if EN > 3 * abs(object.EIN[75]):
            if object.NANISO > 0:
                object.PEQIN[75][I] = object.PEQEL[1][I - IOFFN[75]]

        # POLYAD 8
        object.QIN[76][I] = 0.0
        object.PEQIN[76][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[76][I] = 0.0
        if EN > object.EIN[76]:
            object.QIN[76][I] = GasUtil.CALQINP(EN, NPD8, YVPD8, XVPD8, 1) * 100
        if EN > 3 * abs(object.EIN[76]):
            if object.NANISO > 0:
                object.PEQIN[76][I] = object.PEQEL[1][I - IOFFN[76]]

        # POLYAD 9
        object.QIN[77][I] = 0.0
        object.PEQIN[77][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[77][I] = 0.0
        if EN > object.EIN[77]:
            object.QIN[77][I] = GasUtil.CALQINP(EN, NPD9, YVPD9, XVPD9, 1) * 100
        if EN > 3 * abs(object.EIN[77]):
            if object.NANISO > 0:
                object.PEQIN[77][I] = object.PEQEL[1][I - IOFFN[77]]

        # SUM OF HIGHER POLYADS
        object.QIN[78][I] = 0.0
        object.PEQIN[78][I] = 0.5
        if object.NANISO == 2:
            object.PEQIN[78][I] = 0.0
        if EN > object.EIN[78]:
            object.QIN[78][I] = GasUtil.CALQINP(EN, NPDH, YVPDH, XVPDH, 1) * 100
        if EN > 3 * abs(object.EIN[78]):
            if object.NANISO > 0:
                object.PEQIN[78][I] = object.PEQEL[1][I - IOFFN[78]]

        FI = 0
        CONI = 0
        for J in range(79, 89):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.0
            if object.NANISO == 2:
                object.PEQIN[J][I] = 0.0
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2]) * CONS[CONI]
            if EN > 2 * object.EIN[J]:
                if object.NANISO > 0:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        # TRIPLET
        object.QIN[89][I] = 0.0
        object.PEQIN[89][I] = 0.0
        if object.NANISO == 2:
            object.PEQIN[89][I] = 0.0
        if EN > object.EIN[89]:
            object.QIN[89][I] = GasUtil.CALQINP(EN, NTRP1, YTRP1, XTRP1, 2) * 100
        if EN > 2.0 * object.EIN[89]:
            if object.NANISO > 0:
                object.PEQIN[89][I] = object.PEQEL[1][I - IOFFN[89]]

        for J in range(90, 98):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.0
            if object.NANISO == 2:
                object.PEQIN[J][I] = 0.0
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2]) * CONS[CONI]
            if EN > 2 * object.EIN[J]:
                if object.NANISO > 0:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        # TRIPLET
        object.QIN[98][I] = 0.0
        object.PEQIN[98][I] = 0.0
        if object.NANISO == 2:
            object.PEQIN[98][I] = 0.0
        if EN > object.EIN[98]:
            object.QIN[98][I] = GasUtil.CALQINP(EN, NTRP2, YTRP2, XTRP2, 2) * 100
        if EN > 2.0 * object.EIN[98]:
            if object.NANISO > 0:
                object.PEQIN[98][I] = object.PEQEL[1][I - IOFFN[98]]

        for J in range(99, 143):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.0
            if object.NANISO == 2:
                object.PEQIN[J][I] = 0.0
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2]) * CONS[CONI]
            if EN > 2 * object.EIN[J]:
                if object.NANISO > 0:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        #LOAD BREMSSTRAHLUNG X-SECTIONS
        object.QIN[144][I] = 0.0
        object.QIN[145][I] = 0.0
        if EN > 1000:
            object.QIN[144][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z6T, EBRM) * 1e-8
            object.QIN[145][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z8T, EBRM) * 2e-8

        # SUM ROTATION
        SUMR = 0.0
        for J in range(60):
            SUMR += object.QIN[J][I]

        # SUM VIBRATION
        SUMV = 0.0
        for J in range(60, 79):
            SUMV += object.QIN[J][I]

        # SUM DIPOLE + TRIPLET EXCITATION
        SUME = 0.0
        for J in range(79, 144):
            SUME += object.QIN[J][I]

        # SUM TRIPLET EXCITATION
        SUMTRP = object.QIN[89][I] + object.QIN[98][I] + object.QIN[143][I]
        # GET SUM DIPOLE
        SUME = SUME - SUMTRP
        SUMEXC = SUME + SUMTRP
        # SUM IONISATION
        SUMION = 0.0

        for J in range(11):
            SUMION += object.QION[J][I]
        # GET CORRECT ELASTIC X-SECTION
        object.Q[1][I] -= SUMR

        object.Q[0][I] = QELA + object.Q[3][I] + SUMV + SUME + SUMTRP + SUMION


    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
    return
