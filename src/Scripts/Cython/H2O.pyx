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
cdef void Gas14(Gas*object):
    gd = np.load('gases.npy').item()

    cdef double XEL[159], YEL[159], XMT[156], YMT[156], XEPS[156], YEPS[156], XVIB1[17], YVIB1[17], XVIB2[18], YVIB2[18], XVIB3[12], YVIB3[12],
    cdef double XION[55], YIONC[55], YIONG[55], XION1[31], YION1[31], XION2[28], YION2[28], XION3[28], YION3[28], XION4[26], YION4[26], XION5[25],
    cdef double YION5[25], XION6[23], YION6[23], XION7[21], YION7[21], XION8[17], YION8[17], XKSH[81], YKSH[81], XATT1[38], YATT1[38]
    cdef double XATT2[30], YATT2[30], XATT3[28], YATT3[28], XTRP1[11], YTRP1[11], XTRP2[10], YTRP2[10], XTRP3[10], YTRP3[10],
    cdef double XTRP4[9], YTRP4[9], XNUL1[12], YNUL1[12], XNUL2[33], YNUL2[33], XNUL3[20], YNUL3[20], XNUL4[18], YNUL4[18],
    cdef double XSECDUM[210], ENROT[145], ENRTS[145], YEPSR[145], YMTRT[145], Z8T[25], EBRM[25]

    cdef int IOFFN[250], IOFFION[9]

    cdef double PJ[100], ELEV[100], AJL[100], SALPHA[105], EROT[105], AJIN[210]
    cdef int IMAP[210]
    EROT = gd['gas14/EROT']
    AJL = gd['gas14/AJL']
    ELEV = gd['gas14/ELEV']
    SALPHA = gd['gas14/SALPHA']
    AJIN = gd['gas14/AJIN']
    IMAP = gd['gas14/IMAP']
    XEL = gd['gas14/XEL']
    YEL = gd['gas14/YEL']
    XMT = gd['gas14/XMT']
    YMT = gd['gas14/YMT']
    XEPS = gd['gas14/XEPS']
    YEPS = gd['gas14/YEPS']
    XVIB1 = gd['gas14/XVIB1']
    YVIB1 = gd['gas14/YVIB1']
    XVIB2 = gd['gas14/XVIB2']
    YVIB2 = gd['gas14/YVIB2']
    XVIB3 = gd['gas14/XVIB3']
    YVIB3 = gd['gas14/YVIB3']
    XION = gd['gas14/XION']
    YIONC = gd['gas14/YIONC']
    YIONG = gd['gas14/YIONG']
    XION1 = gd['gas14/XION1']
    YION1 = gd['gas14/YION1']
    XION2 = gd['gas14/XION2']
    YION2 = gd['gas14/YION2']
    XION3 = gd['gas14/XION3']
    YION3 = gd['gas14/YION3']
    XION4 = gd['gas14/XION4']
    YION4 = gd['gas14/YION4']
    XION5 = gd['gas14/XION5']
    YION5 = gd['gas14/YION5']
    XION6 = gd['gas14/XION6']
    YION6 = gd['gas14/YION6']
    XION7 = gd['gas14/XION7']
    YION7 = gd['gas14/YION7']
    XION8 = gd['gas14/XION8']
    YION8 = gd['gas14/YION8']
    XKSH = gd['gas14/XKSH']
    YKSH = gd['gas14/YKSH']
    XATT1 = gd['gas14/XATT1']
    YATT1 = gd['gas14/YATT1']
    XATT2 = gd['gas14/XATT2']
    YATT2 = gd['gas14/YATT2']
    XATT3 = gd['gas14/XATT3']
    YATT3 = gd['gas14/YATT3']
    XTRP1 = gd['gas14/XTRP1']
    YTRP1 = gd['gas14/YTRP1']
    XTRP2 = gd['gas14/XTRP2']
    YTRP2 = gd['gas14/YTRP2']
    XTRP3 = gd['gas14/XTRP3']
    YTRP3 = gd['gas14/YTRP3']
    XTRP4 = gd['gas14/XTRP4']
    YTRP4 = gd['gas14/YTRP4']
    XNUL1 = gd['gas14/XNUL1']
    YNUL1 = gd['gas14/YNUL1']
    XNUL2 = gd['gas14/XNUL2']
    YNUL2 = gd['gas14/YNUL2']
    XNUL3 = gd['gas14/XNUL3']
    YNUL3 = gd['gas14/YNUL3']
    XNUL4 = gd['gas14/XNUL4']
    YNUL4 = gd['gas14/YNUL4']
    ENROT = gd['gas14/ENROT']
    ENRTS = gd['gas14/ENRTS']
    YEPSR = gd['gas14/YEPSR']
    YMTRT = gd['gas14/YMTRT']
    Z8T = gd['gas14/Z8T']
    EBRM = gd['gas14/EBRM']

    cdef double A0, RY, CONST, EMASS2, API, BBCONST, AM2, C, AUGK, AMPROT,
    cdef int NBREM, i, j, I, J, NTRANG
    A0 = 0.52917720859e-08
    RY = 13.60569193
    CONST = 1.873884e-20
    EMASS2 = 1021997.804
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / EMASS2
    #BORN BETHE VALUES FOR IONISATION
    AM2 = 2.895
    C = 30.7
    # AVERAGE AUGER EMISSION FROM OXYGEN KSHELL
    AUGK = 2.0
    AMPROT = 0.98
    object.NION = 9
    object.NATT = 3
    object.NIN = 250
    object.NNULL = 4
    NBREM = 25
    NRTANG = 145

    # USE OKRIMOVSKKY
    for J in range(6):
        object.KEL[J] = 2
    for J in range(object.NIN):
        object.KIN[J] = 2

    cdef int NELA, NMMT, NEPS, NVIB1, NVIB2, NVIB3, NIONC, NION1, NION2, NION3, NION4, NION5, NION6, NION7, NION8, NKSH, NATT1, NATT2
    cdef int NATT3, NTRP1, NTRP2, NTRP3, NTRP4, NUL1, NUL2, NUL3, NUL4,
    NELA = 159
    NMMT = 156
    NEPS = 156
    NVIB1 = 17
    NVIB2 = 18
    NVIB3 = 12
    NIONC = 55
    NION1 = 31
    NION2 = 28
    NION3 = 28
    NION4 = 26
    NION5 = 25
    NION6 = 23
    NION7 = 21
    NION8 = 17
    NKSH = 81
    NATT1 = 38
    NATT2 = 30
    NATT3 = 28
    NTRP1 = 11
    NTRP2 = 10
    NTRP3 = 10
    NTRP4 = 9
    NUL1 = 12
    NUL2 = 33
    NUL3 = 20
    NUL4 = 18

    # SCALING OF NULL COLLISIONS
    object.SCLN[0:4] = [1.0, 1.0, 1.0, 1.0]
    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, GPARA, GORTHO, DBA, DRAT, DBK, RSUM, EOBY[9], ENRT, AL
    cdef int L2,

    object.E = [0.0, 1.0, 12.617, 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * EMASS / (18.01528 * AMU)

    object.EION[0:9] = [12.617, 18.1, 18.72, 21.0, 23.0, 35.4, 45.0, 70.0, 532.0]

    # DBA IS DIPOLE MOMENT
    # DRAT IS RATIO OF MOMENTUM TRANSFER TO TOTAL X-SECTION FOR DIPOLE
    GPARA = 1.0
    GORTHO = 3.0
    DBA = 0.728
    DRAT = 0.07
    DBK = 8.37758 * RY * (DBA * A0) ** 2

    #CALCULATE POPULATION DENSITIES OF ROTATIONAL LEVELS
    for J in range(1, 100, 2):
        PJ[J - 1] = GPARA * (2.0 * AJL[J - 1] + 1.0) * exp(-1 * ELEV[J - 1] * 1e-3 / object.AKT)
    for J in range(2, 101, 2):
        PJ[J - 1] = GORTHO * (2.0 * AJL[J - 1] + 1.0) * exp(-1 * ELEV[J - 1] * 1e-3 / object.AKT)
    RSUM = 0.0
    for J in range(100):
        RSUM += PJ[J]
    for J in range(100):
        PJ[J] /= RSUM

    for J in range(1, 106):
        object.EIN[(2 * J) - 2] = EROT[J - 1] * 1e-3
        object.EIN[(2 * J) - 1] = -1 * EROT[J - 1] * 1e-3
    object.EIN[210:250] = [-0.1977, 0.1977, 0.4535, 0.919, 7.04, 6.8425, 7.2675, 7.7725, 8.3575, 9.1, 8.91, 9.43, 9.95,
                           10.47, 9.95, 9.994, 10.172, 10.39, 10.575, 10.78, 11.01, 11.122, 11.377, 11.525, 11.75,
                           11.94, 12.08, 12.24, 12.34, 12.45, 13.0, 13.117, 14.117, 15.117, 16.117, 17.117, 18.117,
                           19.117, 20.117, 21.117]

    for J in range(object.NION):
        EOBY[J] = object.EION[0] * 0.93

    for J in range(object.NION):
        object.NC0[J] = 0
        object.EC0[J] = 0.0
        object.WK[J] = 0.0
        object.EFL[J] = 0.0
        object.NG1[J] = 0
        object.EG1[J] = 0.0
        object.NG2[J] = 0
        object.EG2[J] = 0.0
    # DOUBLE CHARGED STATES
    object.NC0[5] = 1
    object.EC0[5] = 6.0
    object.NC0[6] = 1
    object.EC0[6] = 6.0
    object.NC0[7] = 1
    object.EC0[7] = 6.0
    # FLUORESCENCE DATA
    object.NC0[8] = 3
    object.EC0[8] = 485
    object.WK[8] = 0.0069
    object.EFL[8] = 518
    object.NG1[8] = 1
    object.EG1[8] = 480
    object.NG2[8] = 2
    object.EG2[8] = 5.0

    for j in range(0, object.NION):
        for i in range(0, 4000):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i - 1
                break

    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION
    cdef int NL = 0
    for NL in range(object.NIN):
        for i in range(4000):
            if object.EG[i] > abs(object.EIN[NL]):
                IOFFN[NL] = i - 1
                break

    for I in range(object.NIN):
        object.PENFRA[0][I] = 0.0

    for J in range(214, object.NIN):
        object.PENFRA[0][J] = 0.0
        object.PENFRA[1][J] = 1.0
        object.PENFRA[2][J] = 1.0

    cdef double APOPV1, APOPGS, APOPSUM, GAMMA1, GAMMA2, BETA, BETA2, EN, QELA, QMMT, PQ[3], EPS, QCOUNT, QGROSS, EPOINT
    cdef double F[32]
    F = [.003437, .017166, .019703, .005486, .006609, .030025, .030025, .006609, .005200, .014000, .010700, .009200,
         .006900, .021800, .023900, .013991, .009905, .023551, .007967, .018315, .011109, .008591, .028137, .119100,
         .097947, .039540, .042191, .059428, .052795, .024912, .010524, .002614]
    cdef int FI
    # CALC POPULATION OF LOW ENERGY VIBRATIONAL STATE
    APOPV1 = exp(object.EIN[210] / object.AKT)
    APOPGS = 1.0
    APOPSUM = APOPGS + APOPV1
    APOPV1 = APOPV1 / APOPSUM

    #KEEP APOPGS=1 TO ALLOW FOR EXCITATIONS FROM UPPER STATE

    for I in range(4000):
        EN = object.EG[I]
        GAMMA1 = (EMASS2 + 2 * EN) / EMASS2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA

        # ELASTIC
        if EN <= XEL[0]:
            QELA = YEL[0] * 1e-16
        else:
            QELA = GasUtil.QLSCALE(EN, NELA, YEL, XEL)

        #MOMENTUM TRANSFER ELASTIC
        if EN <= XMT[0]:
            QMMT = YMT[0] * 1e-16
        else:
            QMMT = GasUtil.QLSCALE(EN, NMMT, YMT, XMT)

        #ANGULAR DISTRIBUTION FUNCTION OKHRIMOVSKY
        if EN <= XEPS[0]:
            PQ[2] = YEPS[0]
        else:
            PQ[2] = GasUtil.CALPQ3(EN, NEPS, YEPS, XEPS)
        PQ[2] = 1 - PQ[2]
        object.Q[1][I] = QELA
        object.PEQEL[1][I] = PQ[2]

        # IONISATION CALCULATION
        for J in range(object.NION):
            object.PEQION[J][I] = 0.0
            object.QION[J][I] = 0.0
        #IF ENERGY LESS THAN 5KEV CALCULATE TOTAL COUNTING AND GROSS IONISATION
        if EN <= 5000:
            if EN > object.EION[0]:
                QCOUNT = GasUtil.CALQIONX(EN, NIONC, YIONC, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
                QGROSS = GasUtil.CALQIONX(EN, NIONC, YIONG, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
        else:
            QCOUNT = GasUtil.CALQIONX(EN, NIONC, YIONC, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
            QGROSS = QCOUNT * 1.022

        #IONISATION TO H2O+
        if EN <= XION1[NION1 - 1] and EN > XION1[0]:
            object.QION[0][I] = GasUtil.CALQION(EN, NION1, YION1, XION1)
            if object.QION[0][I] == 0:
                object.QION[0][I] = QCOUNT * 0.62996

        #IONISATION TO OH+
        if EN <= XION2[NION2 - 1] and EN > XION2[0]:
            object.QION[1][I] = GasUtil.CALQION(EN, NION2, YION2, XION2)
            if object.QION[1][I] == 0:
                object.QION[1][I] = QCOUNT * 0.19383

        #IONISATION TO H+
        if EN <= XION3[NION3 - 1] and EN > XION3[0]:
            object.QION[2][I] = GasUtil.CALQION(EN, NION3, YION3, XION3)
            if object.QION[2][I] == 0:
                object.QION[2][I] = QCOUNT * 0.13275

        #IONISATION TO O+
        if EN <= XION4[NION4 - 1] and EN > XION4[0]:
            object.QION[3][I] = GasUtil.CALQION(EN, NION4, YION4, XION4)
            if object.QION[3][I] == 0:
                object.QION[3][I] = QCOUNT * 0.02129

        #IONISATION TO H2+
        if EN <= XION5[NION5 - 1] and EN > XION5[0]:
            object.QION[4][I] = GasUtil.CALQION(EN, NION5, YION5, XION5)
            if object.QION[4][I] == 0:
                object.QION[4][I] = QCOUNT * 0.00035

        #IONISATION TO H+ + OH+
        if EN <= XION6[NION6 - 1] and EN > XION6[0]:
            object.QION[5][I] = GasUtil.CALQION(EN, NION6, YION6, XION6)
            if object.QION[5][I] == 0:
                object.QION[5][I] = QCOUNT * 0.01395

        #IONISATION TO H+ + O+
        if EN <= XION7[NION7 - 1] and EN > XION7[0]:
            object.QION[6][I] = GasUtil.CALQION(EN, NION7, YION7, XION7)
            if object.QION[6][I] == 0:
                object.QION[6][I] = QCOUNT * 0.00705

        #IONISATION TO O++
        if EN <= XION8[NION8 - 1] and EN > XION8[0]:
            object.QION[7][I] = GasUtil.CALQION(EN, NION8, YION8, XION8)
            if object.QION[7][I] == 0:
                object.QION[7][I] = QCOUNT * 0.00085

        #IONISATION TO OXYGEN K-SHELL
        if EN > XKSH[0]:
            object.QION[8][I] = GasUtil.CALQIONREG(EN, NKSH, YKSH, XKSH)

        for J in range(object.NION):
            if EN > 2 * object.EION[J]:
                object.PEQION[J][I] = object.PEQEL[1][I - IOFFION[J]]

        # ATTACHMENT H-
        object.Q[3][I] = 0.0
        object.QATT[0][I] = 0.0
        if EN > XATT1[0] and EN < XATT1[NATT1 - 1]:
            object.QATT[0][I] = GasUtil.QLSCALE(EN, NATT1, YATT1, XATT1) * 1e-5

        # ATTACHMENT O-
        object.QATT[1][I] = 0.0
        if EN > XATT2[0] and EN < XATT2[NATT2 - 1]:
            object.QATT[1][I] = GasUtil.QLSCALE(EN, NATT2, YATT2, XATT2) * 1e-5

        #ATTACHMENT OH-
        object.QATT[2][I] = 0.0
        if EN > XATT3[0] and EN < XATT3[NATT3 - 1]:
            object.QATT[2][I] = GasUtil.QLSCALE(EN, NATT3, YATT3, XATT3) * 1e-5

        object.Q[3][I] = 0.0
        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0
        # ZERO INELASTIC ARRAYS
        for J in range(object.NIN):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.0

        # DIPOLE BORN ROTATIONAL STATES
        ENRT = sqrt(EN)

        # SUPER ELASTIC ROTATIONAL COLLISIONS
        for J in range(2, 211, 2):
            AL = AJIN[J - 1]
            L2 = J / 2
            object.QIN[J - 1][I] = DBK * SALPHA[L2 - 1] * PJ[IMAP[J - 1] - 1] * log(
                (ENRT + sqrt(EN - object.EIN[J - 1])) / (sqrt(EN - object.EIN[J - 1]) - ENRT)) / (
                                           (2.0 * AL + 1.0) * EN) * AMPROT
            if EN > 2000:
                object.QIN[J - 1][I] = 0.0
                continue

            EPOINT = EN / abs(object.EIN[J - 1])
            #TODO: PRINT ERROR STATEMENT

            object.PEQIN[J - 1][I] = 1.0 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENRTS)
            XSECDUM[J - 1] = GasUtil.CALPQ3(EPOINT, NRTANG, YMTRT, ENRTS) * object.QIN[J - 1][I]

        # ROTATIONAL COLLISIONS
        for J in range(1, 210, 2):
            object.QIN[J - 1][I] = 0.0
            if EN > object.EIN[J - 1]:
                AL = AJIN[J - 1]
                L2 = (J + 1) / 2
                object.QIN[J - 1][I] = DBK * SALPHA[L2 - 1] * PJ[IMAP[J - 1] - 1] * log(
                    (ENRT + sqrt(EN - object.EIN[J - 1])) / (ENRT - sqrt(EN - object.EIN[J - 1]))) / (
                                               (2.0 * AL + 1.0) * EN) * AMPROT
                if EN > 2000:
                    object.QIN[J - 1][I] = 0.0
                    continue

                EPOINT = EN / abs(object.EIN[J - 1])
                #TODO: PRINT ERROR STATEMENT

                object.PEQIN[J - 1][I] = 1.0 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENRTS)
                XSECDUM[J - 1] = GasUtil.CALPQ3(EPOINT, NRTANG, YMTRT, ENRTS) * object.QIN[J - 1][I]

        # VIBRATION BEND V2 SUPERELASTIC (DIPOLE 1/E FALL OFF ABOVE ENERGY OF
        # XVIB1(NVIB1) EV )
        object.QIN[210][I] = 0.0
        if EN > 0.0:
            object.QIN[210][I] = GasUtil.CALQINP(EN+object.EIN[211], NVIB1, YVIB1, XVIB1,1)*APOPV1*100/EN
            if EN+object.EIN[211] <= XVIB1[NVIB1-1]:
                object.QIN[210][I] *=(EN + object.EIN[211])
            object.PEQIN[210][I] = object.PEQEL[1][I - IOFFN[210]]

        # VIBRATION BEND V2  (DIPOLE 1/E FALL OFF ABOVE ENERGY OF
        # XVIB1(NVIB1) EV )
        object.QIN[211][I] = 0.0
        if EN > object.EIN[211]:
            object.QIN[211][I] = GasUtil.CALQINP(EN, NVIB1, YVIB1, XVIB1,1)*APOPGS*100
            object.PEQIN[211][I] = object.PEQEL[1][I - IOFFN[211]]
            # CALCULATE DIPOLE ANGULAR DISTRIBUTION FACTOR FOR TRANSITION
            EPOINT = EN / abs(object.EIN[211])
            if EPOINT <= 500:
                object.PEQIN[211][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
            else:
                object.PEQIN[211][I] = object.PEQEL[1][I - IOFFN[211]]

        # VIBRATION STRETCH V1+V3
        object.QIN[212][I] = 0.0
        if EN > object.EIN[212]:
            object.QIN[212][I] = GasUtil.CALQINP(EN, NVIB2, YVIB2, XVIB2, 1.5) * 100
            if EN < 1.5:
                object.PEQIN[212][I] = 0.0
            else:
                EPOINT = EN / abs(object.EIN[212])
                if EPOINT <= 500:
                    object.PEQIN[212][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEQIN[212][I] = object.PEQEL[1][I - IOFFN[212]]

        # VIBRATION HARMONICS NV2+ NV1+NV3
        object.QIN[213][I] = 0.0
        if EN > object.EIN[213]:
            object.QIN[213][I] = GasUtil.CALQINP(EN, NVIB3, YVIB3, XVIB3, 1.5) * 100
            EPOINT = EN / abs(object.EIN[213])
            if EPOINT <= 500:
                object.PEQIN[213][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
            else:
                object.PEQIN[213][I] = object.PEQEL[1][I - IOFFN[213]]

        # TRIPLET 3B1
        if EN > object.EIN[214]:
            if EN <= XTRP1[NTRP1 - 1]:
                object.QIN[214][I] = GasUtil.QLSCALE(EN, NTRP1, YTRP1, XTRP1)
            else:
                object.QIN[214][I] = YTRP1[NTRP1 - 1] * (XTRP1[NTRP1 - 1] / EN) ** 1.5 * 1e-16

            if EN > 2.0 * object.EIN[214]:
                object.PEQIN[214][I] = object.PEQEL[1][I - IOFFN[214]]

        FI = 0
        # EXCITATION  1B1 (7.48EV LEVEL SPLIT INTO 4 GROUPS)
        for J in range(215, 219):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
                EPOINT = EN / abs(object.EIN[J])
                if EPOINT <= 500:
                    object.PEQIN[J][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1

        # TRIPLET 3A2 + 1A2 + 3A1 9.1EV
        if EN > object.EIN[219]:
            if EN <= XTRP2[NTRP2 - 1]:
                object.QIN[219][I] = GasUtil.QLSCALE(EN, NTRP2, YTRP2, XTRP2)
            else:
                object.QIN[219][I] = YTRP2[NTRP2 - 1] * (XTRP2[NTRP2 - 1] / EN) ** 1.5 * 1e-16

            if EN > 2.0 * object.EIN[219]:
                object.PEQIN[219][I] = object.PEQEL[1][I - IOFFN[219]]

        # EXCITATION  1A1 (9.69EV LEVEL SPLIT INTO 4 GROUPS)
        for J in range(220, 224):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
                EPOINT = EN / abs(object.EIN[J])
                if EPOINT <= 500:
                    object.PEQIN[J][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1

        # TRIPLET 3B1 9.95EV
        if EN > object.EIN[224]:
            if EN <= XTRP3[NTRP3 - 1]:
                object.QIN[224][I] = GasUtil.QLSCALE(EN, NTRP3, YTRP3, XTRP3)
            else:
                object.QIN[224][I] = YTRP3[NTRP3 - 1] * (XTRP3[NTRP3 - 1] / EN) ** 1.5 * 1e-16

            if EN > 2.0 * object.EIN[224]:
                object.PEQIN[224][I] = object.PEQEL[1][I - IOFFN[224]]

        # EXCITATION  1B1 (3pa1) 9.994 EV
        for J in range(225, 240):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
                EPOINT = EN / abs(object.EIN[J])
                if EPOINT <= 500:
                    object.PEQIN[J][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1

        # TRIPLET SUM OF HIGHER TRIPLETS   13.0 EV
        if EN > object.EIN[240]:
            if EN <= XTRP4[NTRP4 - 1]:
                object.QIN[240][I] = GasUtil.QLSCALE(EN, NTRP4, YTRP4, XTRP4)
            else:
                object.QIN[240][I] = YTRP4[NTRP4 - 1] * (XTRP4[NTRP4 - 1] / EN) ** 1.5 * 1e-16

            if EN > 2.0 * object.EIN[240]:
                object.PEQIN[240][I] = object.PEQEL[1][I - IOFFN[240]]

        # EXCITATION  1B1 (3pa1) 9.994 EV
        for J in range(241, 250):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
                if object.QIN[J][I] < 0.0:
                    object.QIN[J][I] = 0.0
                EPOINT = EN / abs(object.EIN[J])
                if EPOINT <= 500:
                    object.PEQIN[J][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI += 1

        # LOAD NULL COLLISIONS
        #  OH PRODUCTION FROM DISSOCIATION HARB ET AL J.CHEM.PHYS. 115(2001)5507
        # SCALED ABOVE 200EV BY 1/ENERGY
        object.QNULL[0][I] = 0.0
        if EN > XNUL1[0]:
            object.QNULL[0][I] = GasUtil.CALQINP(EN, NUL1,YNUL1,XNUL1, 1)*100*object.SCLN[0]

        # LIGHT EMISSION FROM OH(A2-X) MOHLMANN AND DEHEER CHEM.PHYS.19(1979)233
        object.QNULL[1][I] = 0.0
        if EN > XNUL2[0]:
            object.QNULL[1][I] = GasUtil.CALQINP(EN, NUL2,YNUL2,XNUL2, 1)*100*object.SCLN[1]

        # LIGHT EMISSION FROM H(3-2) , MOHLMANN AND DEHEER CHEM.PHYS.19(1979)233
        object.QNULL[2][I] = 0.0
        if EN > XNUL3[0]:
            object.QNULL[2][I] = GasUtil.CALQINP(EN, NUL3,YNUL3,XNUL3, 1)*100*object.SCLN[2]

        # LIGHT EMISSION FROM H(2-1) , MOHLMANN AND DEHEER CHEM.PHYS.19(1979)233
        object.QNULL[3][I] = 0.0
        if EN > XNUL4[0]:
            object.QNULL[3][I] = GasUtil.CALQINP(EN, NUL4,YNUL4,XNUL4, 1)*100*object.SCLN[3]


    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
    return