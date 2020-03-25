from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow
cimport libc.math
import numpy as np
cimport numpy as np
import sys
from Gas cimport Gas
from cython.parallel import prange
cimport GasUtil
import os
sys.path.append('../hdf5_python')
import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.fast_getattr(True)
cdef void Gas14(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for H2O gas.
    """
    gd = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"gases.npy")).item()

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

    cdef double A0, RY, CONST, ElectronMass2, API, BBCONST, AM2, C, AUGK, AMPROT,
    cdef int NBREM, i, j, I, J, NTRANG
    A0 = 0.52917720859e-08
    RY = <float> (13.60569193)
    CONST = 1.873884e-20
    ElectronMass2 = <float> (1021997.804)
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / ElectronMass2
    #BORN BETHE VALUES FOR IONISATION
    AM2 = <float> (2.895)
    C = <float> (30.7)
    # AVERAGE AUGER EMISSION FROM OXYGEN KSHELL
    AUGK = 2.0
    AMPROT = <float> (0.98)
    object.N_Ionization = 9
    object.N_Attachment = 3
    object.N_Inelastic = 250
    object.N_Null = 4
    NBREM = 25
    NRTANG = 145

    # USE OKRIMOVSKKY
    for J in range(6):
        object.AngularModel[J] = 2
    for J in range(object.N_Inelastic):
        object.KIN[J] = 2

    cdef int NELA, NMMT, NEPS, NVIB1, NVIB2, NVIB3, N_IonizationC, N_Ionization1, N_Ionization2, N_Ionization3, N_Ionization4, N_Ionization5, N_Ionization6, N_Ionization7, N_Ionization8, NKSH, N_Attachment1, N_Attachment2
    cdef int N_Attachment3, NTRP1, NTRP2, NTRP3, NTRP4, NUL1, NUL2, NUL3, NUL4,
    NELA = 159
    NMMT = 156
    NEPS = 156
    NVIB1 = 17
    NVIB2 = 18
    NVIB3 = 12
    N_IonizationC = 55
    N_Ionization1 = 31
    N_Ionization2 = 28
    N_Ionization3 = 28
    N_Ionization4 = 26
    N_Ionization5 = 25
    N_Ionization6 = 23
    N_Ionization7 = 21
    N_Ionization8 = 17
    NKSH = 81
    N_Attachment1 = 38
    N_Attachment2 = 30
    N_Attachment3 = 28
    NTRP1 = 11
    NTRP2 = 10
    NTRP3 = 10
    NTRP4 = 9
    NUL1 = 12
    NUL2 = 33
    NUL3 = 20
    NUL4 = 18

    # SCALING OF NULL COLLISIONS
    object.ScaleNull[0:4] = [1.0, 1.0, 1.0, 1.0]
    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, GPARA, GORTHO, DBA, DRAT, DBK, RSum, ENRT, AL
    cdef int L2,

    object.E = [0.0, 1.0, <float> (12.617), 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * ElectronMass / (<float> (18.01528) * AMU)

    object.IonizationEnergy[0:9] = [<float> (12.617), <float> (18.1), <float> (18.72), 21.0, 23.0, <float> (35.4), 45.0, 70.0,
                        532.0]

    # DBA IS DIPOLE MOMENT
    # DRAT IS RATIO OF MOMENTUM TRANSFER TO TOTAL X-SECTION FOR DIPOLE
    GPARA = 1.0
    GORTHO = 3.0
    DBA = <float> (0.728)
    DRAT = <float> (0.07)
    A0 = 0.5291772083e-08
    RY = <float> (13.60569172)
    DBK = <float> (8.37758) * RY * (DBA * A0) ** 2

    #CALCULATE POPULATION DENSITIES OF ROTATIONAL LEVELS
    for J in range(1, 100, 2):
        PJ[J - 1] = GPARA * (2.0 * AJL[J - 1] + 1.0) * exp(-1 * ELEV[J - 1] * 1e-3 / object.ThermalEnergy)
    for J in range(2, 101, 2):
        PJ[J - 1] = GORTHO * (2.0 * AJL[J - 1] + 1.0) * exp(-1 * ELEV[J - 1] * 1e-3 / object.ThermalEnergy)
    RSum = 0.0
    for J in range(100):
        RSum += PJ[J]
    for J in range(100):
        PJ[J] /= RSum

    for J in range(1, 106):
        object.EnergyLevels[(2 * J) - 2] = EROT[J - 1] * 1e-3
        object.EnergyLevels[(2 * J) - 1] = -1 * EROT[J - 1] * 1e-3
    object.EnergyLevels[210:250] = [<float> (-0.1977), <float> (0.1977), <float> (0.4535), <float> (0.919), <float> (7.04),
                           <float> (6.8425), <float> (7.2675), <float> (7.7725), <float> (8.3575), <float> (9.1),
                           <float> (8.91), <float> (9.43), <float> (9.95), <float> (10.47), <float> (9.95),
                           <float> (9.994), <float> (10.172), <float> (10.39), <float> (10.575), <float> (10.78),
                           <float> (11.01), <float> (11.122), <float> (11.377), <float> (11.525), <float> (11.75),
                           <float> (11.94), <float> (12.08), <float> (12.24), <float> (12.34), <float> (12.45),
                           <float> (13.0), <float> (13.117), <float> (14.117), <float> (15.117), <float> (16.117),
                           <float> (17.117), <float> (18.117), <float> (19.117), <float> (20.117), <float> (21.117)]

    for J in range(object.N_Ionization):
        object.EOBY[J] = object.IonizationEnergy[0] * <float> (0.93)

    for J in range(object.N_Ionization):
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
    object.WK[8] = <float> (0.0069)
    object.EFL[8] = 518
    object.NG1[8] = 1
    object.EG1[8] = 480
    object.NG2[8] = 2
    object.EG2[8] = 5.0

    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break

    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION
    cdef int NL = 0
    for NL in range(object.N_Inelastic):
        for i in range(4000):
            if object.EG[i] > abs(object.EnergyLevels[NL]):
                IOFFN[NL] = i
                break

    for I in range(object.N_Inelastic):
        object.PenningFraction[0][I] = 0.0

    for J in range(214, object.N_Inelastic):
        object.PenningFraction[0][J] = 0.0
        object.PenningFraction[1][J] = 1.0
        object.PenningFraction[2][J] = 1.0

    cdef double APOPV1, APOPGS, APOPSum, GAMMA1, GAMMA2, BETA, BETA2, EN, ElasticCrossSectionA, QMMT, PQ[3], EPS, QCOUNT, QGROSS, EPOINT
    cdef double F[32]
    F = [<float> (.003437), <float> (.017166), <float> (.019703), <float> (.005486), <float> (.006609),
         <float> (.030025), <float> (.030025), <float> (.006609), <float> (.005200), <float> (.014000),
         <float> (.010700), <float> (.009200), <float> (.006900), <float> (.021800), <float> (.023900),
         <float> (.013991), <float> (.009905), <float> (.023551), <float> (.007967), <float> (.018315),
         <float> (.011109), <float> (.008591), <float> (.028137), <float> (.119100), <float> (.097947),
         <float> (.039540), <float> (.042191), <float> (.059428), <float> (.052795), <float> (.024912),
         <float> (.010524), <float> (.002614), ]
    cdef int FI
    # CALC POPULATION OF LOW ENERGY VIBRATIONAL STATE
    APOPV1 = exp(object.EnergyLevels[210] / object.ThermalEnergy)
    APOPGS = 1.0
    APOPSum = APOPGS + APOPV1
    APOPV1 = APOPV1 / APOPSum

    #KEEP APOPGS=1 TO ALLOW FOR EXCITATIONS FROM UPPER STATE

    for I in range(4000):
        EN = object.EG[I]
        GAMMA1 = (ElectronMass2 + 2 * EN) / ElectronMass2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA

        # ELASTIC
        if EN <= XEL[0]:
            ElasticCrossSectionA = YEL[0] * 1e-16
        else:
            ElasticCrossSectionA = GasUtil.QLSCALE(EN, NELA, YEL, XEL)

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
        object.Q[1][I] = ElasticCrossSectionA
        object.PEElasticCrossSection[1][I] = PQ[2]

        # IONISATION CALCULATION
        for J in range(object.N_Ionization):
            object.PEIonizationCrossSection[J][I] = 0.0
            object.IonizationCrossSection[J][I] = 0.0
        #IF ENERGY LESS THAN 5KEV CALCULATE TOTAL COUNTING AND GROSS IONISATION
        if EN <= 5000:
            if EN > object.IonizationEnergy[0]:
                QCOUNT = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationC, YIONC, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
                QGROSS = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationC, YIONG, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
        else:
            QCOUNT = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationC, YIONC, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
            QGROSS = QCOUNT * 1.022

        #IONISATION TO H2O+
        if EN > XION1[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization1, YION1, XION1)
            if object.IonizationCrossSection[0][I] == 0:
                object.IonizationCrossSection[0][I] = QCOUNT * <float>(0.62996)

        #IONISATION TO OH+
        if EN > XION2[0]:
            object.IonizationCrossSection[1][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization2, YION2, XION2)
            if object.IonizationCrossSection[1][I] == 0:
                object.IonizationCrossSection[1][I] = QCOUNT *  <float>(0.19383)

        #IONISATION TO H+
        if EN > XION3[0]:
            object.IonizationCrossSection[2][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization3, YION3, XION3)
            if object.IonizationCrossSection[2][I] == 0:
                object.IonizationCrossSection[2][I] = QCOUNT *  <float>(0.13275)

        #IONISATION TO O+
        if EN > XION4[0]:
            object.IonizationCrossSection[3][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization4, YION4, XION4)
            if object.IonizationCrossSection[3][I] == 0:
                object.IonizationCrossSection[3][I] = QCOUNT *  <float>(0.02129)

        #IONISATION TO H2+
        if EN > XION5[0]:
            object.IonizationCrossSection[4][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization5, YION5, XION5)
            if object.IonizationCrossSection[4][I] == 0:
                object.IonizationCrossSection[4][I] = QCOUNT * <float>(0.00035)

        #IONISATION TO H+ + OH+
        if EN > XION6[0]:
            object.IonizationCrossSection[5][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization6, YION6, XION6)
            if object.IonizationCrossSection[5][I] == 0:
                object.IonizationCrossSection[5][I] = QCOUNT *  <float>(0.01395)

        #IONISATION TO H+ + O+
        if EN > XION7[0]:
            object.IonizationCrossSection[6][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization7, YION7, XION7)
            if object.IonizationCrossSection[6][I] == 0:
                object.IonizationCrossSection[6][I] = QCOUNT *  <float>(0.00705)

        #IONISATION TO O++
        if EN > XION8[0]:
            object.IonizationCrossSection[7][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization8, YION8, XION8)
            if object.IonizationCrossSection[7][I] == 0:
                object.IonizationCrossSection[7][I] = QCOUNT *  <float>(0.00085)

        #IONISATION TO OXYGEN K-SHELL
        if EN > XKSH[0]:
            object.IonizationCrossSection[8][I] = GasUtil.CALIonizationCrossSectionREG(EN, NKSH, YKSH, XKSH)

        for J in range(object.N_Ionization):
            if EN > 2 * object.IonizationEnergy[J]:
                object.PEIonizationCrossSection[J][I] = object.PEElasticCrossSection[1][I - IOFFION[J]]

        # ATTACHMENT H-
        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = 0.0
        if EN > XATT1[0] and EN < XATT1[N_Attachment1 - 1]:
            object.AttachmentCrossSection[0][I] = GasUtil.QLSCALE(EN, N_Attachment1, YATT1, XATT1) * 1e-5

        # ATTACHMENT O-
        object.AttachmentCrossSection[1][I] = 0.0
        if EN > XATT2[0] and EN < XATT2[N_Attachment2 - 1]:
            object.AttachmentCrossSection[1][I] = GasUtil.QLSCALE(EN, N_Attachment2, YATT2, XATT2) * 1e-5

        #ATTACHMENT OH-
        object.AttachmentCrossSection[2][I] = 0.0
        if EN > XATT3[0] and EN < XATT3[N_Attachment3 - 1]:
            object.AttachmentCrossSection[2][I] = GasUtil.QLSCALE(EN, N_Attachment3, YATT3, XATT3) * 1e-5

        object.Q[3][I] = 0.0
        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0
        # ZERO INELASTIC ARRAYS
        for J in range(object.N_Inelastic):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.0

        # DIPOLE BORN ROTATIONAL STATES
        ENRT = sqrt(EN)

        # SUPER ELASTIC ROTATIONAL COLLISIONS
        for J in range(2, 211, 2):
            AL = AJIN[J - 1]
            L2 = J / 2
            object.InelasticCrossSectionPerGas[J - 1][I] = DBK * SALPHA[L2 - 1] * PJ[IMAP[J - 1] - 1] * log(
                (ENRT + sqrt(EN - object.EnergyLevels[J - 1])) / (sqrt(EN - object.EnergyLevels[J - 1]) - ENRT)) / (
                                           (2.0 * AL + 1.0) * EN) * AMPROT

            if EN > 2000:
                object.InelasticCrossSectionPerGas[J - 1][I] = 0.0
                continue

            EPOINT = EN / abs(object.EnergyLevels[J - 1])
            #TODO: PRINT ERROR STATEMENT

            object.PEInelasticCrossSectionPerGas[J - 1][I] = 1.0 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENRTS)
            if EPOINT<=ENRTS[0]:
                    object.PEInelasticCrossSectionPerGas[J-1][I] = 1 - (YEPSR[0]/ENRTS[0])*EPOINT
            XSECDUM[J - 1] = GasUtil.CALPQ3(EPOINT, NRTANG, YMTRT, ENRTS) * object.InelasticCrossSectionPerGas[J - 1][I]

        # ROTATIONAL COLLISIONS
        for J in range(1, 210, 2):
            object.InelasticCrossSectionPerGas[J - 1][I] = 0.0
            if EN > object.EnergyLevels[J - 1]:
                AL = AJIN[J - 1]
                L2 = (J + 1) / 2
                object.InelasticCrossSectionPerGas[J - 1][I] = DBK * SALPHA[L2 - 1] * PJ[IMAP[J - 1] - 1] * log(
                    (ENRT + sqrt(EN - object.EnergyLevels[J - 1])) / (ENRT - sqrt(EN - object.EnergyLevels[J - 1]))) / (
                                               (2.0 * AL + 1.0) * EN) * AMPROT
                if EN > 2000:
                    object.InelasticCrossSectionPerGas[J - 1][I] = 0.0
                    continue

                EPOINT = EN / abs(object.EnergyLevels[J - 1])
                #TODO: PRINT ERROR STATEMENT

                object.PEInelasticCrossSectionPerGas[J - 1][I] = 1.0 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                if EPOINT<=ENROT[0]:
                    object.PEInelasticCrossSectionPerGas[J-1][I] = 1 - (YEPSR[0]/ENROT[0])*EPOINT
                XSECDUM[J - 1] = GasUtil.CALPQ3(EPOINT, NRTANG, YMTRT, ENRTS) * object.InelasticCrossSectionPerGas[J - 1][I]

        # VIBRATION BEND V2 SUPERELASTIC (DIPOLE 1/E FALL OFF ABOVE ENERGY OF
        # XVIB1(NVIB1) EV )
        object.InelasticCrossSectionPerGas[210][I] = 0.0
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[210][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN + object.EnergyLevels[211], NVIB1, YVIB1, XVIB1, 1) * APOPV1 * 100 / EN
            if EN + object.EnergyLevels[211] <= XVIB1[NVIB1 - 1]:
                object.InelasticCrossSectionPerGas[210][I] *= (EN + object.EnergyLevels[211])
            object.PEInelasticCrossSectionPerGas[210][I] = object.PEElasticCrossSection[1][I - IOFFN[210]]

        # VIBRATION BEND V2  (DIPOLE 1/E FALL OFF ABOVE ENERGY OF
        # XVIB1(NVIB1) EV )
        object.InelasticCrossSectionPerGas[211][I] = 0.0
        if EN > object.EnergyLevels[211]:
            object.InelasticCrossSectionPerGas[211][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB1, YVIB1, XVIB1, 1) * APOPGS * 100
            object.PEInelasticCrossSectionPerGas[211][I] = object.PEElasticCrossSection[1][I - IOFFN[211]]
            # CALCULATE DIPOLE ANGULAR DISTRIBUTION FACTOR FOR TRANSITION
            EPOINT = EN / abs(object.EnergyLevels[211])
            if EPOINT <= 500:
                object.PEInelasticCrossSectionPerGas[211][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
            else:
                object.PEInelasticCrossSectionPerGas[211][I] = object.PEElasticCrossSection[1][I - IOFFN[211]]

        # VIBRATION STRETCH V1+V3
        object.InelasticCrossSectionPerGas[212][I] = 0.0
        if EN > object.EnergyLevels[212]:
            object.InelasticCrossSectionPerGas[212][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB2, YVIB2, XVIB2, 1.5) * 100
            if EN < 1.5:
                object.PEInelasticCrossSectionPerGas[212][I] = 0.0
            else:
                EPOINT = EN / abs(object.EnergyLevels[212])
                if EPOINT <= 500:
                    object.PEInelasticCrossSectionPerGas[212][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEInelasticCrossSectionPerGas[212][I] = object.PEElasticCrossSection[1][I - IOFFN[212]]

        # VIBRATION HARMONICS NV2+ NV1+NV3
        object.InelasticCrossSectionPerGas[213][I] = 0.0
        if EN > object.EnergyLevels[213]:
            object.InelasticCrossSectionPerGas[213][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB3, YVIB3, XVIB3, 1.5) * 100
            EPOINT = EN / abs(object.EnergyLevels[213])
            if EPOINT <= 500:
                object.PEInelasticCrossSectionPerGas[213][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
            else:
                object.PEInelasticCrossSectionPerGas[213][I] = object.PEElasticCrossSection[1][I - IOFFN[213]]

        # TRIPLET 3B1
        if EN > object.EnergyLevels[214]:
            if EN <= XTRP1[NTRP1 - 1]:
                object.InelasticCrossSectionPerGas[214][I] = GasUtil.QLSCALE(EN, NTRP1, YTRP1, XTRP1)
            else:
                object.InelasticCrossSectionPerGas[214][I] = YTRP1[NTRP1 - 1] * (XTRP1[NTRP1 - 1] / EN) ** 1.5 * 1e-16

            if EN > 2.0 * object.EnergyLevels[214]:
                object.PEInelasticCrossSectionPerGas[214][I] = object.PEElasticCrossSection[1][I - IOFFN[214]]

        FI = 0
        # EXCITATION  1B1 (7.48EV LEVEL SPLIT INTO 4 GROUPS)
        for J in range(215, 219):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                EPOINT = EN / abs(object.EnergyLevels[J])
                if EPOINT <= 500:
                    object.PEInelasticCrossSectionPerGas[J][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        # TRIPLET 3A2 + 1A2 + 3A1 9.1EV
        if EN > object.EnergyLevels[219]:
            if EN <= XTRP2[NTRP2 - 1]:
                object.InelasticCrossSectionPerGas[219][I] = GasUtil.QLSCALE(EN, NTRP2, YTRP2, XTRP2)
            else:
                object.InelasticCrossSectionPerGas[219][I] = YTRP2[NTRP2 - 1] * (XTRP2[NTRP2 - 1] / EN) ** 1.5 * 1e-16

            if EN > 2.0 * object.EnergyLevels[219]:
                object.PEInelasticCrossSectionPerGas[219][I] = object.PEElasticCrossSection[1][I - IOFFN[219]]

        # EXCITATION  1A1 (9.69EV LEVEL SPLIT INTO 4 GROUPS)
        for J in range(220, 224):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                EPOINT = EN / abs(object.EnergyLevels[J])
                if EPOINT <= 500:
                    object.PEInelasticCrossSectionPerGas[J][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        # TRIPLET 3B1 9.95EV
        if EN > object.EnergyLevels[224]:
            if EN <= XTRP3[NTRP3 - 1]:
                object.InelasticCrossSectionPerGas[224][I] = GasUtil.QLSCALE(EN, NTRP3, YTRP3, XTRP3)
            else:
                object.InelasticCrossSectionPerGas[224][I] = YTRP3[NTRP3 - 1] * (XTRP3[NTRP3 - 1] / EN) ** 1.5 * 1e-16

            if EN > 2.0 * object.EnergyLevels[224]:
                object.PEInelasticCrossSectionPerGas[224][I] = object.PEElasticCrossSection[1][I - IOFFN[224]]

        # EXCITATION  1B1 (3pa1) 9.994 EV
        for J in range(225, 240):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                EPOINT = EN / abs(object.EnergyLevels[J])
                if EPOINT <= 500:
                    object.PEInelasticCrossSectionPerGas[J][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        # TRIPLET Sum OF HIGHER TRIPLETS   13.0 EV
        if EN > object.EnergyLevels[240]:
            if EN <= XTRP4[NTRP4 - 1]:
                object.InelasticCrossSectionPerGas[240][I] = GasUtil.QLSCALE(EN, NTRP4, YTRP4, XTRP4)
            else:
                object.InelasticCrossSectionPerGas[240][I] = YTRP4[NTRP4 - 1] * (XTRP4[NTRP4 - 1] / EN) ** 1.5 * 1e-16

            if EN > 2.0 * object.EnergyLevels[240]:
                object.PEInelasticCrossSectionPerGas[240][I] = object.PEElasticCrossSection[1][I - IOFFN[240]]

        # EXCITATION  1B1 (3pa1) 9.994 EV
        for J in range(241, 250):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                EPOINT = EN / abs(object.EnergyLevels[J])
                if EPOINT <= 500:
                    object.PEInelasticCrossSectionPerGas[J][I] = 1 - GasUtil.CALPQ3(EPOINT, NRTANG, YEPSR, ENROT)
                else:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        # LOAD NULL COLLISIONS
        #  OH PRODUCTION FROM DISSOCIATION HARB ET AL J.CHEM.PHYS. 115(2001)5507
        # SCALED ABOVE 200EV BY 1/ENERGY
        object.NullCrossSection[0][I] = 0.0
        if EN > XNUL1[0]:
            object.NullCrossSection[0][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NUL1, YNUL1, XNUL1, 1) * 100 * object.ScaleNull[0]

        # LIGHT EMISSION FROM OH(A2-X) MOHLMMoleculesPerCm3PerGas AND DEHEER CHEM.PHYS.19(1979)233
        object.NullCrossSection[1][I] = 0.0
        if EN > XNUL2[0]:
            object.NullCrossSection[1][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NUL2, YNUL2, XNUL2, 1) * 100 * object.ScaleNull[1]

        # LIGHT EMISSION FROM H(3-2) , MOHLMMoleculesPerCm3PerGas AND DEHEER CHEM.PHYS.19(1979)233
        object.NullCrossSection[2][I] = 0.0
        if EN > XNUL3[0]:
            object.NullCrossSection[2][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NUL3, YNUL3, XNUL3, 1) * 100 * object.ScaleNull[2]

        # LIGHT EMISSION FROM H(2-1) , MOHLMMoleculesPerCm3PerGas AND DEHEER CHEM.PHYS.19(1979)233
        object.NullCrossSection[3][I] = 0.0
        if EN > XNUL4[0]:
            object.NullCrossSection[3][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NUL4, YNUL4, XNUL4, 1) * 100 * object.ScaleNull[3]

    for J in range(object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break
    return
