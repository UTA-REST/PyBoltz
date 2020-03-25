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
cdef void Gas9(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Ethane gas.
    """
    gd = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"gases.npy")).item()
    cdef double XEN[164], YMT[164], YEL[164], YEPS[164], XATT1[11], YATT1[11], XATT2[9], YATT2[9], XVIB1[29], YVIB1[29], XVIB2[28]
    cdef double YVIB2[28], XVIB3[28], YVIB3[28], XVIB4[46], YVIB4[46], XVIB5[16], YVIB5[16], XTR1[12], YTR1[12], XTR2[11], YTR2[11]
    cdef double XTR3[11], YTR3[11], XNUL1[25], YNUL1[25], XNUL2[13], YNUL2[13], XNUL3[14], YNUL3[14]
    cdef double XION1[31], YION1[31], XION2[31], YION2[31], XION3[31], YION3[31], XION4[30], YION4[30], XION5[29], YION5[29], XION6[29]
    cdef double YION6[29], XION7[26], YION7[26], XION8[26], YION8[26], XION9[25], YION9[25], XION10[24], YION10[24], XION11[24]
    cdef double YION12[24], XION13[23], YION13[23], XION14[21], YION14[21], YION11[24], XION12[24]
    cdef double XION15[21], YION15[21], XION16[83], YION16[83], XION[50], YIONG[50], YIONC[50], Z1T[25], Z6T[25], EBRM[25]
    cdef int IOFFION[16], IOFFN[250]

    XEN = gd['gas9/XEN']
    YMT = gd['gas9/YMT']
    YEL = gd['gas9/YEL']
    YEPS = gd['gas9/YEPS']
    XATT1 = gd['gas9/XATT1']
    YATT1 = gd['gas9/YATT1']
    XATT2 = gd['gas9/XATT2']
    YATT2 = gd['gas9/YATT2']
    XVIB1 = gd['gas9/XVIB1']
    YVIB1 = gd['gas9/YVIB1']
    XVIB2 = gd['gas9/XVIB2']
    YVIB2 = gd['gas9/YVIB2']
    XVIB3 = gd['gas9/XVIB3']
    YVIB3 = gd['gas9/YVIB3']
    XVIB4 = gd['gas9/XVIB4']
    YVIB4 = gd['gas9/YVIB4']
    XVIB5 = gd['gas9/XVIB5']
    YVIB5 = gd['gas9/YVIB5']
    XTR1 = gd['gas9/XTR1']
    YTR1 = gd['gas9/YTR1']
    XTR2 = gd['gas9/XTR2']
    YTR2 = gd['gas9/YTR2']
    XTR3 = gd['gas9/XTR3']
    YTR3 = gd['gas9/YTR3']
    XNUL1 = gd['gas9/XNUL1']
    YNUL1 = gd['gas9/YNUL1']
    XNUL2 = gd['gas9/XNUL2']
    YNUL2 = gd['gas9/YNUL2']
    XNUL3 = gd['gas9/XNUL3']
    YNUL3 = gd['gas9/YNUL3']
    XION1 = gd['gas9/XION1']
    YION1 = gd['gas9/YION1']
    XION2 = gd['gas9/XION2']
    YION2 = gd['gas9/YION2']
    XION3 = gd['gas9/XION3']
    YION3 = gd['gas9/YION3']
    XION4 = gd['gas9/XION4']
    YION4 = gd['gas9/YION4']
    XION5 = gd['gas9/XION5']
    YION5 = gd['gas9/YION5']
    XION6 = gd['gas9/XION6']
    YION6 = gd['gas9/YION6']
    XION7 = gd['gas9/XION7']
    YION7 = gd['gas9/YION7']
    XION8 = gd['gas9/XION8']
    YION8 = gd['gas9/YION8']
    XION9 = gd['gas9/XION9']
    YION9 = gd['gas9/YION9']
    XION10 = gd['gas9/XION10']
    YION10 = gd['gas9/YION10']
    XION11 = gd['gas9/XION11']
    YION11 = gd['gas9/YION11']
    XION12 = gd['gas9/XION12']
    YION12 = gd['gas9/YION12']
    XION13 = gd['gas9/XION13']
    YION13 = gd['gas9/YION13']
    XION14 = gd['gas9/XION14']
    YION14 = gd['gas9/YION14']
    XION15 = gd['gas9/XION15']
    YION15 = gd['gas9/YION15']
    XION16 = gd['gas9/XION16']
    YION16 = gd['gas9/YION16']
    XION = gd['gas9/XION']
    YIONG = gd['gas9/YIONG']
    YIONC = gd['gas9/YIONC']
    Z1T = gd['gas9/Z1T']
    Z6T = gd['gas9/Z6T']
    EBRM = gd['gas9/EBRM']

    cdef double A0, RY, CONST, ElectronMass2, API, BBCONST, AM2, C,
    cdef int NBREM, NASIZE, NDATA, N_IonizationD, N_Ionization1, N_Ionization2, N_Ionization3, N_Ionization4, N_Ionization5, N_Ionization6, N_Ionization7, N_Ionization8, N_Ionization9, N_Ionization10, N_Ionization11, N_Ionization12, N_Ionization13
    cdef int N_Ionization14, N_Ionization15, N_Ionization16, N_Attachment1, N_Attachment2, NVIB1, NVIB2, NVIB3, NVIB4, NVIB5, NTR1, NTR2, NTR3, NUL1, NUL2, NUL3, i, j, I, J, NL

    # BORN-BETHE CONSTANTS
    A0 = 0.52917720859e-08
    RY = <float> (13.60569193)
    CONST = 1.8738843 - 20
    ElectronMass2 = <float> (1021997.804)
    API = acos(-1.0)
    BBCONST = 16.0 * API * A0 * A0 * RY * RY / ElectronMass2

    # BORN BETHE FOR IONISATION
    AM2 = <float> (7.21)
    C = <float> (70.5)
    # ARRAY SIZE
    NASIZE = 4000
    object.N_Ionization = 16
    object.N_Attachment = 2
    object.N_Inelastic = 55
    object.N_Null = 3
    NBREM = 25

    for i in range(6):
        object.AngularModel[i] = object.WhichAngularModel
    for i in range(10):
        object.KIN[i] = 0
    for i in range(10, object.N_Inelastic):
        object.KIN[i] = 2
    NDATA = 164

    N_IonizationD = 50
    N_Ionization1 = 31
    N_Ionization2 = 31
    N_Ionization3 = 31
    N_Ionization4 = 30
    N_Ionization5 = 29
    N_Ionization6 = 29
    N_Ionization7 = 26
    N_Ionization8 = 26
    N_Ionization9 = 25
    N_Ionization10 = 24
    N_Ionization11 = 24
    N_Ionization12 = 24
    N_Ionization13 = 23
    N_Ionization14 = 21
    N_Ionization15 = 21
    N_Ionization16 = 83
    N_Attachment1 = 11
    N_Attachment2 = 9
    NVIB1 = 29
    NVIB2 = 28
    NVIB3 = 28
    NVIB4 = 46
    NVIB5 = 16
    NTR1 = 12
    NTR2 = 11
    NTR3 = 11
    NUL1 = 25
    NUL2 = 13
    NUL3 = 14

    object.ScaleNull[0:3] = [1.0, 10, 10]
    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, EOBY[16], SCLOBY, APOP1, APOP2, APOP3, APOP4, QCOUNT = 0.0

    object.E = [0.0, 1.0, <float> (11.52), 0.0, 0.0, 0.0]
    object.E[1] = <float>(2.0) * ElectronMass / (<float> (30.06964) * AMU)

    object.IonizationEnergy[0:16] = [<float> (11.52), <float> (12.05), <float> (12.65), <float> (13.65), <float> (14.8),
                         <float> (14.8), <float> (20.5), <float> (21.5), <float> (25.8), <float> (26.2), <float> (32.0),
                         <float> (32.5), <float> (36.0), <float> (37.0), <float> (37.0), <float> (285.0)]

    # OPAL BEATY
    SCLOBY = 1.0

    for J in range(object.N_Ionization):
        object.EOBY[J] = object.IonizationEnergy[J] * SCLOBY
    object.EOBY[object.N_Ionization - 1] = object.IonizationEnergy[object.N_Ionization - 1] * <float> (0.63)

    for J in range(15):
        object.NC0[J] = 0
        object.EC0[J] = 0.0
        object.WK[J] = 0.0
        object.EFL[J] = 0.0
        object.NG1[J] = 0
        object.EG1[J] = 0.0
        object.NG2[J] = 0
        object.EG2[J] = 0.0
    #DOUBLE CHARGE , ++ ION STATES ( EXTRA ELECTRON )
    object.NC0[10] = 1
    object.EC0[10] = 6.0
    #FLUORESCENCE DATA  (KSHELL)
    object.NC0[15] = 2
    object.EC0[15] = 253
    object.WK[15] = <float>(0.0026)
    object.EFL[15] = 273
    object.NG1[15] = 1
    object.EG1[15] = 253
    object.NG2[15] = 2
    object.EG2[15] = 5.0

    #OFFSET ENERGY FOR IONISATION ELECTRON ANGULAR DISTRIBUTION
    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break

    object.EnergyLevels = gd['gas9/EnergyLevels']
    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION

    for NL in range(object.N_Inelastic):
        for i in range(4000):
            if object.EG[i] > object.EnergyLevels[NL]:
                IOFFN[NL] = i
                break

    for i in range(object.N_Inelastic):
        for j in range(3):
            object.PenningFraction[j][i]=0.0

    # CALC LEVEL POPULATIONS
    APOP1 = exp(object.EnergyLevels[0] / object.ThermalEnergy)
    APOP2 = exp(object.EnergyLevels[2] / object.ThermalEnergy)
    APOP3 = exp(object.EnergyLevels[4] / object.ThermalEnergy)
    APOP4 = exp(object.EnergyLevels[6] / object.ThermalEnergy)
    for J in range(NBREM):
        EBRM[J] = exp(EBRM[J])
    cdef double EN, ENLG, GAMMA1, GAMMA2, BETA, BETA2, QMT, ElasticCrossSection, PQ[3], X1, X2, QBB = 0.0, CrossSectionSum, EFAC, F[42]
    cdef int FI
    F = [<float> (0.000136), <float> (0.001744), <float> (0.008187), <float> (0.006312), <float> (0.011877),
         <float> (0.020856), <float> (0.031444), <float> (0.039549), <float> (0.042350), <float> (0.041113),
         <float> (0.038256), <float> (0.036556), <float> (0.096232), <float> (.083738), <float> (.043456),
         <float> (.047436), <float> (.047800), <float> (.048914), <float> (.054353), <float> (.061019),
         <float> (.244430), <float> (.284790), <float> (.095973), <float> (.090728), <float> (0.071357),
         <float> (.074875), <float> (.054542), <float> (.022479), <float> (.008585), <float> (.004524),
         <float> (.004982), <float> (.010130), <float> (.013320), <float> (.013310), <float> (.010760),
         <float> (.009797), <float> (.009198), <float> (.008312), <float> (.007139), <float> (.004715),
         <float> (.002137), <float> (.000662), ]

    object.EnergySteps = 4000
    for I in range(object.EnergySteps):
        EN = object.EG[I]
        ENLG = log(EN)
        GAMMA1 = (ElectronMass2 + 2 * EN) / ElectronMass2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA

        if EN <= 10:
            QMT = GasUtil.CALPQ3(EN, NDATA, YMT, XEN) * 1e-16
            ElasticCrossSection = GasUtil.CALPQ3(EN, NDATA, YEL, XEN) * 1e-16
            PQ[2] = GasUtil.CALPQ3(EN, NDATA, YEPS, XEN)
        else:
            ElasticCrossSection = GasUtil.QLSCALE(EN, NDATA, YEL, XEN)
            QMT = GasUtil.QLSCALE(EN, NDATA, YMT, XEN)
            PQ[2] = GasUtil.QLSCALE(EN, NDATA, YEPS, XEN) * 1e16
        PQ[2] = 1 - PQ[2]
        PQ[1] = 0.5 + (ElasticCrossSection - QMT) / ElasticCrossSection
        PQ[0] = 0.5
        object.PEElasticCrossSection[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = ElasticCrossSection

        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMT

        # IONISATION

        for i in range(object.N_Ionization):
            object.IonizationCrossSection[i][I] = 0.0
            object.PEIonizationCrossSection[i][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEIonizationCrossSection[i][I] = 0.0

        # C2H6+
        if EN > object.IonizationEnergy[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization1, YION1, XION1)
            if object.IonizationCrossSection[0][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    QCOUNT = GasUtil.QLSCALE(EN, N_IonizationD, YIONC, XION)
                    # fraction of QCOUNT
                    object.IonizationCrossSection[0][I] = QCOUNT * <float>(0.1378)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    X2 = 1 / BETA2
                    X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                    QBB = CONST * (AM2 * (X1 - object.DEN[I] / 2) + C * X2)
                    object.IonizationCrossSection[0][I] = QBB * <float>(0.1378)
            if EN >= 2 * object.IonizationEnergy[0]:
                object.PEIonizationCrossSection[0][I] = object.PEElasticCrossSection[1][I - IOFFION[0]]

        # C2H4+
        if EN > object.IonizationEnergy[1]:
            object.IonizationCrossSection[1][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization2, YION2, XION2)
            if object.IonizationCrossSection[1][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    # fraction of QCOUNT
                    object.IonizationCrossSection[1][I] = QCOUNT * <float>(0.4481)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[1][I] = QBB * <float>(0.4481)
            if EN >= 2 * object.IonizationEnergy[1]:
                object.PEIonizationCrossSection[1][I] = object.PEElasticCrossSection[1][I - IOFFION[1]]

        # C2H5+
        if EN > object.IonizationEnergy[2]:
            object.IonizationCrossSection[2][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization3, YION3, XION3)
            if object.IonizationCrossSection[2][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[2][I] = QCOUNT * <float>(0.1104)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[2][I] = QBB * <float>(0.1104)
            if EN >= 2 * object.IonizationEnergy[2]:
                object.PEIonizationCrossSection[2][I] = object.PEElasticCrossSection[1][I - IOFFION[2]]

        # CH3+
        if EN > object.IonizationEnergy[3]:
            object.IonizationCrossSection[3][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization4, YION4, XION4)
            if object.IonizationCrossSection[3][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[3][I] = QCOUNT * <float>(0.01718)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[3][I] = QBB * <float>(0.01718)
            if EN >= 2 * object.IonizationEnergy[3]:
                object.PEIonizationCrossSection[3][I] = object.PEElasticCrossSection[1][I - IOFFION[3]]

        # C2H3+
        if EN > object.IonizationEnergy[4]:
            object.IonizationCrossSection[4][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization5, YION5, XION5)
            if object.IonizationCrossSection[4][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[4][I] = QCOUNT * <float>(0.1283)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[4][I] = QBB * <float>(0.1283)
            if EN >= 2 * object.IonizationEnergy[4]:
                object.PEIonizationCrossSection[4][I] = object.PEElasticCrossSection[1][I - IOFFION[4]]

        # C2H2+
        if EN > object.IonizationEnergy[5]:
            object.IonizationCrossSection[5][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization6, YION6, XION6)
            if object.IonizationCrossSection[5][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[5][I] = QCOUNT * <float>(0.07)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[5][I] = QBB * <float>(0.07)
            if EN >= 2 * object.IonizationEnergy[5]:
                object.PEIonizationCrossSection[5][I] = object.PEElasticCrossSection[1][I - IOFFION[5]]

        # H+
        if EN > object.IonizationEnergy[6]:
            object.IonizationCrossSection[6][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization7, YION7, XION7)
            if object.IonizationCrossSection[6][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[6][I] = QCOUNT * <float>(0.000011)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[6][I] = QBB * <float>(0.000011)
            if EN >= 2 * object.IonizationEnergy[6]:
                object.PEIonizationCrossSection[6][I] = object.PEElasticCrossSection[1][I - IOFFION[6]]

        # H2+
        if EN > object.IonizationEnergy[7]:
            object.IonizationCrossSection[7][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization8, YION8, XION8)
            if object.IonizationCrossSection[7][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[7][I] = QCOUNT * <float>(0.00036)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[7][I] = QBB * <float>(0.00036)
            if EN >= 2 * object.IonizationEnergy[7]:
                object.PEIonizationCrossSection[7][I] = object.PEElasticCrossSection[1][I - IOFFION[7]]

        # CH2+
        if EN > object.IonizationEnergy[8]:
            object.IonizationCrossSection[8][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization9, YION9, XION9)
            if object.IonizationCrossSection[8][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[8][I] = QCOUNT * <float>(0.0066)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[8][I] = QBB * <float>(0.0066)
            if EN >= 2 * object.IonizationEnergy[8]:
                object.PEIonizationCrossSection[8][I] = object.PEElasticCrossSection[1][I - IOFFION[8]]

        # C2H+
        if EN > object.IonizationEnergy[9]:
            object.IonizationCrossSection[9][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization10, YION10, XION10)
            if object.IonizationCrossSection[9][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[9][I] = QCOUNT * <float>(0.0062)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[9][I] = QBB * <float>(0.0062)
            if EN >= 2 * object.IonizationEnergy[9]:
                object.PEIonizationCrossSection[9][I] = object.PEElasticCrossSection[1][I - IOFFION[9]]

        # C2H6++
        if EN > object.IonizationEnergy[10]:
            object.IonizationCrossSection[10][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization11, YION11, XION11)
            if object.IonizationCrossSection[10][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[10][I] = QCOUNT * <float>(0.0745)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[10][I] = QBB * <float>(0.0745)
            if EN >= 2 * object.IonizationEnergy[10]:
                object.PEIonizationCrossSection[10][I] = object.PEElasticCrossSection[1][I - IOFFION[10]]

        # H3+
        if EN > object.IonizationEnergy[11]:
            object.IonizationCrossSection[11][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization12, YION12, XION12)
            if object.IonizationCrossSection[11][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[11][I] = QCOUNT * <float>(0.0000055)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[11][I] = QBB * <float>(0.0000055)
            if EN >= 2 * object.IonizationEnergy[11]:
                object.PEIonizationCrossSection[11][I] = object.PEElasticCrossSection[1][I - IOFFION[11]]

        # CH+
        if EN > object.IonizationEnergy[12]:
            object.IonizationCrossSection[12][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization13, YION13, XION13)
            if object.IonizationCrossSection[12][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[12][I] = QCOUNT * <float>(0.00037)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[12][I] = QBB * <float>(0.00037)
            if EN >= 2 * object.IonizationEnergy[12]:
                object.PEIonizationCrossSection[12][I] = object.PEElasticCrossSection[1][I - IOFFION[12]]

        # C2+
        if EN > object.IonizationEnergy[13]:
            object.IonizationCrossSection[13][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization14, YION14, XION14)
            if object.IonizationCrossSection[13][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[13][I] = QCOUNT * <float>(0.000022)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[13][I] = QBB * <float>(0.000022)
            if EN >= 2 * object.IonizationEnergy[13]:
                object.PEIonizationCrossSection[13][I] = object.PEElasticCrossSection[1][I - IOFFION[13]]

        # C+
        if EN > object.IonizationEnergy[14]:
            object.IonizationCrossSection[14][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization15, YION15, XION15)
            if object.IonizationCrossSection[14][I] == 0:
                if EN <= XION[N_IonizationD - 1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.IonizationCrossSection[14][I] = QCOUNT * <float>(0.00011)
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.IonizationCrossSection[14][I] = QBB * <float>(0.00011)
            if EN >= 2 * object.IonizationEnergy[14]:
                object.PEIonizationCrossSection[14][I] = object.PEElasticCrossSection[1][I - IOFFION[14]]

        #CARBON K-SHELL
        if EN > object.IonizationEnergy[15]:
            object.IonizationCrossSection[15][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_Ionization16, YION16, XION16) * 2
            if EN >= 2 * object.IonizationEnergy[15]:
                object.PEIonizationCrossSection[15][I] = object.PEElasticCrossSection[1][I - IOFFION[15]]

        CrossSectionSum = 0.0
        for J in range(15):
            CrossSectionSum += object.IonizationCrossSection[J][I]
        if CrossSectionSum != 0.0:
            for J in range(15):
                object.IonizationCrossSection[J][I] *= (CrossSectionSum - object.IonizationCrossSection[15][I]) / CrossSectionSum

        object.Q[3][I] = 0.0

        object.AttachmentCrossSection[0][I] = 0.0
        # ATTACHMENT to H-
        if EN >= XATT1[0]:
            object.AttachmentCrossSection[0][I] = GasUtil.CALIonizationCrossSection(EN, N_Attachment1, YATT1, XATT1)

        object.AttachmentCrossSection[1][I] = 0.0
        # ATTACHMENT to CH2-
        if EN >= XATT2[0]:
            object.AttachmentCrossSection[1][I] = GasUtil.CALIonizationCrossSection(EN, N_Attachment2, YATT2, XATT2)

        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        # set ZEROS

        for J in range(object.N_Inelastic):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.0

        # SUPERELASTIC VIBRATION-TORSION         AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > 0.0:
            EFAC = sqrt(1.0 - (object.EnergyLevels[0] / EN))
            object.InelasticCrossSectionPerGas[0][I] = <float>(0.0045) * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.InelasticCrossSectionPerGas[0][I] *= APOP1 / (1.0 + APOP1) * 1.e-16
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[0][I] = object.PEElasticCrossSection[1][I - IOFFN[0]]

        #VIBRATION-TORSION                      AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > object.EnergyLevels[1]:
            EFAC = sqrt(1.0 - (object.EnergyLevels[1] / EN))
            object.InelasticCrossSectionPerGas[1][I] = <float>(0.0045) * log((EFAC + 1.0) / (1.0 - EFAC)) / EN
            object.InelasticCrossSectionPerGas[1][I] *= 1.0 / (1.0 + APOP1) * 1.e-16
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[1][I] = object.PEElasticCrossSection[1][I - IOFFN[1]]

        #SUPERELASTIC VIB1                     AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[2][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB1, YVIB1, XVIB1, APOP2 / (1 + APOP2), object.EnergyLevels[3], 1,
                                                  -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[2][I] = object.PEElasticCrossSection[1][I - IOFFN[2]]

        #VIB1                           AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > object.EnergyLevels[3]:
            object.InelasticCrossSectionPerGas[3][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB1, YVIB1, XVIB1, 1 / (1 + APOP2), 0, 1, -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[3][I] = object.PEElasticCrossSection[1][I - IOFFN[3]]

        #SUPERELASTIC VIB2                     AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[4][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB2, YVIB2, XVIB2, APOP3 / (1 + APOP3), object.EnergyLevels[5], 1,
                                                  -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[4][I] = object.PEElasticCrossSection[1][I - IOFFN[4]]

        #VIB2                           AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > object.EnergyLevels[5]:
            object.InelasticCrossSectionPerGas[5][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB2, YVIB2, XVIB2, 1 / (1 + APOP3), 0, 1, -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[5][I] = object.PEElasticCrossSection[1][I - IOFFN[5]]

        #SUPERELASTIC VIB3                     AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[6][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB3, YVIB3, XVIB3, APOP4 / (1 + APOP4), object.EnergyLevels[7], 1,
                                                  -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[6][I] = object.PEElasticCrossSection[1][I - IOFFN[6]]

        #VIB3                           AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > object.EnergyLevels[7]:
            object.InelasticCrossSectionPerGas[7][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB3, YVIB3, XVIB3, 1 / (1 + APOP4), 0, 1, -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[7][I] = object.PEElasticCrossSection[1][I - IOFFN[7]]

        #VIB4                           AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > object.EnergyLevels[8]:
            object.InelasticCrossSectionPerGas[8][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB4, YVIB4, XVIB4, 1, 0, 1, -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[8][I] = object.PEElasticCrossSection[1][I - IOFFN[8]]

        #VIB HARMONICS                  AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > object.EnergyLevels[9]:
            object.InelasticCrossSectionPerGas[9][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB5, YVIB5, XVIB5, 1, 0, 1, -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[9][I] = object.PEElasticCrossSection[1][I - IOFFN[9]]

        # EXCITATION TO TRIPLET AND SINGLET LEVELS
        #
        # FIRST TRIPLET AT  6.85 EV

        if EN > object.EnergyLevels[10]:
            object.InelasticCrossSectionPerGas[10][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTR1, YTR1, XTR1, 2) * 100
            if EN > 3 * object.EnergyLevels[10]:
                object.PEInelasticCrossSectionPerGas[10][I] = object.PEElasticCrossSection[1][I - IOFFN[10]]

        #SINGLET DISSOCIATION AT  7.93  EV     BEF SCALING F[FI]
        FI = 0
        J = 11

        if EN > object.EnergyLevels[J]:
            object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
            if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                object.InelasticCrossSectionPerGas[J][I] = 0.0
            if EN > 3 * object.EnergyLevels[J]:
                object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
        FI += 1

        # SECOND TRIPLET AT  8.00 EV
        if EN > object.EnergyLevels[12]:
            object.InelasticCrossSectionPerGas[12][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTR2, YTR2, XTR2, 2) * 100
            if EN > 3 * object.EnergyLevels[12]:
                object.PEInelasticCrossSectionPerGas[12][I] = object.PEElasticCrossSection[1][I - IOFFN[12]]

        for J in range(13, 24):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 3 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        # THIRD TRIPLET AT  10.00 EV
        if EN > object.EnergyLevels[24]:
            object.InelasticCrossSectionPerGas[24][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTR3, YTR3, XTR3, 2) * 100
            if EN > 3 * object.EnergyLevels[24]:
                object.PEInelasticCrossSectionPerGas[24][I] = object.PEElasticCrossSection[1][I - IOFFN[24]]

        for J in range(25, 33):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 3 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1
        for J in range(33, 55):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        #LOAD BREMSSTRAHLUNG X-SECTION
        object.InelasticCrossSectionPerGas[55][I] = 0.0
        object.InelasticCrossSectionPerGas[56][I] = 0.0
        if EN > 1000:
            object.InelasticCrossSectionPerGas[55][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z6T, EBRM) * 2e-8
            object.InelasticCrossSectionPerGas[56][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z1T, EBRM) * 6e-8

        # LOAD NULL COLLISIONS
        object.NullCrossSection[0][I] = 0.0
        if EN > XNUL1[0]:
            object.NullCrossSection[0][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NUL1, YNUL1, XNUL1, 1) * 100 * <float>(0.9) * object.ScaleNull[0]

        # LIGHT EMISSION FROM H ALPHA   
        #   MOHLMMoleculesPerCm3PerGas AND DE HEER  CHEM.PHYS.19(1979)233 
        object.NullCrossSection[1][I] = 0.0
        if EN > XNUL2[0]:
            object.NullCrossSection[1][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NUL2, YNUL2, XNUL2, 1) * 100 * object.ScaleNull[1]

        # LIGHT EMISSION FROM CH2(A2DELTA - X2PI)
        #  MOHLMMoleculesPerCm3PerGas AND DE HEER  CHEM.PHYS.19(1979)233 
        object.NullCrossSection[2][I] = 0.0
        if EN > XNUL3[0]:
            object.NullCrossSection[2][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NUL3, YNUL3, XNUL3, 1) * 100 * object.ScaleNull[2]


    for J in range(object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break
    return
