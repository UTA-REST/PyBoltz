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
cdef void Gas15(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Oxygen gas.
    """
    gd = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"gases.npy")).item()
    cdef double XELA[153], YELA[153], YMOM[153], YEPS[153], XROT13[63], YROT13[63], XROT35[55], YROT35[55], XROT57[55], YROT57[55],
    cdef double XROT79[50], YROT79[50], XROT911[48], YROT911[48], XROT1113[46], YROT1113[46], XROT1315[45], YROT1315[45], XROT1517[44], YROT1517[44],
    cdef double XROT1719[43], YROT1719[43], XROT1921[41], YROT1921[41], XROT2123[40], YROT2123[40], XROT2325[39], YROT2325[39], XROT2527[38], YROT2527[38],
    cdef double XROT2729[37], YROT2729[37], XROT2931[36], YROT2931[36], XROT3133[34],
    cdef double YROT3133[34], XROT3335[33], YROT3335[33], XROT3537[32], YROT3537[32],
    cdef double XROT3739[32], YROT3739[32], XROT3941[31], YROT3941[31], XROT4143[31],
    cdef double YROT4143[31], XROT4345[30], YROT4345[30], XROT4547[30], YROT4547[30],
    cdef double XROT4749[29], YROT4749[29], YVIB9[60], YVIB10[60], YVIB11[60], YVIB12[60], YVIB13[60], YVIB14[60],
    cdef double XVIB[60], YVIB1[60], YVIB2[60], YVIB15[60], YVIB16[60], YVIB17[60], YVIB18[60], YVIB19[60],
    cdef double YVIB3[60], YVIB4[60], YVIB5[60], YVIB6[60], YVIB7[60], YVIB8[60], XEXC4[15], YEXC4[15], XEXC5[14], YEXC5[14], XEXC6[14], YEXC6[14],
    cdef double YVIB20[60], YVIB21[60], X3ATT[32], Y3ATT[32], XATT[31], YATT[31], XEXC1[40], YEXC1[40], XEXC2[31], YEXC2[31], XEXC3[16], YEXC3[16],
    cdef double XEXC7[14], YEXC7[14], XEXC8[15], YEXC8[15], XEXC9[14], YEXC9[14], XROT[48], YROT[48],
    cdef double XIONC[85], YIONC[85], XION1[85], YION1[85], XION2[70], YION2[70], XION3[50], YION3[50], XION4[47], YION4[47], XION5[44], YION5[44],
    cdef double XKSH[81], YKSH[81], PJ[50], Z8T[25], EBRM[25]

    cdef int IOFFN[150], IOFFION[8]

    XELA = gd['gas15/XELA']
    YELA = gd['gas15/YELA']
    YMOM = gd['gas15/YMOM']
    YEPS = gd['gas15/YEPS']
    XROT13 = gd['gas15/XROT13']
    YROT13 = gd['gas15/YROT13']
    XROT35 = gd['gas15/XROT35']
    YROT35 = gd['gas15/YROT35']
    XROT57 = gd['gas15/XROT57']
    YROT57 = gd['gas15/YROT57']
    XROT79 = gd['gas15/XROT79']
    YROT79 = gd['gas15/YROT79']
    XROT911 = gd['gas15/XROT911']
    YROT911 = gd['gas15/YROT911']
    XROT1113 = gd['gas15/XROT1113']
    YROT1113 = gd['gas15/YROT1113']
    XROT1315 = gd['gas15/XROT1315']
    YROT1315 = gd['gas15/YROT1315']
    XROT1517 = gd['gas15/XROT1517']
    YROT1517 = gd['gas15/YROT1517']
    XROT1719 = gd['gas15/XROT1719']
    YROT1719 = gd['gas15/YROT1719']
    XROT1921 = gd['gas15/XROT1921']
    YROT1921 = gd['gas15/YROT1921']
    XROT2123 = gd['gas15/XROT2123']
    YROT2123 = gd['gas15/YROT2123']
    XROT2325 = gd['gas15/XROT2325']
    YROT2325 = gd['gas15/YROT2325']
    XROT2527 = gd['gas15/XROT2527']
    YROT2527 = gd['gas15/YROT2527']
    XROT2729 = gd['gas15/XROT2729']
    YROT2729 = gd['gas15/YROT2729']
    XROT2931 = gd['gas15/XROT2931']
    YROT2931 = gd['gas15/YROT2931']
    XROT3133 = gd['gas15/XROT3133']
    YROT3133 = gd['gas15/YROT3133']
    XROT3335 = gd['gas15/XROT3335']
    YROT3335 = gd['gas15/YROT3335']
    XROT3537 = gd['gas15/XROT3537']
    YROT3537 = gd['gas15/YROT3537']
    XROT3739 = gd['gas15/XROT3739']
    YROT3739 = gd['gas15/YROT3739']
    XROT3941 = gd['gas15/XROT3941']
    YROT3941 = gd['gas15/YROT3941']
    XROT4143 = gd['gas15/XROT4143']
    YROT4143 = gd['gas15/YROT4143']
    XROT4345 = gd['gas15/XROT4345']
    YROT4345 = gd['gas15/YROT4345']
    XROT4547 = gd['gas15/XROT4547']
    YROT4547 = gd['gas15/YROT4547']
    XROT4749 = gd['gas15/XROT4749']
    YROT4749 = gd['gas15/YROT4749']
    XVIB = gd['gas15/XVIB']
    YVIB1 = gd['gas15/YVIB1']
    YVIB2 = gd['gas15/YVIB2']
    YVIB3 = gd['gas15/YVIB3']
    YVIB4 = gd['gas15/YVIB4']
    YVIB5 = gd['gas15/YVIB5']
    YVIB6 = gd['gas15/YVIB6']
    YVIB7 = gd['gas15/YVIB7']
    YVIB8 = gd['gas15/YVIB8']
    YVIB9 = gd['gas15/YVIB9']
    YVIB10 = gd['gas15/YVIB10']
    YVIB11 = gd['gas15/YVIB11']
    YVIB12 = gd['gas15/YVIB12']
    YVIB13 = gd['gas15/YVIB13']
    YVIB14 = gd['gas15/YVIB14']
    YVIB15 = gd['gas15/YVIB15']
    YVIB16 = gd['gas15/YVIB16']
    YVIB17 = gd['gas15/YVIB17']
    YVIB18 = gd['gas15/YVIB18']
    YVIB19 = gd['gas15/YVIB19']
    YVIB20 = gd['gas15/YVIB20']
    YVIB21 = gd['gas15/YVIB21']
    X3ATT = gd['gas15/X3ATT']
    Y3ATT = gd['gas15/Y3ATT']
    XATT = gd['gas15/XATT']
    YATT = gd['gas15/YATT']
    XEXC1 = gd['gas15/XEXC1']
    YEXC1 = gd['gas15/YEXC1']
    XEXC2 = gd['gas15/XEXC2']
    YEXC2 = gd['gas15/YEXC2']
    XEXC3 = gd['gas15/XEXC3']
    YEXC3 = gd['gas15/YEXC3']
    XEXC4 = gd['gas15/XEXC4']
    YEXC4 = gd['gas15/YEXC4']
    XEXC5 = gd['gas15/XEXC5']
    YEXC5 = gd['gas15/YEXC5']
    XEXC6 = gd['gas15/XEXC6']
    YEXC6 = gd['gas15/YEXC6']
    XEXC7 = gd['gas15/XEXC7']
    YEXC7 = gd['gas15/YEXC7']
    XEXC8 = gd['gas15/XEXC8']
    YEXC8 = gd['gas15/YEXC8']
    XEXC9 = gd['gas15/XEXC9']
    YEXC9 = gd['gas15/YEXC9']
    XROT = gd['gas15/XROT']
    YROT = gd['gas15/YROT']
    XIONC = gd['gas15/XIONC']
    YIONC = gd['gas15/YIONC']
    XION1 = gd['gas15/XION1']
    YION1 = gd['gas15/YION1']
    XION2 = gd['gas15/XION2']
    YION2 = gd['gas15/YION2']
    XION3 = gd['gas15/XION3']
    YION3 = gd['gas15/YION3']
    XION4 = gd['gas15/XION4']
    YION4 = gd['gas15/YION4']
    XION5 = gd['gas15/XION5']
    YION5 = gd['gas15/YION5']
    XKSH = gd['gas15/XKSH']
    YKSH = gd['gas15/YKSH']
    Z8T = gd['gas15/Z8T']
    EBRM = gd['gas15/EBRM']

    cdef int NBREM
    object.N_Ionization = 8
    object.N_Attachment = 1
    object.N_Inelastic = 148
    object.N_Null = 0
    NBREM = 25
    cdef double A0, RY, CONST, ElectronMass2, API, BBCONST, AM2, C, AUGK, B0, ROTSCALE, QBQA, QBK, ASum,
    cdef int i, j, I, J,
    A0 = 0.52917720859e-08
    RY = <float> (13.60569193)
    CONST = 1.873884e-20
    ElectronMass2 = <float> (1021997.804)
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / ElectronMass2
    #BORN BETHE VALUES FOR IONISATION
    AM2 = 4.00
    C = <float> (43.4)
    # AVERAGE AUGER EMISSION FROM KSHELL
    AUGK = 2.0
    # SCALE ROTATION X-SECTIONS BY ROTSCALE

    ROTSCALE = <float> (0.75)
    # ROTATIONAL QUADRUPOLE MOMENT
    QBQA = <float> (0.9)
    QBK = <float> (1.67552) * (QBQA * A0) ** 2
    B0 = 1.783e-4
    for J in range(2, 49, 2):
        PJ[J - 1] = 0.0

    for J in range(1, 50, 2):
        PJ[J - 1] = 3.0 * (2.0 * J + 1.0) * exp(-1 * J * (J + 1.0) * B0 / object.ThermalEnergy)

    ASum = 0.0
    for J in range(49):
        ASum += PJ[J]

    for J in range(49):
        PJ[J] /= ASum

    object.EnergyLevels = gd['gas15/EnergyLevels']

    for J in range(6):
        object.AngularModel[J] = object.WhichAngularModel
    for J in range(object.N_Inelastic):
        object.KIN[J] = object.WhichAngularModel

    cdef int NROT, NROT13, NROT35, NROT57, NROT79, NROT911, NROT1113, NROT1315, NROT1517, NROT1719, NROT1921, NROT2123
    cdef int NROT2325, NROT2527, NROT2729, NROT2931, NROT3133, NROT3335, NROT3537, NROT3739, NROT3941, NROT4143, NROT4345
    cdef int NROT4547, NROT4749, NELA, NVIB, N_Attachment1, N3ATT, NEXC1, NEXC2, NEXC3, NEXC4, NEXC5, NEXC6, NEXC7, NEXC8, NEXC9
    cdef int N_IonizationC, N_Ionization1, N_Ionization2, N_Ionization3, N_Ionization4, N_Ionization5, NKSH

    NROT = 48
    NROT13 = 63
    NROT35 = 55
    NROT57 = 55
    NROT79 = 50
    NROT911 = 48
    NROT1113 = 46
    NROT1315 = 45
    NROT1517 = 44
    NROT1719 = 43
    NROT1921 = 41
    NROT2123 = 40
    NROT2325 = 39
    NROT2527 = 38
    NROT2729 = 37
    NROT2931 = 36
    NROT3133 = 34
    NROT3335 = 33
    NROT3537 = 32
    NROT3739 = 32
    NROT3941 = 31
    NROT4143 = 31
    NROT4345 = 30
    NROT4547 = 30
    NROT4749 = 29
    NELA = 153
    NVIB = 60
    N_Attachment1 = 31
    N3ATT = 32
    NEXC1 = 40
    NEXC2 = 31
    NEXC3 = 16
    NEXC4 = 15
    NEXC5 = 14
    NEXC6 = 14
    NEXC7 = 14
    NEXC8 = 15
    NEXC9 = 14
    N_IonizationC = 85
    N_Ionization1 = 85
    N_Ionization2 = 70
    N_Ionization3 = 50
    N_Ionization4 = 47
    N_Ionization5 = 44
    NKSH = 81

    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, EOBY[8], FAC, APOP2

    object.E = [0.0, 1.0, <float> (12.071), 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * ElectronMass / (<float> (31.9988) * AMU)

    object.IonizationEnergy[0:8] = [<float> (12.071), <float> (16.104), <float> (18.171), <float> (20.701),
                                    <float> (38.46), 68.0,
                                    90.0, 532.0]
    for J in range(object.N_Ionization):
        object.EOBY[J] = 12.9

    for J in range(object.N_Ionization):
        object.NC0[J] = 0.0
        object.EC0[J] = 0.0
        object.WK[J] = 0.0
        object.EFL[J] = 0.0
        object.NG1[J] = 0.0
        object.EG1[J] = 0.0
        object.NG2[J] = 0.0
        object.EG2[J] = 0.0
    #DOUBLE CHARGED STATES
    object.NC0[4] = 1
    object.EC0[4] = 6.0
    object.NC0[5] = 1
    object.EC0[5] = 6.0
    #TRIPLE CHARGED STATES
    object.NC0[6] = 2
    object.EC0[6] = 6.0
    #FLUORESCENCE DATA
    object.NC0[7] = 3
    object.EC0[7] = 485
    object.WK[7] = <float> (0.0069)
    object.EFL[7] = 518
    object.NG1[7] = 1
    object.EG1[7] = 480
    object.NG2[7] = 2
    object.EG2[7] = 5.0

    # OFFSET ENERGY FOR I   ONISATION ELECTRON ANGULAR DISTRIBUTION
    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break

    cdef int NL = 0, FI
    for NL in range(object.N_Inelastic):
        for i in range(4000):
            if object.EG[i] > abs(object.EnergyLevels[NL]):
                IOFFN[NL] = i
                break
    cdef double F[70]
    F = [<float> (.00026), <float> (.000408), <float> (.000623), <float> (.001016), <float> (.001562),
         <float> (.002312), <float> (.003234), <float> (.004362), <float> (.005573), <float> (.006930),
         <float> (.008342), <float> (.009692), <float> (.010816), <float> (.011839),
         <float> (.012580), <float> (.013160), <float> (.013432), <float> (.013571), <float> (.013425),
         <float> (.012948), <float> (.010892), <float> (.006688), <float> (.002784), <float> (.001767),
         <float> (.000633), <float> (.000438), <float> (.000465), <float> (.008432), <float> (.007598),
         <float> (.000829), <float> (.000644), <float> (.001460), <float> (.000818), <float> (.000736),
         <float> (.000598), <float> (.001482), <float> (.000425), <float> (.001669), <float> (.001766),
         <float> (.001613), <float> (.001746), <float> (.003329), <float> (.006264), <float> (.013513),
         <float> (.011373), <float> (.006052), <float> (.006051), <float> (.004993), <float> (.005045),
         <float> (.004962), <float> (.006520), <float> (.008432), <float> (.011304), <float> (.015172),
         <float> (.022139), <float> (.032682), <float> (.039457), <float> (.029498), <float> (.018923),
         <float> (.017762), <float> (.015115), <float> (.013220), <float> (.009540), <float> (.005854),
         <float> (.008733), <float> (.007914), <float> (.008002), <float> (.006519), <float> (.003528),
         <float> (.001469), ]
    for J in range(75, object.N_Inelastic):
        object.PenningFraction[0][J] = 0.0
        object.PenningFraction[1][J] = 1.0
        object.PenningFraction[2][J] = 1.0
    cdef double EN, QMOM, ElasticCrossSectionA, PQ[3], BETA2, GAMMA1, GAMMA2, BETA, SINGLE , THREEB, SFAC, QRES1, ETEMP
    # CALCULATE DENSITY CORRECTION FOR THREE BODY ATTACHMENT CROSS-SECTION
    FAC = 273.15 * object.Pressure / ((object.TemperatureC + 273.15) * 760.0)
    T3B = 1.0
    # FIRST VIBRATIONAL LEVEL POPULATION
    APOP2 = exp(object.EnergyLevels[48] / object.ThermalEnergy)
    for J in range(NBREM):
        EBRM[J] = exp(EBRM[J])
    for I in range(4000):
        EN = object.EG[I]
        GAMMA1 = (ElectronMass2 + 2 * EN) / ElectronMass2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA
        # ELASTIC (+ROTATIONAL)
        ElasticCrossSectionA = GasUtil.CALIonizationCrossSectionREG(EN, NELA, YELA, XELA)
        QMOM = GasUtil.CALIonizationCrossSectionREG(EN, NELA, YMOM, XELA)
        PQ[2] = GasUtil.CALPQ3(EN, NELA, YEPS, XELA)

        PQ[2] = 1 - PQ[2]
        PQ[1] = 0.5 + (ElasticCrossSectionA - QMOM) / ElasticCrossSectionA
        PQ[0] = 0.5

        object.PEElasticCrossSection[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = ElasticCrossSectionA
        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM
        # IONISATION CALCULATION

        for J in range(object.N_Ionization):
            object.PEIonizationCrossSection[J][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEIonizationCrossSection[J][I] = 0.0
            object.IonizationCrossSection[J][I] = 0.0
        # IONISATION TO ALL CHMoleculesPerCm3PerGasELS WITH O2+
        # IONISATION TO O2+ X2PI
        if EN > object.IonizationEnergy[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization1, YION1, XION1,
                                                                                     BETA2,
                                                                                     <float> (0.6475) * <float> (0.558),
                                                                                     CONST, object.DEN[I],
                                                                                     C, AM2)

        # IONISATION TO O2+ X2PI
        if EN > object.IonizationEnergy[1]:
            object.IonizationCrossSection[1][I] = GasUtil.CALIonizationCrossSectionX(
                (EN - (object.IonizationEnergy[1] - object.IonizationEnergy[0])), N_Ionization1, YION1, XION1, BETA2,
                <float> (0.6475), CONST, object.DEN[I], C, AM2) * <float> (0.308)
            if EN <= XION1[N_Ionization1 - 1]:
                object.IonizationCrossSection[0][I] -= object.IonizationCrossSection[1][I]

        # IONISATION TO O2+ B4SIGMA
        if EN > object.IonizationEnergy[2]:
            object.IonizationCrossSection[2][I] = GasUtil.CALIonizationCrossSectionX(
                (EN - (object.IonizationEnergy[2] - object.IonizationEnergy[0])), N_Ionization1, YION1, XION1, BETA2,
                <float> (0.6475), CONST, object.DEN[I], C, AM2) * <float> (0.136)
            if EN <= XION1[N_Ionization1 - 1]:
                object.IonizationCrossSection[0][I] -= object.IonizationCrossSection[2][I]

        # DISSOCIATIVE IONISATION TO O+ + O
        if EN > object.IonizationEnergy[3]:
            object.IonizationCrossSection[3][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization2, YION2, XION2,
                                                                                     BETA2, <float> (0.2993), CONST,
                                                                                     object.DEN[I], C, AM2)

        # DISSOCIATIVE DOUBLE IONISATION TO O+ + O
        if EN > object.IonizationEnergy[4]:
            object.IonizationCrossSection[4][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization3, YION3, XION3,
                                                                                     BETA2, <float> (0.0446), CONST,
                                                                                     object.DEN[I], C, AM2)

        # DISSOCIATIVE DOUBLE IONISATION TO O++ + O
        if EN > object.IonizationEnergy[5]:
            object.IonizationCrossSection[5][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization4, YION4, XION4,
                                                                                     BETA2, <float> (0.0061), CONST,
                                                                                     object.DEN[I], C, AM2)

        # DISSOCIATIVE DOUBLE IONISATION TO O++ + O+
        if EN > object.IonizationEnergy[6]:
            object.IonizationCrossSection[6][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization5, YION5, XION5,
                                                                                     BETA2, <float> (0.0025), CONST,
                                                                                     object.DEN[I], C, AM2)

        # K-SHELL IONISATION
        if EN > object.IonizationEnergy[7]:
            object.IonizationCrossSection[7][I] = GasUtil.CALIonizationCrossSectionREG(EN, NKSH, YKSH, XKSH) * 2

        for J in range(object.N_Ionization):
            if EN > 2.0 * object.IonizationEnergy[J]:
                object.PEIonizationCrossSection[J][I] = object.PEElasticCrossSection[1][I - IOFFION[J]]

        # CORRECTION TO IONISATION FO AUGER EMISSION FROM KSHELL
        object.IonizationCrossSection[0][I] -= AUGK * object.IonizationCrossSection[7][I]

        # TWO BODY ATTACHMENT
        SINGLE = 0.0
        # OFFSET FOR ENERGY SCALE
        if EN >= XATT[0]:
            SINGLE = GasUtil.CALInelasticCrossSectionPerGasP(EN, N_Attachment1, YATT, XATT, 3) * 100

            #  THREE BODY ATTACHMENT
            # ***************************************************************
            #  ENTER HERE SCALING FACTOR FOR THREE BODY ATTACHMENT IN MIXTURES:
            #  FOR NORMAL SCALING T3B=1.0
            if EN>XATT[N_Attachment1-1]:
                T3B = 1.0
            #    SCALING FACTOR NORMALLY PROPORTIONAL TO OXYGEN FRACTION
            #    IN RARE GAS MIXTURES
            #
            #***********************************************************

        THREEB = 0.0
        if EN >= X3ATT[0]:
            THREEB = GasUtil.CALIonizationCrossSection(EN, N3ATT, Y3ATT, X3ATT) * FAC * T3B
        object.Q[3][I] = SINGLE + THREEB
        object.AttachmentCrossSection[0][I] = object.Q[3][I]

        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        # SET ZERO
        for J in range(object.N_Inelastic):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J][I] = 0.0

        # SUPERELASTIC ROTATION
        for J in range(1, 25):
            SFAC = (4.0 * J - 1.0) / (4.0 * J + 3.0)
            object.InelasticCrossSectionPerGas[J - 1][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.0
            if EN < 0.0:
                continue
            ETEMP = EN - object.EnergyLevels[J - 1]
            if J == 1:
                if (ETEMP <= XROT13[NROT13 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT13, YROT13, XROT13) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT13, YROT13,
                        XROT13, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 2:
                if (ETEMP <= XROT35[NROT35 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT35, YROT35, XROT35) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT35, YROT35,
                        XROT35, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 3:
                if (ETEMP <= XROT57[NROT57 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT57, YROT57, XROT57) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT57, YROT57,
                        XROT57, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 4:
                if (ETEMP <= XROT79[NROT79 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT79, YROT79, XROT79) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT79, YROT79,
                        XROT79, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 5:
                if (ETEMP <= XROT911[NROT911 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT911, YROT911, XROT911) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT911, YROT911,
                        XROT911, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 6:
                if (ETEMP <= XROT1113[NROT1113 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1113, YROT1113, XROT1113) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1113,
                        YROT1113, XROT1113, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 7:
                if (ETEMP <= XROT1315[NROT1315 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1315, YROT1315, XROT1315) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1315,
                        YROT1315, XROT1315, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 8:
                if (ETEMP <= XROT1517[NROT1517 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1517, YROT1517, XROT1517) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1517,
                        YROT1517, XROT1517, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 9:
                if (ETEMP <= XROT1719[NROT1719 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1719, YROT1719, XROT1719) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1719,
                        YROT1719, XROT1719, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 10:
                if (ETEMP <= XROT1921[NROT1921 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1921, YROT1921, XROT1921) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1921,
                        YROT1921, XROT1921, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 11:
                if (ETEMP <= XROT2123[NROT2123 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2123, YROT2123, XROT2123) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2123,
                        YROT2123, XROT2123, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 12:
                if (ETEMP <= XROT2325[NROT2325 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2325, YROT2325, XROT2325) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2325,
                        YROT2325, XROT2325, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 13:
                if (ETEMP <= XROT2527[NROT2527 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2527, YROT2527, XROT2527) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2527,
                        YROT2527, XROT2527, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 14:
                if (ETEMP <= XROT2729[NROT2729 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2729, YROT2729, XROT2729) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2729,
                        YROT2729, XROT2729, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 15:
                if (ETEMP <= XROT2931[NROT2931 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2931, YROT2931, XROT2931) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2931,
                        YROT2931, XROT2931, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 16:
                if (ETEMP <= XROT3133[NROT3133 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3133, YROT3133, XROT3133) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3133,
                        YROT3133, XROT3133, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 17:
                if (ETEMP <= XROT3335[NROT3335 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3335, YROT3335, XROT3335) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3335,
                        YROT3335, XROT3335, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 18:
                if (ETEMP <= XROT3537[NROT3537 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3537, YROT3537, XROT3537) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3537,
                        YROT3537, XROT3537, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 19:
                if (ETEMP <= XROT3739[NROT3739 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3739, YROT3739, XROT3739) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3739,
                        YROT3739, XROT3739, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 20:
                if (ETEMP <= XROT3941[NROT3941 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3941, YROT3941, XROT3941) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3941,
                        YROT3941, XROT3941, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 21:
                if (ETEMP <= XROT4143[NROT4143 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT4143, YROT4143, XROT4143) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT4143,
                        YROT4143, XROT4143, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 22:
                if (ETEMP <= XROT4345[NROT4345 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT4345, YROT4345, XROT4345) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT4345,
                        YROT4345, XROT4345, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 23:
                if (ETEMP <= XROT4547[NROT4547 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT4547, YROT4547, XROT4547) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT4547,
                        YROT4547, XROT4547, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif J == 24:
                if (ETEMP <= XROT4749[NROT4749 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = (ETEMP) * PJ[2 * J] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT4749, YROT4749, XROT4749) / EN
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT4749,
                        YROT4749, XROT4749, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            QRES1 = 0.0
            #CALCULATE ENHANCEMENT OF ROTATION DUE TO VIBRATIONAL RESONANCES
            if (ETEMP) <= XROT[NROT - 1] and (ETEMP) > XROT[0]:
                QRES1 = GasUtil.CALIonizationCrossSection(ETEMP, NROT, YROT, XROT) * (ETEMP) / EN * PJ[2 * J]
            object.InelasticCrossSectionPerGas[J - 1][I] += QRES1

        # INELASTIC ROTATION
        for J in range(25, 49):
            SFAC = 1
            i = J - 24
            object.InelasticCrossSectionPerGas[J - 1][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.0
            if EN < 0.0:
                continue
            if EN<object.EnergyLevels[J-1]:
                continue
            ETEMP = EN
            if i == 1:
                if (ETEMP <= XROT13[NROT13 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT13, YROT13, XROT13)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT13, YROT13, XROT13, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 2:
                if (ETEMP <= XROT35[NROT35 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT35, YROT35, XROT35)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT35, YROT35,
                        XROT35, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 3:
                if (ETEMP <= XROT57[NROT57 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT57, YROT57, XROT57)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT57, YROT57,
                        XROT57, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 4:
                if (ETEMP <= XROT79[NROT79 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT79, YROT79, XROT79)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT79, YROT79,
                        XROT79, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 5:
                if (ETEMP <= XROT911[NROT911 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT911, YROT911, XROT911)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT911, YROT911,
                        XROT911, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 6:
                if (ETEMP <= XROT1113[NROT1113 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1113, YROT1113, XROT1113)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1113,
                        YROT1113, XROT1113, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 7:
                if (ETEMP <= XROT1315[NROT1315 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1315, YROT1315, XROT1315)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1315,
                        YROT1315, XROT1315, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 8:
                if (ETEMP <= XROT1517[NROT1517 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1517, YROT1517, XROT1517)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1517,
                        YROT1517, XROT1517, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 9:
                if (ETEMP <= XROT1719[NROT1719 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1719, YROT1719, XROT1719)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1719,
                        YROT1719, XROT1719, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 10:
                if (ETEMP <= XROT1921[NROT1921 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT1921, YROT1921, XROT1921)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT1921,
                        YROT1921, XROT1921, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 11:
                if (ETEMP <= XROT2123[NROT2123 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2123, YROT2123, XROT2123)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2123,
                        YROT2123, XROT2123, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 12:
                if (ETEMP <= XROT2325[NROT2325 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2325, YROT2325, XROT2325)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2325,
                        YROT2325, XROT2325, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 13:
                if (ETEMP <= XROT2527[NROT2527 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2527, YROT2527, XROT2527)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2527,
                        YROT2527, XROT2527, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 14:
                if (ETEMP <= XROT2729[NROT2729 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2729, YROT2729, XROT2729)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2729,
                        YROT2729, XROT2729, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 15:
                if (ETEMP <= XROT2931[NROT2931 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT2931, YROT2931, XROT2931)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT2931,
                        YROT2931, XROT2931, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 16:
                if (ETEMP <= XROT3133[NROT3133 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3133, YROT3133, XROT3133)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3133,
                        YROT3133, XROT3133, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 17:
                if (ETEMP <= XROT3335[NROT3335 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3335, YROT3335, XROT3335)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3335,
                        YROT3335, XROT3335, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 18:
                if (ETEMP <= XROT3537[NROT3537 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3537, YROT3537, XROT3537)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3537,
                        YROT3537, XROT3537, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 19:
                if (ETEMP <= XROT3739[NROT3739 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3739, YROT3739, XROT3739)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3739,
                        YROT3739, XROT3739, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 20:
                if (ETEMP <= XROT3941[NROT3941 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT3941, YROT3941, XROT3941)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT3941,
                        YROT3941, XROT3941, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 21:
                if (ETEMP <= XROT4143[NROT4143 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT4143, YROT4143, XROT4143)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT4143,
                        YROT4143, XROT4143, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 22:
                if (ETEMP <= XROT4345[NROT4345 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT4345, YROT4345, XROT4345)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT4345,
                        YROT4345, XROT4345, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 23:
                if (ETEMP <= XROT4547[NROT4547 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT4547, YROT4547, XROT4547)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT4547,
                        YROT4547, XROT4547, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            elif i == 24:
                if (ETEMP <= XROT4749[NROT4749 - 1]):
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[2 * J - 50] * SFAC * GasUtil.QLSCALE(
                        ETEMP, NROT4749, YROT4749, XROT4749)
                else:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[
                                                                       2 * J - 50] * SFAC * GasUtil.CALInelasticCrossSectionPerGasP(
                        ETEMP, NROT4749,
                        YROT4749, XROT4749, 1) * 100
                if EN > 3.0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            QRES1 = 0.0
            #CALCULATE ENHANCEMENT OF ROTATION DUE TO VIBRATIONAL RESONANCES
            if (EN) <= XROT[NROT - 1] and (EN) > XROT[0]:
                QRES1 = GasUtil.CALIonizationCrossSection(EN, NROT, YROT, XROT) * PJ[2 * J - 50]
            object.InelasticCrossSectionPerGas[J - 1][I] += QRES1

        # FORCE ROTATIONAL X-SECTION TO FALL AS 1/E**2 ABOVE 3 EV
        # NB 1/E ALREADY USED SO ONLY 1/E EXTRA
        if EN >= 3.0:
            for J in range(48):
                object.InelasticCrossSectionPerGas[J][I] *= (3.0 / EN)

        # SCALE ROTATION X-SECTIONS BY ROTSCALE
        for J in range(48):
            object.InelasticCrossSectionPerGas[J][I] *= ROTSCALE

        # SUPERELASTIC V1
        if EN != 0.0:
            ETEMP = EN + object.EnergyLevels[49]
            object.InelasticCrossSectionPerGas[48][I] = GasUtil.CALInelasticCrossSectionPerGasP(ETEMP, NVIB, YVIB1,
                                                                                                XVIB, 1) * 100 * (
                                                                    APOP2 / (1 + APOP2))
            if ETEMP <= XVIB[NVIB - 1]:
                object.InelasticCrossSectionPerGas[48][I] *= (ETEMP / EN)
            else:
                object.InelasticCrossSectionPerGas[48][I] = YVIB1[NVIB - 1] * (XVIB[NVIB - 1] / EN) * (
                            APOP2 / (1 + APOP2)) * 1e-16
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[48][I] = object.PEElasticCrossSection[1][I - IOFFN[48]]

        # VIB1
        if EN > object.EnergyLevels[49]:
            object.InelasticCrossSectionPerGas[49][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB1, XVIB,
                                                                                                1) * 100 * (
                                                                    1 / (1 + APOP2))
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[49][I] = object.PEElasticCrossSection[1][I - IOFFN[49]]

        if EN > object.EnergyLevels[49]:
            object.InelasticCrossSectionPerGas[49][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB1, XVIB,
                                                                                                1) * 100 * (
                                                                    1 / (1 + APOP2))
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[49][I] = object.PEElasticCrossSection[1][I - IOFFN[49]]

        if EN > object.EnergyLevels[50]:
            object.InelasticCrossSectionPerGas[50][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB2, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[50][I] = object.PEElasticCrossSection[1][I - IOFFN[50]]

        if EN > object.EnergyLevels[51]:
            object.InelasticCrossSectionPerGas[51][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB3, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[51][I] = object.PEElasticCrossSection[1][I - IOFFN[51]]

        if EN > object.EnergyLevels[52]:
            object.InelasticCrossSectionPerGas[52][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB4, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[52][I] = object.PEElasticCrossSection[1][I - IOFFN[52]]

        if EN > object.EnergyLevels[53]:
            object.InelasticCrossSectionPerGas[53][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB5, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[53][I] = object.PEElasticCrossSection[1][I - IOFFN[53]]

        # A1 DELTA
        if EN > object.EnergyLevels[54]:
            object.InelasticCrossSectionPerGas[54][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC1, YEXC1, XEXC1,
                                                                                                2) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[54][I] = object.PEElasticCrossSection[1][I - IOFFN[54]]

        if EN > object.EnergyLevels[55]:
            object.InelasticCrossSectionPerGas[55][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB6, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[55][I] = object.PEElasticCrossSection[1][I - IOFFN[55]]

        if EN > object.EnergyLevels[56]:
            object.InelasticCrossSectionPerGas[56][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB7, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[56][I] = object.PEElasticCrossSection[1][I - IOFFN[56]]

        if EN > object.EnergyLevels[57]:
            object.InelasticCrossSectionPerGas[57][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB8, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[57][I] = object.PEElasticCrossSection[1][I - IOFFN[57]]

        # B1 SIGMA
        if EN > object.EnergyLevels[58]:
            object.InelasticCrossSectionPerGas[58][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC2, YEXC2, XEXC2,
                                                                                                2) * 100
            if EN > 3.0 * object.EnergyLevels[58]:
                object.PEInelasticCrossSectionPerGas[60][I] = object.PEElasticCrossSection[1][I - IOFFN[58]]

        if EN > object.EnergyLevels[59]:
            object.InelasticCrossSectionPerGas[59][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB9, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[59][I] = object.PEElasticCrossSection[1][I - IOFFN[59]]

        if EN > object.EnergyLevels[60]:
            object.InelasticCrossSectionPerGas[60][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB10, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[60][I] = object.PEElasticCrossSection[1][I - IOFFN[60]]

        if EN > object.EnergyLevels[61]:
            object.InelasticCrossSectionPerGas[61][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB11, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[61][I] = object.PEElasticCrossSection[1][I - IOFFN[61]]

        if EN > object.EnergyLevels[62]:
            object.InelasticCrossSectionPerGas[62][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB12, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[62][I] = object.PEElasticCrossSection[1][I - IOFFN[62]]

        if EN > object.EnergyLevels[63]:
            object.InelasticCrossSectionPerGas[63][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB13, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[63][I] = object.PEElasticCrossSection[1][I - IOFFN[63]]

        if EN > object.EnergyLevels[64]:
            object.InelasticCrossSectionPerGas[64][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB14, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[64][I] = object.PEElasticCrossSection[1][I - IOFFN[64]]

        if EN > object.EnergyLevels[65]:
            object.InelasticCrossSectionPerGas[65][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB15, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[65][I] = object.PEElasticCrossSection[1][I - IOFFN[65]]

        if EN > object.EnergyLevels[66]:
            object.InelasticCrossSectionPerGas[66][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB16, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[66][I] = object.PEElasticCrossSection[1][I - IOFFN[66]]

        if EN > object.EnergyLevels[67]:
            object.InelasticCrossSectionPerGas[67][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB17, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[67][I] = object.PEElasticCrossSection[1][I - IOFFN[67]]

        if EN > object.EnergyLevels[68]:
            object.InelasticCrossSectionPerGas[68][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB18, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[68][I] = object.PEElasticCrossSection[1][I - IOFFN[68]]

        if EN > object.EnergyLevels[69]:
            object.InelasticCrossSectionPerGas[69][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB19, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[69][I] = object.PEElasticCrossSection[1][I - IOFFN[69]]

        if EN > object.EnergyLevels[70]:
            object.InelasticCrossSectionPerGas[70][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB20, XVIB,
                                                                                                1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[70][I] = object.PEElasticCrossSection[1][I - IOFFN[70]]

        if EN > object.EnergyLevels[71]:
            object.InelasticCrossSectionPerGas[71][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB21, XVIB,
                                                                                                1) * 100
            # TODO: ERROR maybe matching magboltz
            if EN > XVIB[NVIB - 1]:
                object.InelasticCrossSectionPerGas[71][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB, YVIB20,
                                                                                                    XVIB, 1) * 100
            if EN > 6.0:
                object.PEInelasticCrossSectionPerGas[71][I] = object.PEElasticCrossSection[1][I - IOFFN[71]]

        #   HERZBERG CONTINUUM  C1SIG +A!3DEL + A3SIG
        # PART1
        if EN > object.EnergyLevels[72]:
            object.InelasticCrossSectionPerGas[72][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC3, YEXC3, XEXC3,
                                                                                                1) * 100
            if EN > 2 * object.EnergyLevels[72]:
                object.PEInelasticCrossSectionPerGas[72][I] = object.PEElasticCrossSection[1][I - IOFFN[72]]

        # PART2
        if EN > object.EnergyLevels[73]:
            object.InelasticCrossSectionPerGas[73][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC4, YEXC4, XEXC4,
                                                                                                1) * 100
            if EN > 2 * object.EnergyLevels[73]:
                object.PEInelasticCrossSectionPerGas[73][I] = object.PEElasticCrossSection[1][I - IOFFN[73]]

        # PART3
        if EN > object.EnergyLevels[74]:
            object.InelasticCrossSectionPerGas[74][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC5, YEXC5, XEXC5,
                                                                                                1) * 100
            if EN > 2 * object.EnergyLevels[74]:
                object.PEInelasticCrossSectionPerGas[74][I] = object.PEElasticCrossSection[1][I - IOFFN[74]]

        FI = 0
        for J in range(75, 87):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        if object.InelasticCrossSectionPerGas[75][I] < 0.0:
            object.InelasticCrossSectionPerGas[75][I] = 0.0

        # Sum OF RESONANCES (NON-DIPOLE) IN S-R CONTINUUM AT 8.20EV
        if EN > object.EnergyLevels[87]:
            object.InelasticCrossSectionPerGas[87][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC6, YEXC6, XEXC6,
                                                                                                1) * 100
            if EN > 2 * object.EnergyLevels[87]:
                object.PEInelasticCrossSectionPerGas[87][I] = object.PEElasticCrossSection[1][I - IOFFN[87]]

        for J in range(88, 104):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        # ADD RESONANT COMPONENT TO LONG BAND
        if EN > object.EnergyLevels[103]:
            object.InelasticCrossSectionPerGas[103][I] += GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC7, YEXC7,
                                                                                                  XEXC7, 1) * 100
            if EN > 2 * object.EnergyLevels[103]:
                object.PEInelasticCrossSectionPerGas[103][I] = object.PEElasticCrossSection[1][I - IOFFN[103]]

        for J in range(104, 106):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
            if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                object.InelasticCrossSectionPerGas[J][I] = 0.0
            if EN > 2 * object.EnergyLevels[J]:
                object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        # TRIPLET Sum BELOW IP
        if EN > object.EnergyLevels[106]:
            object.InelasticCrossSectionPerGas[106][I] += GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC8, YEXC8,
                                                                                                  XEXC8, 1.5) * 100
            if EN > 2 * object.EnergyLevels[106]:
                object.PEInelasticCrossSectionPerGas[106][I] = object.PEElasticCrossSection[1][I - IOFFN[106]]

        for J in range(107, 122):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
            if EN > 2 * object.EnergyLevels[J]:
                object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        # TRIPLET Sum ABOVE IP
        if EN > object.EnergyLevels[122]:
            object.InelasticCrossSectionPerGas[122][I] += GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC9, YEXC9,
                                                                                                  XEXC9, 1.5) * 100
            if EN > 2 * object.EnergyLevels[122]:
                object.PEInelasticCrossSectionPerGas[122][I] = object.PEElasticCrossSection[1][I - IOFFN[122]]

        for J in range(123, 148):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        #LOAD BREMSSTRAHLUNG X-SECTIONS
        object.InelasticCrossSectionPerGas[148][I] = 0.0
        if EN > 1000:
            object.InelasticCrossSectionPerGas[148][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z8T, EBRM) * 2e-8

        SumION = 0.0
        for J in range(object.N_Ionization):
            SumION += object.IonizationCrossSection[J][I]

        SumEXC = 0.0
        for J in range(48, object.N_Inelastic):
            SumEXC += object.InelasticCrossSectionPerGas[J][I]
        SumEXC += object.Q[3][I]

        object.Q[0][I] = object.Q[1][I] + SumION + SumEXC


    for I in range(1, 149):
        J = 149 - I - 1
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
    if object.N_Inelastic < 52:
        object.N_Inelastic = 52
    '''
    print(object.PEElasticCrossSection[1][0])
    print("************")
    print(object.PEElasticCrossSection[1][9])
    if object.FinalEnergy >= 22.62:
        for J in range(6):
            print(object.Q[J][99], J)

        print("I = 4000")
        for J in range(48, object.N_Inelastic):
            print(object.InelasticCrossSectionPerGas[J][99], J)
        sys.exit()
    '''
    '''5000-6000'''
    return
