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
    """
    This function is used to calculate the needed momentum cross sections for Nitrogen gas.
    """
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

    cdef double A0, RY, CONST, ElectronMass2, API, BBCONST, AM2, C, AUGK, B0
    cdef int NBREM, i, j, I, J,
    A0 = 0.52917720859e-08
    RY = <float>(13.60569193)
    CONST = 1.873884e-20
    ElectronMass2 = <float>(1021997.804)
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / ElectronMass2
    #BORN BETHE VALUES FOR IONISATION
    AM2 = <float>(3.35)
    C = <float>(38.1)
    # AVERAGE AUGER EMISSION FROM KSHELL
    AUGK = <float>(1.99)

    object.N_Ionization = 12
    object.N_Attachment = 1
    object.N_Inelastic = 127
    object.N_Null = 0
    NBREM = 25

    # ANGULAR DISTRIBUTIONS OF ELASTIC AND IONISATION CAN BE EITHER
    # ISOTROPIC (KEL=0) OR
    # CAPITELLI-LONGO (KEL =1)  OR OKHRIMOVSKKY TYPES (KEL=2)
    for J in range(6):
        object.AngularModel[J] = object.WhichAngularModel

    # USE ISOTROPIC SCATTERING FOR ROTATIONAL AND VIBRATIONAL STATES
    for J in range(92):
        object.KIN[J] = 0

    # USE AAnisotropicDetectedTROPIC SCATTERING FOR VIBRATIONAL AND EXCITED STATES .
    # ANGULAR DISTRIBUTIONS ARE CAPITELLI-LONGO (FORWARD BACKWARD ASYMMETRY)
    # OR OKRIMOVSKKY

    for J in range(92, object.N_Inelastic):
        object.KIN[J] = 1

    cdef int NELA, NROT, NVIB1, NVIB2, NVIB3, NVIB4, NVIB5, NVIB6, NVIB7, NVIB8, NVIB9, NVIB10, NVIB11, NVIB12, NVIB13
    cdef int NVIB14, NVIB15, NTRP1, NTRP2, NTRP3, NTRP4, NTRP5, NTRP6, NTRP7, NTRP8, NTRP9, NTRP10, NTRP11, NTRP12,
    cdef int NTRP13, NTRP14, NSNG1, NSNG2, NSNG3, NSNG4, NSNG5, NSNG6, NSNG7, NSNG8, NSNG9, NSNG10, NSNG11, NSNG12,
    cdef int NSNG13, NSNG14, NSNG15, N_IonizationD, N_Ionization1, N_Ionization2, N_Ionization3, N_Ionization4, NKSH,

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
    N_IonizationD = 87
    N_Ionization1 = 87
    N_Ionization2 = 63
    N_Ionization3 = 48
    N_Ionization4 = 54
    NKSH = 89

    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, EOBY[12], SumR, SumV, SumEX, SumEX1

    object.E = [0.0, 1.0, <float>(15.581), 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * ElectronMass / (<float>(27.7940) * AMU)
    object.IonizationEnergy[0:12] = [<float>(15.581), <float>(15.855), <float>(16.699), <float>(16.935), <float>(17.171), <float>(18.751), <float>(23.591), <float>(24.294), <float>(24.4), <float>(35.7), <float>(38.8), <float>(401.6)]

    for J in range(12):
        EOBY[J] = <float>(13.6)
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
    object.EC0[11] = <float>(358.6)
    object.WK[11] = <float>(0.0044)
    object.EFL[11] = 385.
    object.NG1[11] = 1
    object.EG1[11] = 353.
    object.NG2[11] = 1
    object.EG2[11] = 6.

    cdef double QBQA, QBK, Sum, FROT0, PJ[39], RAT
    #CALC FRACTIONAL POPULATION DENSITY FOR ROTATIONAL STATES
    B0 = 2.4668e-4
    #ROTATIONAL QUADRUPOLE MOMENT
    QBQA = <float>(1.045)
    QBK = <float>(1.67552) * (QBQA * A0) ** 2
    for J in range(1, 40, 2):
        PJ[J - 1] = 3.0 * (2.0 * J + 1.0) * exp(-1.0 * J * (J + 1.0) * B0 / object.ThermalEnergy)
    for J in range(2, 39, 2):
        PJ[J - 1] = 6.0 * (2.0 * J + 1.0) * exp(-1.0 * J * (J + 1.0) * B0 / object.ThermalEnergy)
    Sum = 6.0
    for J in range(39):
        Sum += PJ[J]
    FROT0 = 6.0 / Sum
    for J in range(39):
        PJ[J] /= Sum

    object.EnergyLevels = gd['gas16/EnergyLevels']

    # OFFSET ENERGY FOR IONISATION ELECTRON ANGULAR DISTRIBUTION
    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break

    cdef int NL = 0
    for NL in range(object.N_Inelastic):
        for i in range(4000):
            if object.EG[i] > abs(object.EnergyLevels[NL]):
                IOFFN[NL] = i
                break

    for I in range(106):
        for J in range(3):
            object.PenningFraction[I][J] = 0.0

    for J in range(106, 127):
        object.PenningFraction[0][J] = 0.0
        object.PenningFraction[1][J] = 1.0
        object.PenningFraction[2][J] = 1.0

    cdef double APOPV1, APOPGS, APOPSum, EN, GAMMA1, GAMMA2, BETA, BETA2, ElasticCrossSectionA, QMOM, PQ[3], QN2PTOT, QNPTOT, RESFAC, ASCALE

    for J in range(NBREM):
        EBRM[J] = exp(EBRM[J])
    # CALC VIBRATIONAL LEVEL V1 POPULATION
    APOPV1 = exp(object.EnergyLevels[76] / object.ThermalEnergy)
    APOPGS = 1.0
    APOPSum = APOPGS + APOPV1
    APOPV1 = APOPV1 / APOPSum
    APOPGS = APOPGS / APOPSum
    #  RENORMALISE GROUND STATE TO ALLOW FOR EXCITATION FROM
    #  THE EXCITED VIBRATIONAL STATE
    APOPGS = 1.0

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

        for J in range(12):
            object.PEIonizationCrossSection[J][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEIonizationCrossSection[J][I] = 0.0
            object.IonizationCrossSection[J][I] = 0.0

        # IONISATION TO ALL CHMoleculesPerCm3PerGasELS WITH N2+
        QN2PTOT = 0.0
        if EN > object.IonizationEnergy[0]:
            QN2PTOT = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization1, YION1, XION1, BETA2, <float>(0.7973), CONST, object.DEN[I], C, AM2)

        object.IonizationCrossSection[0][I] = QN2PTOT

        if EN > object.IonizationEnergy[1] and EN <= object.IonizationEnergy[2]:
            object.IonizationCrossSection[1][I] = QN2PTOT * <float>(0.2)
            object.IonizationCrossSection[0][I] = QN2PTOT * <float>(0.8)
        elif EN > object.IonizationEnergy[2] and EN <= object.IonizationEnergy[3]:
            object.IonizationCrossSection[2][I] = QN2PTOT * <float>(0.1986)
            object.IonizationCrossSection[1][I] = QN2PTOT * <float>(0.1603)
            object.IonizationCrossSection[0][I] = QN2PTOT * <float>(0.6411)
        elif EN > object.IonizationEnergy[3] and EN <= object.IonizationEnergy[4]:
            object.IonizationCrossSection[3][I] = QN2PTOT * <float>(0.2296)
            object.IonizationCrossSection[2][I] = QN2PTOT * <float>(0.1530)
            object.IonizationCrossSection[1][I] = QN2PTOT * <float>(0.1235)
            object.IonizationCrossSection[0][I] = QN2PTOT * <float>(0.4939)
        elif EN > object.IonizationEnergy[4] and EN <= object.IonizationEnergy[5]:
            object.IonizationCrossSection[4][I] = QN2PTOT * <float>(0.2765)
            object.IonizationCrossSection[3][I] = QN2PTOT * <float>(0.1659)
            object.IonizationCrossSection[2][I] = QN2PTOT * <float>(0.1106)
            object.IonizationCrossSection[1][I] = QN2PTOT * <float>(0.0894)
            object.IonizationCrossSection[0][I] = QN2PTOT * <float>(0.3576)
        elif EN > object.IonizationEnergy[5] and EN <= object.IonizationEnergy[6]:
            object.IonizationCrossSection[5][I] = QN2PTOT * <float>(0.1299)
            object.IonizationCrossSection[4][I] = QN2PTOT * <float>(0.2408)
            object.IonizationCrossSection[3][I] = QN2PTOT * <float>(0.1445)
            object.IonizationCrossSection[2][I] = QN2PTOT * <float>(0.0963)
            object.IonizationCrossSection[1][I] = QN2PTOT * <float>(0.0777)
            object.IonizationCrossSection[0][I] = QN2PTOT * <float>(0.3108)
        elif EN > object.IonizationEnergy[6]:
            object.IonizationCrossSection[6][I] = QN2PTOT * <float>(0.022)
            object.IonizationCrossSection[5][I] = QN2PTOT * <float>(0.127)
            object.IonizationCrossSection[4][I] = QN2PTOT * <float>(0.2355)
            object.IonizationCrossSection[3][I] = QN2PTOT * <float>(0.1413)
            object.IonizationCrossSection[2][I] = QN2PTOT * <float>(0.0942)
            object.IonizationCrossSection[1][I] = QN2PTOT * <float>(0.076)
            object.IonizationCrossSection[0][I] = QN2PTOT * <float>(0.304)

        # IONISATION TO aLL CHMoleculesPerCm3PerGasELS WITH N +
        QNPTOT = 0.0
        if EN > object.IonizationEnergy[7]:
            QNPTOT = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization2, YION2, XION2, BETA2, <float>(0.197), CONST, object.DEN[I], C, AM2)

        object.IonizationCrossSection[7][I] = QNPTOT
        if EN > object.IonizationEnergy[8] and EN <= object.IonizationEnergy[9]:
            if EN < 110:
                object.IonizationCrossSection[8][I] = ((EN - object.IonizationEnergy[8]) / (110. - object.IonizationEnergy[8])) * <float>(0.095) * 1.e-16
            else:
                object.IonizationCrossSection[8][I] = object.IonizationCrossSection[7][I] * <float>(0.1439)
            object.IonizationCrossSection[7][I] = object.IonizationCrossSection[7][I] - object.IonizationCrossSection[8][I]

        elif EN > object.IonizationEnergy[9]:
            if EN < 110:
                object.IonizationCrossSection[8][I] = ((EN - object.IonizationEnergy[8]) / (110. - object.IonizationEnergy[8])) * <float>(0.095) * 1.e-16
            if EN >= 110:
                object.IonizationCrossSection[8][I] = object.IonizationCrossSection[7][I] * <float>(0.1439)
            if EN < 120:
                object.IonizationCrossSection[9][I] = ((EN - object.IonizationEnergy[9]) / (120. - object.IonizationEnergy[9])) * <float>(0.037) * 1.e-16
            else:
                object.IonizationCrossSection[9][I] = object.IonizationCrossSection[7][I] * <float>(0.056)
            object.IonizationCrossSection[7][I] = object.IonizationCrossSection[7][I] - object.IonizationCrossSection[8][I] - object.IonizationCrossSection[9][I]

        if EN > object.IonizationEnergy[10]:
            # Sum OF DOUBLE IONISATION CHMoleculesPerCm3PerGasELS: N+,N+  AND N++,N
            object.IonizationCrossSection[10][I] = 0.0
            object.IonizationCrossSection[10][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization3, YION3, XION3, BETA2, <float>(0.0338), CONST, object.DEN[I], C, AM2)
            object.IonizationCrossSection[7][I] -= object.IonizationCrossSection[10][I]

        if EN > 65.0:
            object.IonizationCrossSection[10][I] += GasUtil.CALIonizationCrossSectionX(EN, N_Ionization4, YION4, XION4, BETA2, <float>(0.0057), CONST, object.DEN[I], C, AM2)

        if EN > object.IonizationEnergy[11]:
            object.IonizationCrossSection[11][I] = 2 * GasUtil.CALIonizationCrossSectionREG(EN, NKSH, YKSH, XKSH)

        for J in range(12):
            if EN > 2 * object.IonizationEnergy[J]:
                object.PEIonizationCrossSection[J][I] = object.PEElasticCrossSection[1][I - IOFFION[J]]

        # CORRECTION TO IONISATION FOR AUGER EMISSION
        object.IonizationCrossSection[0][I] -= AUGK * object.IonizationCrossSection[11][I]

        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = 0.0
        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        #---------------------------------------------------------------------
        #  QUADRUPOLE BORN ROTATIONAL STATES  ( GERJUOY AND STEnergyLevels)
        #---------------------------------------------------------------------
        #
        #  SUPERELASTIC ROTATION
        #

        for J in range(1, 39):
            object.InelasticCrossSectionPerGas[J - 1][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.0
            if EN > 0.0:
                i = J + 1
                object.InelasticCrossSectionPerGas[J - 1][I] = PJ[J] * QBK * sqrt(1.0 - object.EnergyLevels[J - 1] / EN) * i * (i - 1.0) / (
                        (2.0 * i + 1.0) * (2.0 * i - 1.0))
                #CALCULATE ENHANCEMENT OF ROTATIONAL XSEC IN THE RESONANCE REGION
                RESFAC = GasUtil.CALPQ3(EN - object.EnergyLevels[J - 1], NROT, YROT, XROT)
                if (EN - object.EnergyLevels[J-1])<=XROT[0]:
                    RESFAC = (YROT[0]/XROT[0]*(EN - object.EnergyLevels[J-1]))
                RESFAC*=(EN - object.EnergyLevels[J-1])/EN
                #USE 30% FOR RESFAC
                RESFAC *= <float>(0.3)
                #BORN ROTATIONAL X-SEC Sum IN RESONANCE REGION = 0.249
                RESFAC = 1.0 + RESFAC / <float>(0.249)
            object.InelasticCrossSectionPerGas[J - 1][I] *= RESFAC

        # INELASTIC ROTATION
        #
        # CALCULATE ENHANCEMENT OF ROTATIONAL XSEC IN THE RESONANCE REGION

        for J in range(39, 77):
            object.InelasticCrossSectionPerGas[J - 1][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.0
            if EN> 0.0:
                # CALCULATE ENHANCEMENT OF ROTATIONAL XSEC IN THE RESONANCE REGION
                RESFAC = GasUtil.CALPQ3(EN, NROT, YROT, XROT)
                if (EN - object.EnergyLevels[J-1])<=XROT[0]:
                    RESFAC = (YROT[0]*EN)
                #USE 30% FOR RESFAC
                RESFAC *= <float>(0.3)
                #BORN ROTATIONAL X-SEC Sum IN RESONANCE REGION = 0.249
                RESFAC = 1.0 + RESFAC / <float>(0.249)

        # ROT 0-2
        if EN > object.EnergyLevels[38]:
            object.InelasticCrossSectionPerGas[38][I] = FROT0 * QBK * sqrt(1.0 - object.EnergyLevels[38] / EN) * <float>(2.0 / 3.0)
            object.InelasticCrossSectionPerGas[38][I] *= RESFAC
            object.PEInelasticCrossSectionPerGas[38][I] = 0.0
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[38][I] = 0.0
            for J in range(40, 77):
                i = J - 39
                if EN > object.EnergyLevels[J - 1]:
                    object.InelasticCrossSectionPerGas[J - 1][I] = PJ[i - 1] * QBK * sqrt(1.0 - object.EnergyLevels[J - 1] / EN) * (i + 2.0) * (
                            i + 1.0) / ((2.0 * i + 3.0) * (2.0 * i + 1.0))
                    object.InelasticCrossSectionPerGas[J - 1][I] *= RESFAC

            if EN >= 5.0:
                ASCALE = QMOM / 8.9e-16
                for J in range(76):
                    object.InelasticCrossSectionPerGas[J][I] *= ASCALE

        #---------------------------------------------------------------------
        #  VIBRATIONAL AND EXCITATION X-SECTIONS
        #---------------------------------------------------------------------
        #  V1 SUPERELASTIC
        object.InelasticCrossSectionPerGas[76][I] = 0
        object.PEInelasticCrossSectionPerGas[76][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[76][I] = 0.0
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[76][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, NVIB1, YVB1, XVB1, APOPV1, -1 * object.EnergyLevels[76], 1,  object.EnergyLevels[76], 0, 0)

        # V1
        object.InelasticCrossSectionPerGas[77][I] = 0.0
        if EN > object.EnergyLevels[77]:
            object.InelasticCrossSectionPerGas[77][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB1, YVB1, XVB1, 1) * 100 * APOPGS

        # V2
        object.InelasticCrossSectionPerGas[78][I] = 0.0
        if EN > object.EnergyLevels[78]:
            object.InelasticCrossSectionPerGas[78][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB2, YVB2, XVB2, 1) * 100 * APOPGS

        # 3V1
        object.InelasticCrossSectionPerGas[79][I] = 0.0
        if EN > object.EnergyLevels[79]:
            object.InelasticCrossSectionPerGas[79][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB3, YVB3, XVB3, 1) * 100 * APOPGS

        # 4V1
        object.InelasticCrossSectionPerGas[80][I] = 0.0
        if EN > object.EnergyLevels[80]:
            object.InelasticCrossSectionPerGas[80][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB4, YVB4, XVB4, 1) * 100 * APOPGS

        # 5V1
        object.InelasticCrossSectionPerGas[81][I] = 0.0
        if EN > object.EnergyLevels[81]:
            object.InelasticCrossSectionPerGas[81][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB5, YVB5, XVB5, 1) * 100 * APOPGS
        # 6V1
        object.InelasticCrossSectionPerGas[82][I] = 0.0
        if EN > object.EnergyLevels[82]:
            object.InelasticCrossSectionPerGas[82][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB6, YVB6, XVB6, 1) * 100 * APOPGS
        # 7V1
        object.InelasticCrossSectionPerGas[83][I] = 0.0
        if EN > object.EnergyLevels[83]:
            object.InelasticCrossSectionPerGas[83][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB7, YVB7, XVB7, 1) * 100 * APOPGS
        # 8V1
        object.InelasticCrossSectionPerGas[84][I] = 0.0
        if EN > object.EnergyLevels[84]:
            object.InelasticCrossSectionPerGas[84][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB8, YVB8, XVB8, 1) * 100 * APOPGS
        # 9V1
        object.InelasticCrossSectionPerGas[85][I] = 0.0
        if EN > object.EnergyLevels[85]:
            object.InelasticCrossSectionPerGas[85][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB9, YVB9, XVB9, 1) * 100 * APOPGS
        # 10V1
        object.InelasticCrossSectionPerGas[86][I] = 0.0
        if EN > object.EnergyLevels[86]:
            object.InelasticCrossSectionPerGas[86][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB10, YVB10, XVB10, 1) * 100 * APOPGS
        # 11V1
        object.InelasticCrossSectionPerGas[87][I] = 0.0
        if EN > object.EnergyLevels[87]:
            object.InelasticCrossSectionPerGas[87][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB11, YVB11, XVB11, 1) * 100 * APOPGS
        # 12V1
        object.InelasticCrossSectionPerGas[88][I] = 0.0
        if EN > object.EnergyLevels[88]:
            object.InelasticCrossSectionPerGas[88][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB12, YVB12, XVB12, 1) * 100 * APOPGS
        # 13V1
        object.InelasticCrossSectionPerGas[89][I] = 0.0
        if EN > object.EnergyLevels[89]:
            object.InelasticCrossSectionPerGas[89][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB13, YVB13, XVB13, 1) * 100 * APOPGS
        # 14V1
        object.InelasticCrossSectionPerGas[90][I] = 0.0
        if EN > object.EnergyLevels[90]:
            object.InelasticCrossSectionPerGas[90][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB14, YVB14, XVB14, 1) * 100 * APOPGS
        # 15V1
        object.InelasticCrossSectionPerGas[91][I] = 0.0
        if EN > object.EnergyLevels[91]:
            object.InelasticCrossSectionPerGas[91][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB15, YVB15, XVB15, 1) * 100 * APOPGS

        # SET ROTATIONAL AND VIBRATIONAL ANGULAR DISTRIBUTIONS ( IF KIN NE 0 )
        for J in range(92):
            object.PEInelasticCrossSectionPerGas[J][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if EN > 3 * abs(object.EnergyLevels[J]):
                if object.WhichAngularModel > 0:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]

        # A3SIGMA (V = 0-4)
        object.InelasticCrossSectionPerGas[92][I] = 0.0
        object.PEInelasticCrossSectionPerGas[92][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[92][I] = 0.
        if EN > object.EnergyLevels[92]:
            object.InelasticCrossSectionPerGas[92][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP1, YTRP1, XTRP1, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP1, YTP1M, XTRP1, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[92]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[92][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[92][I] = object.PEElasticCrossSection[1][I - IOFFN[92]]

        # A3SIGMA (V = 5-9)
        object.InelasticCrossSectionPerGas[93][I] = 0.0
        object.PEInelasticCrossSectionPerGas[93][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[93][I] = 0.
        if EN > object.EnergyLevels[93]:
            object.InelasticCrossSectionPerGas[93][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP2, YTRP2, XTRP2, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP2, YTP2M, XTRP2, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[93]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[93][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[93][I] = object.PEElasticCrossSection[1][I - IOFFN[93]]

        # B3PI (V=0-3)
        object.InelasticCrossSectionPerGas[94][I] = 0.0
        object.PEInelasticCrossSectionPerGas[94][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[94][I] = 0.
        if EN > object.EnergyLevels[94]:
            object.InelasticCrossSectionPerGas[94][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP3, YTRP3, XTRP3, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP3, YTP3M, XTRP3, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[94]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[94][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[94][I] = object.PEElasticCrossSection[1][I - IOFFN[94]]

        # W3DELTA (V = 0-5)
        object.InelasticCrossSectionPerGas[95][I] = 0.0
        object.PEInelasticCrossSectionPerGas[95][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[95][I] = 0.
        if EN > object.EnergyLevels[95]:
            object.InelasticCrossSectionPerGas[95][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP4, YTRP4, XTRP4, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP4, YTP4M, XTRP4, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[95]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[95][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[95][I] = object.PEElasticCrossSection[1][I - IOFFN[95]]

        # A3SIGMA (V = 10-21)
        object.InelasticCrossSectionPerGas[96][I] = 0.0
        object.PEInelasticCrossSectionPerGas[96][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[96][I] = 0.
        if EN > object.EnergyLevels[96]:
            object.InelasticCrossSectionPerGas[96][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP5, YTRP5, XTRP5, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP5, YTP5M, XTRP5, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[96]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[96][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[96][I] = object.PEElasticCrossSection[1][I - IOFFN[96]]

        # B3PI (V=4-16)
        object.InelasticCrossSectionPerGas[97][I] = 0.0
        object.PEInelasticCrossSectionPerGas[97][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[97][I] = 0.
        if EN > object.EnergyLevels[97]:
            object.InelasticCrossSectionPerGas[97][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP6, YTRP6, XTRP6, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP6, YTP6M, XTRP6, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[97]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[97][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[97][I] = object.PEElasticCrossSection[1][I - IOFFN[97]]

        # W3DEL (V=6-10)
        object.InelasticCrossSectionPerGas[98][I] = 0.0
        object.PEInelasticCrossSectionPerGas[98][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[98][I] = 0.
        if EN > object.EnergyLevels[98]:
            object.InelasticCrossSectionPerGas[98][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP7, YTRP7, XTRP7, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP7, YTP7M, XTRP7, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[98]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[98][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[98][I] = object.PEElasticCrossSection[1][I - IOFFN[98]]

        # A1PI (V=0-3)
        object.InelasticCrossSectionPerGas[99][I] = 0.0
        object.PEInelasticCrossSectionPerGas[99][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[99][I] = 0.
        if EN > object.EnergyLevels[99]:
            object.InelasticCrossSectionPerGas[99][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG1, YSNG1, XSNG1, 1.5) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG1, YSG1M, XSNG1, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[99]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[99][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[99][I] = object.PEElasticCrossSection[1][I - IOFFN[99]]

        # B!3SIG (V=0-6)
        object.InelasticCrossSectionPerGas[100][I] = 0.0
        object.PEInelasticCrossSectionPerGas[100][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[100][I] = 0.
        if EN > object.EnergyLevels[100]:
            object.InelasticCrossSectionPerGas[100][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP8, YTRP8, XTRP8, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP8, YTP8M, XTRP8, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[100]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[100][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[100][I] = object.PEElasticCrossSection[1][I - IOFFN[100]]

        # A!SIG (V=0-6)
        object.InelasticCrossSectionPerGas[101][I] = 0.0
        object.PEInelasticCrossSectionPerGas[101][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[101][I] = 0.
        if EN > object.EnergyLevels[101]:
            object.InelasticCrossSectionPerGas[101][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG2, YSNG2, XSNG2, 1.5) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG2, YSG2M, XSNG2, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[101]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[101][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[101][I] = object.PEElasticCrossSection[1][I - IOFFN[101]]

        # W3DEL(V=11-19)
        object.InelasticCrossSectionPerGas[102][I] = 0.0
        object.PEInelasticCrossSectionPerGas[102][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[102][I] = 0.
        if EN > object.EnergyLevels[102]:
            object.InelasticCrossSectionPerGas[102][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP9, YTRP9, XTRP9, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP9, YTP9M, XTRP9, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[102]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[102][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[102][I] = object.PEElasticCrossSection[1][I - IOFFN[102]]

        # W1DEL (V=0-5)
        object.InelasticCrossSectionPerGas[103][I] = 0.0
        object.PEInelasticCrossSectionPerGas[103][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[103][I] = 0.
        if EN > object.EnergyLevels[103]:
            object.InelasticCrossSectionPerGas[103][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG3, YSNG3, XSNG3, 1.5) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG3, YSG3M, XSNG3, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[103]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[103][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[103][I] = object.PEElasticCrossSection[1][I - IOFFN[103]]

        # A1PI (V=4-15)
        object.InelasticCrossSectionPerGas[104][I] = 0.0
        object.PEInelasticCrossSectionPerGas[104][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[104][I] = 0.
        if EN > object.EnergyLevels[104]:
            object.InelasticCrossSectionPerGas[104][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG4, YSNG4, XSNG4, 1.5) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG4, YSG4M, XSNG4, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[104]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[104][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[104][I] = object.PEElasticCrossSection[1][I - IOFFN[104]]

        # B!3SIG (V = 7-18)
        object.InelasticCrossSectionPerGas[105][I] = 0.0
        object.PEInelasticCrossSectionPerGas[105][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[105][I] = 0.
        if EN > object.EnergyLevels[105]:
            object.InelasticCrossSectionPerGas[105][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP10, YTRP10, XTRP10, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP10, YTP10M, XTRP10, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[105]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[105][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[105][I] = object.PEElasticCrossSection[1][I - IOFFN[105]]

        # A!1SIG (V=7-19)
        object.InelasticCrossSectionPerGas[106][I] = 0.0
        object.PEInelasticCrossSectionPerGas[106][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[106][I] = 0.
        if EN > object.EnergyLevels[106]:
            object.InelasticCrossSectionPerGas[106][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG5, YSNG5, XSNG5, 1.5) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG5, YSG5M, XSNG5, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[106]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[106][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[106][I] = object.PEElasticCrossSection[1][I - IOFFN[106]]

        # W1DEL (V=6-18)
        object.InelasticCrossSectionPerGas[107][I] = 0.0
        object.PEInelasticCrossSectionPerGas[107][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[107][I] = 0.
        if EN > object.EnergyLevels[107]:
            object.InelasticCrossSectionPerGas[107][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG6, YSNG6, XSNG6, 1.5) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG6, YSG6M, XSNG6, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[107]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[107][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[107][I] = object.PEElasticCrossSection[1][I - IOFFN[107]]

        # C3PI (V=0-4)
        object.InelasticCrossSectionPerGas[108][I] = 0.0
        object.PEInelasticCrossSectionPerGas[108][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[108][I] = 0.
        if EN > object.EnergyLevels[108]:
            object.InelasticCrossSectionPerGas[108][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP11, YTRP11, XTRP11, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP11, YTP11M, XTRP11, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[108]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[108][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[108][I] = object.PEElasticCrossSection[1][I - IOFFN[108]]

        # E3SIG
        object.InelasticCrossSectionPerGas[109][I] = 0.0
        object.PEInelasticCrossSectionPerGas[109][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[109][I] = 0.
        if EN > object.EnergyLevels[109]:
            object.InelasticCrossSectionPerGas[109][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP12, YTRP12, XTRP12, 2) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP12, YTP12M, XTRP12, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[109]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[109][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[109][I] = object.PEElasticCrossSection[1][I - IOFFN[109]]

        # A!!1SIG (V=0-1)
        object.InelasticCrossSectionPerGas[110][I] = 0.0
        object.PEInelasticCrossSectionPerGas[110][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[110][I] = 0.
        if EN > object.EnergyLevels[110]:
            object.InelasticCrossSectionPerGas[110][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG7, YSNG7, XSNG7, 1.5) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG7, YSG7M, XSNG7, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[110]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[110][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[110][I] = object.PEElasticCrossSection[1][I - IOFFN[110]]

        # B1PI (V=0-6)   F=0.1855
        object.InelasticCrossSectionPerGas[111][I] = 0.0
        object.PEInelasticCrossSectionPerGas[111][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[111][I] = 0.
        if EN > object.EnergyLevels[111]:
            object.InelasticCrossSectionPerGas[111][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN,NSNG8, YSNG8, XSNG8, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                   BBCONST, object.EnergyLevels[111], object.E[2], <float>(0.1855))
            if EN <= XSNG8[NSNG8 - 1]:
                object.InelasticCrossSectionPerGas[111][I] *= 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG8, YSG8M, XSNG8, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[111]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[111][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[111][I] = object.PEElasticCrossSection[1][I - IOFFN[111]]

        # C!1SIG (V=0-3)   F=0.150
        object.InelasticCrossSectionPerGas[112][I] = 0.0
        object.PEInelasticCrossSectionPerGas[112][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[112][I] = 0.
        if EN > object.EnergyLevels[112]:
            object.InelasticCrossSectionPerGas[112][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN,EN, NSNG9, YSNG9, XSNG9, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                   BBCONST, object.EnergyLevels[112], object.E[2], <float>(0.150))
            if EN <= XSNG9[NSNG9 - 1]:
                object.InelasticCrossSectionPerGas[112][I] *= 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG9, YSG9M, XSNG9, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[112]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[112][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[112][I] = object.PEElasticCrossSection[1][I - IOFFN[112]]

        # G 3PI
        object.InelasticCrossSectionPerGas[113][I] = 0.0
        object.PEInelasticCrossSectionPerGas[113][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[113][I] = 0.
        if EN > object.EnergyLevels[113]:
            object.InelasticCrossSectionPerGas[113][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP13, YTRP13, XTRP13, 1.5) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP13, YTP13M, XTRP13, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[113]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[113][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[113][I] = object.PEElasticCrossSection[1][I - IOFFN[113]]

        # C3 1PI (V=0-3)   F=0.150
        object.InelasticCrossSectionPerGas[114][I] = 0.0
        object.PEInelasticCrossSectionPerGas[114][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[114][I] = 0.
        if EN > object.EnergyLevels[114]:
            object.InelasticCrossSectionPerGas[114][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN,EN, NSNG10, YSNG10, XSNG10, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                   BBCONST, object.EnergyLevels[114], object.E[2], <float>(0.150))
            if EN <= XSNG10[NSNG10 - 1]:
                object.InelasticCrossSectionPerGas[114][I] *= 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG10, YSG10M, XSNG10, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[114]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[114][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[114][I] = object.PEElasticCrossSection[1][I - IOFFN[114]]

        # F 3PI (V = 0-3)
        object.InelasticCrossSectionPerGas[115][I] = 0.0
        object.PEInelasticCrossSectionPerGas[115][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[115][I] = 0.
        if EN > object.EnergyLevels[115]:
            object.InelasticCrossSectionPerGas[115][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP14, YTRP14, XTRP14, 1.5) * 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP14, YTP14M, XTRP14, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[115]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[115][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[115][I] = object.PEElasticCrossSection[1][I - IOFFN[115]]

        # B1PI (V=7-14)   F=0.0663
        object.InelasticCrossSectionPerGas[116][I] = 0.0
        object.PEInelasticCrossSectionPerGas[116][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[116][I] = 0.
        if EN > object.EnergyLevels[116]:
            object.InelasticCrossSectionPerGas[116][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN,NSNG11, YSNG11, XSNG11, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                   BBCONST, object.EnergyLevels[116], object.E[2], <float>(0.0663))
            if EN <= XSNG11[NSNG11 - 1]:
                object.InelasticCrossSectionPerGas[116][I] *= 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG11, YSG11M, XSNG11, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[116]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[116][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[116][I] = object.PEElasticCrossSection[1][I - IOFFN[116]]

        # B! SIG (V=0-10)   F=0.0601
        object.InelasticCrossSectionPerGas[117][I] = 0.0
        object.PEInelasticCrossSectionPerGas[117][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[117][I] = 0.
        if EN > object.EnergyLevels[117]:
            object.InelasticCrossSectionPerGas[117][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN,NSNG12, YSNG12, XSNG12, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                   BBCONST, object.EnergyLevels[117], object.E[2], <float>(0.0601))
            if EN <= XSNG12[NSNG12 - 1]:
                object.InelasticCrossSectionPerGas[117][I] *= 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG12, YSG12M, XSNG12, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[117]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[117][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[117][I] = object.PEElasticCrossSection[1][I - IOFFN[117]]

        # O3 1PI (V=0-3)   F=0.0828
        object.InelasticCrossSectionPerGas[118][I] = 0.0
        object.PEInelasticCrossSectionPerGas[118][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[118][I] = 0.
        if EN > object.EnergyLevels[118]:
            object.InelasticCrossSectionPerGas[118][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN,NSNG13, YSNG13, XSNG13, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                   BBCONST, object.EnergyLevels[118], object.E[2], <float>(0.0828))
            if EN <= XSNG13[NSNG13 - 1]:
                object.InelasticCrossSectionPerGas[118][I] *= 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG13, YSG13M, XSNG13, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[118]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[118][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[118][I] = object.PEElasticCrossSection[1][I - IOFFN[118]]

        # C C!  1SIG (Sum V=4-6) (AVERGAE E=14.090)  F=0.139
        object.InelasticCrossSectionPerGas[119][I] = 0.0
        object.PEInelasticCrossSectionPerGas[119][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[119][I] = 0.
        if EN > object.EnergyLevels[119]:
            object.InelasticCrossSectionPerGas[119][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN,NSNG14, YSNG14, XSNG14, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                   BBCONST, object.EnergyLevels[119], object.E[2], <float>(0.1390))
            if EN <= XSNG14[NSNG14 - 1]:
                object.InelasticCrossSectionPerGas[119][I] *= 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG14, YSG14M, XSNG14, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[119]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[119][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[119][I] = object.PEElasticCrossSection[1][I - IOFFN[119]]

        # C C!  1SIG (Sum V=4-6) (AVERGAE E=14.090)  F=0.139
        object.InelasticCrossSectionPerGas[120][I] = 0.0
        object.PEInelasticCrossSectionPerGas[120][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[120][I] = 0.
        if EN > object.EnergyLevels[120]:
            object.InelasticCrossSectionPerGas[120][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN,NSNG15, YSNG15, XSNG15, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                   BBCONST, object.EnergyLevels[120], object.E[2], <float>(0.2650))
            if EN <= XSNG15[NSNG15 - 1]:
                object.InelasticCrossSectionPerGas[120][I] *= 100
            RAT = GasUtil.CALInelasticCrossSectionPerGasP(EN, NSNG15, YSG15M, XSNG15, 1) * 1e18
            if EN > 3.0 * object.EnergyLevels[120]:
                if object.WhichAngularModel == 1:
                    object.PEInelasticCrossSectionPerGas[120][I] = 1.5 - RAT
                if object.WhichAngularModel == 2:
                    object.PEInelasticCrossSectionPerGas[120][I] = object.PEElasticCrossSection[1][I - IOFFN[120]]

        # E! 1SIG  ELOSS = 14.36 F = 0.0108
        object.InelasticCrossSectionPerGas[121][I] = 0.0
        object.PEInelasticCrossSectionPerGas[121][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[121][I] = 0.0
        if EN > object.EnergyLevels[121]:
            object.InelasticCrossSectionPerGas[121][I] = <float>(0.0108) / (object.EnergyLevels[121] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[121])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[121] + object.E[2])
            if object.InelasticCrossSectionPerGas[121][I] < 0.0:
                object.InelasticCrossSectionPerGas[121][I] = 0.0
            object.PEInelasticCrossSectionPerGas[121][I] = object.PEInelasticCrossSectionPerGas[120][I]

        # E 1PI  ELOSS = 14.45 F = 0.0237
        object.InelasticCrossSectionPerGas[122][I] = 0.0
        object.PEInelasticCrossSectionPerGas[122][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[122][I] = 0.0
        if EN > object.EnergyLevels[122]:
            object.InelasticCrossSectionPerGas[122][I] = <float>(0.0237) / (object.EnergyLevels[122] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[122])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[122] + object.E[2])
            if object.InelasticCrossSectionPerGas[122][I] < 0.0:
                object.InelasticCrossSectionPerGas[122][I] = 0.0
            object.PEInelasticCrossSectionPerGas[122][I] = object.PEInelasticCrossSectionPerGas[120][I]

        # SINGLET  ELOSS = 14.839 F = 0.0117
        object.InelasticCrossSectionPerGas[123][I] = 0.0
        object.PEInelasticCrossSectionPerGas[123][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[123][I] = 0.0
        if EN > object.EnergyLevels[123]:
            object.InelasticCrossSectionPerGas[123][I] = <float>(0.0117) / (object.EnergyLevels[123] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[123])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[123] + object.E[2])
            if object.InelasticCrossSectionPerGas[123][I] < 0.0:
                object.InelasticCrossSectionPerGas[123][I] = 0.0
            object.PEInelasticCrossSectionPerGas[123][I] = object.PEInelasticCrossSectionPerGas[120][I]

        # Sum  OF HIGH ENERGY SINGLETS ELOSS 15.20EV F = 0.1152
        object.InelasticCrossSectionPerGas[124][I] = 0.0
        object.PEInelasticCrossSectionPerGas[124][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[124][I] = 0.0
        if EN > object.EnergyLevels[124]:
            object.InelasticCrossSectionPerGas[124][I] = <float>(0.1152) / (object.EnergyLevels[124] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[124])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[124] + object.E[2])
            if object.InelasticCrossSectionPerGas[124][I] < 0.0:
                object.InelasticCrossSectionPerGas[124][I] = 0.0
            object.PEInelasticCrossSectionPerGas[124][I] = object.PEInelasticCrossSectionPerGas[120][I]

        # Sum NEUTRAL BREAKUP ABOVE IONISATION ENERGY  F=0.160
        object.InelasticCrossSectionPerGas[125][I] = 0.0
        object.PEInelasticCrossSectionPerGas[125][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[125][I] = 0.0
        if EN > object.EnergyLevels[125]:
            object.InelasticCrossSectionPerGas[125][I] = <float>(0.1600) / (object.EnergyLevels[125] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[125])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * (EN + 2 * object.EnergyLevels[125]) / EN
            if object.InelasticCrossSectionPerGas[125][I] < 0.0:
                object.InelasticCrossSectionPerGas[125][I] = 0.0
            object.PEInelasticCrossSectionPerGas[125][I] = object.PEInelasticCrossSectionPerGas[120][I]

        # Sum NEUTRAL BREAKUP ABOVE IONISATION ENERGY  F=0.160
        object.InelasticCrossSectionPerGas[126][I] = 0.0
        object.PEInelasticCrossSectionPerGas[126][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[126][I] = 0.0
        if EN > object.EnergyLevels[126]:
            object.InelasticCrossSectionPerGas[126][I] = <float>(0.090) / (object.EnergyLevels[126] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[126])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * (EN + 2 * object.EnergyLevels[126]) / EN
            if object.InelasticCrossSectionPerGas[126][I] < 0.0:
                object.InelasticCrossSectionPerGas[126][I] = 0.0
            object.PEInelasticCrossSectionPerGas[126][I] = object.PEInelasticCrossSectionPerGas[120][I]

        #LOAD BREMSSTRAHLUNG X-SECTIONS
        object.InelasticCrossSectionPerGas[127][I] = 0.0
        if EN > 1000:
            object.InelasticCrossSectionPerGas[127][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z7T, EBRM) * 2e-8

        # ROTATIONAL Sum
        SumR = 0.0
        for J in range(76):
            SumR += object.InelasticCrossSectionPerGas[J][I]

        # VIBRATIONAL Sum
        SumV = 0.0
        for J in range(76, 92):
            SumV += object.InelasticCrossSectionPerGas[J][I]

        #EXCITATION Sum
        SumEX = 0.0
        for J in range(92, 111):
            SumEX += object.InelasticCrossSectionPerGas[J][I]

        #EXCITATION Sum
        SumEX1 = 0.0
        for J in range(111, 127):
            SumEX1 += object.InelasticCrossSectionPerGas[J][I]
        # GET CORRECT ELASTIC XSECTION BY SUBTRACTION OF ROTATION
        object.Q[1][I] -= SumR

        if object.Q[1][I] < 0.0:
            # FOR VERY HIGH TEMPERATURES SOMETIMES SumR BECOMES LARGER THAN
            # THE ELASTIC+ROT (ONLY IN FIRST TWO ENERGY BINS) FIX GT 0
            object.Q[1][I] = 0.95e-16

        object.Q[0][I] = object.Q[1][I] + object.Q[4][I] + object.IonizationCrossSection[1][I] + SumR + SumV + SumEX + SumEX1

    for I in range(1,128):
        J = 128 - I - 1
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
    if object.N_Inelastic < 77:
        object.N_Inelastic = 77
    return
