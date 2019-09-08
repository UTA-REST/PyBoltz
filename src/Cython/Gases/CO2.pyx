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
    """
    This function is used to calculate the needed momentum cross sections for CO2 gas.
    """
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
    #         BRION GROUP AND SHAW ET AL . OSCILLATOR Sum S(0)=21.9856
    #         S(-1)i=5.372
    #      2) USED STRAUB DATA FOR DISSOCIATIVE IONISATION ABOVE 30EV
    #         AND   RAP AND ENGLADER-GOLDEN  AT LOW ENERGY
    #      3) IONISATION-EXCITATION FROM ITIKAWA REVIEW
    #
    # ANGULAR DISTRIBUTION ONLY ALLOWED FOR ELASTIC , IONISATION AND
    # EXCITATION ABOVE 10EV.
    #
    # ---------------------------------------------------------------------

    cdef double A0, RY, CONST, ElectronMass2, API, BBCONST, AM2, C, EOBFAC, AUGKC, AUGK0
    cdef int NBREM

    # BORN-BETHE CONSTANTS
    A0 = 0.52917720859e-08
    RY = <float> (13.60569193)
    CONST = 1.873884e-20
    ElectronMass2 = <float> (1021997.804)
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / ElectronMass2
    # BORN BETHE VALUES FOR IONISATION
    AM2 = <float> (5.38)
    EOBFAC = <float> (0.56)
    C = <float> (57.0)

    # AVERAGE SUGER EMISSIONS FROM EACH SHELL
    AUGKC = 2.0
    AUGK0 = <float> (1.99)

    object.N_Ionization = 11
    object.N_Attachment = 1
    object.N_Inelastic = 144
    object.N_Null = 0

    NBREM = 25

    for J in range(6):
        object.KEL[J] = object.WhichAngularModel
    for J in range(object.N_Inelastic):
        object.KIN[J] = object.WhichAngularModel

    cdef int NEL, NV2, N2V2, NV1, N3V2, NV3, NPD3, NV130, NPD4, NPD5, NPD6, NPD7, NPD8, NPD9, NPDH, N_Attachment1, NTRP1, NTRP2, N_Ionization1, N_Ionization2, N_Ionization3
    cdef int N_Ionization4, N_Ionization5, N_Ionization6, N_Ionization7, N_Ionization8, N_Ionization9, NKSHC, NKSHO

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
    N_Attachment1 = 68
    NTRP1 = 11
    NTRP2 = 11
    N_Ionization1 = 63
    N_Ionization2 = 66
    N_Ionization3 = 66
    N_Ionization4 = 41
    N_Ionization5 = 41
    N_Ionization6 = 40
    N_Ionization7 = 37
    N_Ionization8 = 30
    N_Ionization9 = 27
    NKSHC = 83
    NKSHO = 81
    cdef double ElectronMass = 9.10938291e-31, PENSum
    cdef double AMU = 1.660538921e-27, EOBY[11]

    object.E = [0.0, 1.0, <float> (13.776), 0.0, 0.0, 0.0]
    object.E[1] = <float> (2.0) * ElectronMass / (<float> (44.0095) * AMU)

    object.IonizationEnergy[0:11] = [ < float > (13.776), <float> (17.314), <float> (18.077), <float> (19.07), <float> (19.47),
                         <float> (27.82), <float> (37.4), <float> (72.0), <float> (74.0), <float> (285.0),
                         <float> (532.0)]

    # OPAL AND BEATY
    for J in range(11):
        EOBY[J] = <float> (13.8)

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
    object.WK[9] = <float> (0.0026)
    object.EFL[9] = 273.0
    object.NG1[9] = 1
    object.EG1[9] = 253.0
    object.NG2[9] = 2
    object.EG2[9] = 5.0
    object.NC0[10] = 3
    object.EC0[10] = 485.0
    object.WK[10] = <float> (0.0069)
    object.EFL[10] = 518.0
    object.NG1[10] = 1
    object.EG1[10] = 480
    object.NG2[10] = 2
    object.EG2[10] = 5.0

    cdef int L
    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break
    cdef double AMPV2, AMPV3, B0, QBQA, QBK, PJ[220], Sum = 0.0
    # DIPOLE TRANSITION STRENGTH FOR VIBRATIONS V010 AND V001
    AMPV2 = <float> (0.1703)
    AMPV3 = <float> (0.3824)
    #-----------------------------------------------------------------------
    #  B0 IS ROTATIONAL CONSTANT
    #  QBQA IS QUADRUPOLE MOMENT
    B0 = 4.838e-5
    A0 = 0.5291772083e-8
    QBQA = <float> (3.24)
    QBK = <float> (1.67552) * (QBQA * A0) ** 2

    #CALC FRACTIONAL POPULATION DENSITY OF ROTATIONAL STATES
    PJ[0] = 1.0
    for J in range(2, 32):
        i = (2 * J) - 2
        PJ[J - 1] = (2 * i + 1) * exp(-1 * i * (i + 1) * B0 / object.ThermalEnergy)
    for J in range(31):
        Sum += PJ[J]
    for J in range(31):
        PJ[J] = PJ[J] / Sum
    object.EnergyLevels = gd['gas12/EnergyLevels']

    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION
    for NL in range(object.N_Inelastic):
        for i in range(4000):
            if object.EG[i] > object.EnergyLevels[NL]:
                IOFFN[NL] = i
                break

    # ENTER PENN_InelasticG TRANSFER FRACTION FOR EACH LEVEL
    # FIRST 81 LEVELS UNLIKELY TO HAVE ENOUGH ENERGY
    for I in range(3):
        for J in range(81):
            object.PenningFraction[I][J] = 0.0

    # PENN_InelasticG TRANSFER FRACTION FOR LEVELS
    for J in range(81, object.N_Inelastic):
        object.PenningFraction[0][J] = 0.0
        #PENN_InelasticG TRANSFER DISTANCE IN MICRONS
        object.PenningFraction[1][J] = 1.0
        #PENN_InelasticG TRANSFER TIME IN PICOSECONDS
        object.PenningFraction[2][J] = 1.0

    for J in range(NBREM):
        EBRM[J] = exp(EBRM[J])

    cdef double DEGV1, DEGV2, DEGV3, DEG2V2, DEG3V2, APOPV2, APOP2V2, APOPV1, APOP3V2, APOPV3, APOPGS, APBEND, AEXT20, AGST20
    DEGV1 = 1.0
    DEGV2 = 2.0
    DEGV3 = 1.0
    DEG2V2 = 3.0
    # 3V2 === Sum (3V2 + V12) =   4+2
    DEG3V2 = 6.0
    #----------------------------------------------------
    # CALC POPULATION OF VIBRATIONAL STATES
    Sum = 0.0
    APOPV2 = DEGV2 * exp(object.EnergyLevels[60] / object.ThermalEnergy)
    APOP2V2 = DEG2V2 * exp(object.EnergyLevels[62] / object.ThermalEnergy)
    APOPV1 = DEGV1 * exp(object.EnergyLevels[64] / object.ThermalEnergy)
    APOP3V2 = DEG3V2 * exp(object.EnergyLevels[66] / object.ThermalEnergy)
    APOPV3 = DEGV3 * exp(object.EnergyLevels[68] / object.ThermalEnergy)
    Sum = 1.0 + APOPV2 + APOP2V2 + APOPV1 + APOP3V2 + APOPV3
    APOPGS = 1.0 / Sum
    APOPV2 = APOPV2 / Sum
    APOP2V2 = APOP2V2 / Sum
    APOPV1 = APOPV1 / Sum
    APOP3V2 = APOP3V2 / Sum
    APOPV3 = APOPV3 / Sum
    APBEND = APOPV2 + APOP2V2 + APOP3V2

    # RENORMALISE VIBRATIONAL GROUND STATE POPULATION IN ORDER TO ACCOUNT
    # FOR EXCITATION FROM VIBRATIONALLY EXCITED STATES
    APOPGS = 1.0
    # BEND MODE AND EFFECTIVE GROUND STATE POPULATION AT 293.15 KELVIN
    AEXT20 = 7.51373753e-2
    AGST20 = 1.0 - AEXT20
    cdef double EN, GAMMA1, GAMMA2, BETA, BETA2, QMT, ElasticCrossSection, PQ[3], X1, X2, QBB = 0.0, QMOM, ElasticCrossSectionA, QBMOM, QBELA, F[63], CONS[63]
    cdef double SumR, SumV, SumE, SumTRP, SumEXC, SumION
    F = [<float> (0.0000698), <float> (0.0000630), <float> (0.0000758), <float> (0.0001638), <float> (0.0003356),
         <float> (0.0007378), <float> (0.001145), <float> (0.001409), <float> (0.001481), <float> (0.000859),
         <float> (0.001687), <float> (0.002115), <float> (0.001920), <float> (0.001180), <float> (0.000683),
         <float> (0.000456), <float> (0.004361), <float> (0.1718), <float> (0.06242), <float> (0.01852),
         <float> (0.01125), <float> (0.01535), <float> (0.01009), <float> (0.01940), <float> (0.03817),
         <float> (0.05814), <float> (0.04769), <float> (0.09315), <float> (0.06305), <float> (0.02477),
         <float> (0.06231), <float> (0.06696), <float> (0.09451), <float> (0.04986), <float> (0.09029),
         <float> (0.07431), <float> (0.15625), <float> (0.08084), <float> (0.02662), <float> (0.01062),
         <float> (0.00644), <float> (0.00485), <float> (0.00880), <float> (0.01522), <float> (0.01683),
         <float> (0.02135), <float> (0.03232), <float> (0.02534), <float> (0.01433), <float> (0.00965),
         <float> (0.01481), <float> (0.01148), <float> (0.00885), <float> (0.00931), <float> (0.00666),
         <float> (0.00443), <float> (0.00371), <float> (0.00344), <float> (0.00356), <float> (0.00530),
         <float> (0.00621), <float> (0.00619), <float>(3.6)]

    CONS = [<float> (1.0192), <float> (1.0185), <float> (1.0179), <float> (1.0172), <float> (1.0167), <float> (1.0161),
            <float> (1.0156), <float> (1.0152), <float> (1.0147), <float> (1.0143), <float> (1.014), <float> (1.0137),
            <float> (1.0133),  <float> (1.0130), <float> (1.0126), <float> (1.0123), <float> (1),
            <float> (1), <float> (1), <float> (1), <float> (1), <float> (1), <float> (1), <float> (1), <float> (1),
            <float> (1), <float> (1), <float> (1), <float> (1), <float> (1), <float> (1), <float> (1), <float> (1),
            <float> (1), <float> (1), <float> (1), <float> (1), <float> (1.0075), <float> (1.0089),
            <float> (1.0088), <float> (1.0086), <float> (1.0085), <float> (1.0083), <float> (1.0082),
            <float> (1.0081), <float> (1.0079), <float> (1.0078), <float> (1.0077), <float> (1.0076), <float> (1.0075),
            <float> (1.0074), <float> (1.0072), <float> (1.0071),  <float> (1.0070), <float> (1.0069),
            <float> (1.0068), <float> (1.0068), <float> (1.0067), <float> (1.0066), <float> (1.0065), <float> (1.0064),
            <float> (1.0070), <float>(1)]
    cdef int FI = 0, CONI = 0

    for I in range(4000):
        EN = object.EG[I]
        ENLG = log(EN)
        GAMMA1 = (ElectronMass2 + 2 * EN) / ElectronMass2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA
        # ELASTIC USE LOG INTERPOLATION
        if EN <= XEN[0]:
            QMOM = YMOM[0] * 1e-16
            ElasticCrossSectionA = YEL[0] * 1e-16
            QBMOM = QMOM
            QBELA = ElasticCrossSectionA
            PQ[2] = 0.0
        else:
            QMOM = GasUtil.QLSCALE(EN, NEL, YMOM, XEN)
            ElasticCrossSectionA = GasUtil.QLSCALE(EN, NEL, YEL, XEN)
            QBMOM = GasUtil.QLSCALE(EN, NEL, YVBMOM, XEN)
            QBELA = GasUtil.QLSCALE(EN, NEL, YVBEL, XEN)
            PQ[2] = GasUtil.QLSCALE(EN, NEL, YEPS, XEN) * 1.0e16
            PQ[2] = 1 - PQ[2]

        # CALC CHANGE IN ELASTIC CROSS SECTION DUE TO CHANGE IN ELASTIC
        # SCATTERING FROM BEND MODES ( CHANGE RELATIVE TO X-SECTION AT 293.15K)
        # BEND MODE POPULATION AT 293.15K == AEXT20,GROUND STATE POP. == AGST20

        QMOM = (1.0 - APBEND) * (QMOM - AEXT20 * QBMOM) / AGST20 + APBEND * QBMOM
        ElasticCrossSectionA = (1.0 - APBEND) * (ElasticCrossSectionA - AEXT20 * QBELA) / AGST20 + APBEND * QBELA
        PQ[1] = 0.5 + (ElasticCrossSectionA - QMOM) / (ElasticCrossSectionA)
        if object.WhichAngularModel == 2:
            object.Q[1][I] = ElasticCrossSectionA
            object.PEElasticCrossSection[1][I] = PQ[2]
            if EN < 10:
                object.PEElasticCrossSection[1][I] = 0.0
                object.Q[1][I] = QMOM

        if object.WhichAngularModel == 1:
            object.Q[1][I] = ElasticCrossSectionA
            object.PEElasticCrossSection[1][I] = PQ[1]
            if EN < 10:
                object.PEElasticCrossSection[1][I] = 0.5
                object.Q[1][I] = QMOM
        if object.WhichAngularModel == 0:
            object.PEElasticCrossSection[1][I] = 0.5
            object.Q[1][I] = QMOM
        for J in range(11):
            object.IonizationCrossSection[J][I] = 0.0
            object.PEIonizationCrossSection[J][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEIonizationCrossSection[J][I] = 0.0
        # IONISATION CO2+
        if EN > object.IonizationEnergy[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization1, YION1, XION1, BETA2, <float> (0.67716), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[0]:
                object.PEIonizationCrossSection[0][I] = object.PEElasticCrossSection[1][I - IOFFION[0]]

        # IONISATION CO2+ (A2PIu)
        if EN > object.IonizationEnergy[1]:
            object.IonizationCrossSection[1][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization2, YION2, XION2, BETA2, <float> (0.67716) * <float> (0.385),
                                                 CONST, object.DEN[I],
                                                 C, AM2)
            if EN > 2 * object.IonizationEnergy[1]:
                object.PEIonizationCrossSection[1][I] = object.PEElasticCrossSection[1][I - IOFFION[1]]

        # IONISATION CO2+ (B2SIGMA+u)
        if EN > object.IonizationEnergy[2]:
            object.IonizationCrossSection[2][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization3, YION3, XION3, BETA2, <float> (0.67716) * <float> (0.220),
                                                 CONST, object.DEN[I],
                                                 C, AM2)
            if EN > 2 * object.IonizationEnergy[2]:
                object.PEIonizationCrossSection[2][I] = object.PEElasticCrossSection[1][I - IOFFION[2]]

        # DISSOCIATIVE IONISATION O+
        if EN > object.IonizationEnergy[3]:
            object.IonizationCrossSection[3][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization4, YION4, XION4, BETA2, <float> (0.16156), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[3]:
                object.PEIonizationCrossSection[3][I] = object.PEElasticCrossSection[1][I - IOFFION[3]]

        # DISSOCIATIVE IONISATION CO+
        if EN > object.IonizationEnergy[4]:
            object.IonizationCrossSection[4][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization5, YION5, XION5, BETA2, <float> (0.07962), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[4]:
                object.PEIonizationCrossSection[4][I] = object.PEElasticCrossSection[1][I - IOFFION[4]]

        # DISSOCIATIVE IONISATION C+
        if EN > object.IonizationEnergy[5]:
            object.IonizationCrossSection[5][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization6, YION6, XION6, BETA2, <float> (0.07452), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[5]:
                object.PEIonizationCrossSection[5][I] = object.PEElasticCrossSection[1][I - IOFFION[5]]

        # IONISATION CO2++
        if EN > object.IonizationEnergy[6]:
            object.IonizationCrossSection[6][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization7, YION7, XION7, BETA2, <float> (0.00559), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[6]:
                object.PEIonizationCrossSection[6][I] = object.PEElasticCrossSection[1][I - IOFFION[6]]

        # DISSOCIATIVE IONISATION C++
        if EN > object.IonizationEnergy[7]:
            object.IonizationCrossSection[7][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization8, YION8, XION8, BETA2, <float> (0.00076), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[7]:
                object.PEIonizationCrossSection[7][I] = object.PEElasticCrossSection[1][I - IOFFION[7]]

        # DISSOCIATIVE IONISATION O++
        if EN > object.IonizationEnergy[8]:
            object.IonizationCrossSection[8][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization9, YION9, XION9, BETA2, <float> (0.00080), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[8]:
                object.PEIonizationCrossSection[8][I] = object.PEElasticCrossSection[1][I - IOFFION[8]]

        # CARBON K-SHELL IONISATION
        if EN > object.IonizationEnergy[9]:
            object.IonizationCrossSection[9][I] = GasUtil.CALIonizationCrossSectionREG(EN, NKSHC, YKSHC, XKSHC)
            if EN > 2 * object.IonizationEnergy[9]:
                object.PEIonizationCrossSection[9][I] = object.PEElasticCrossSection[1][I - IOFFION[9]]

        #OXYGEN K-SHELL IONISATION
        if EN > object.IonizationEnergy[10]:
            object.IonizationCrossSection[10][I] = GasUtil.CALIonizationCrossSectionREG(EN, NKSHO, YKSHO, XKSHO) * 2
            if EN > 2 * object.IonizationEnergy[10]:
                object.PEIonizationCrossSection[10][I] = object.PEElasticCrossSection[1][I - IOFFION[10]]

        #FIX CO2+ X-SECTION FOR SPLIT INTO CO2+ EXCITED STATES
        object.IonizationCrossSection[0][I] -= object.IonizationCrossSection[1][I] + object.IonizationCrossSection[2][I]

        # ATTACHMENT

        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = 0.0

        if EN > XATT[0]:
            object.Q[3][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N_Attachment1, YATT, XATT, 3) * 100

        object.AttachmentCrossSection[0][I] = object.Q[3][I]
        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        # ----------------------------------------------------------------------
        #  QUADRUPOLE BORN ROTATIONAL STATES (GERJUOY AND STEnergyLevels)
        # ----------------------------------------------------------------------
        # SUPERELASTIC ROTATION

        for J in range(2, 61, 2):
            L = (J / 2)
            object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.0
            if EN >= 4 * abs(object.EnergyLevels[J - 1]):
                if object.WhichAngularModel == 0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
            object.InelasticCrossSectionPerGas[J - 1][I] = PJ[L] * QBK * sqrt(1.0 - object.EnergyLevels[J - 1] / EN) * J * (J - 1) / (
                    (2 * J + 1.0) * (2 * J - 1.0))

        # ROTATION
        for J in range(1, 61, 2):
            object.InelasticCrossSectionPerGas[J - 1][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J - 1][I] = 0.0
            if EN > object.EnergyLevels[J - 1]:
                L = (J + 1) / 2
                object.InelasticCrossSectionPerGas[J - 1][I] = PJ[L - 1] * QBK * sqrt(1.0 - object.EnergyLevels[J - 1] / EN) * ((J - 1) + 2.0) * (
                        (J - 1) + 1.0) / ((2 * (J - 1) + 3.0) * (2 * (J - 1) + 1.0))
            if EN >= 4.0 * abs(object.EnergyLevels[J - 1]):
                if object.WhichAngularModel > 0:
                    object.PEInelasticCrossSectionPerGas[J - 1][I] = object.PEElasticCrossSection[1][I - IOFFN[J - 1]]
        # BORN (1/E) FALL OFF IN ROTATONAL X-SEC ABOVE 6.0 EV .
        if EN >= 6.0:
            for J in range(60):
                object.InelasticCrossSectionPerGas[J][I] *= (6.0 / EN)

        # SUPERELASTIC V2 BEND MODE
        object.InelasticCrossSectionPerGas[60][I] = 0.0
        object.PEInelasticCrossSectionPerGas[60][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[60][I] = 0.0
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[60][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, NV2, YV2, XV2, APOPV2, object.EnergyLevels[61], DEGV2, object.EnergyLevels[60],
                                                     AMPV2, 0)
        if EN > 3 * abs(object.EnergyLevels[60]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[60][I] = object.PEElasticCrossSection[1][I - IOFFN[60]]

        # V2 BEND MODE
        object.InelasticCrossSectionPerGas[61][I] = 0.0
        object.PEInelasticCrossSectionPerGas[61][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[61][I] = 0.0
        if EN > object.EnergyLevels[61]:
            object.InelasticCrossSectionPerGas[61][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, NV2, YV2, XV2, APOPGS, 0, 1, object.EnergyLevels[61], AMPV2, 1)
        if EN > 3 * abs(object.EnergyLevels[61]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[61][I] = object.PEElasticCrossSection[1][I - IOFFN[61]]

        # SUPERELASTIC 2V2 BEND MODE HARMONIC
        object.InelasticCrossSectionPerGas[62][I] = 0.0
        object.PEInelasticCrossSectionPerGas[62][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[62][I] = 0.0
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[62][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, N2V2, Y2V2, X2V2, APOP2V2, object.EnergyLevels[63], DEG2V2,
                                                     object.EnergyLevels[60], 0, 0)
        if EN > 3 * abs(object.EnergyLevels[62]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[62][I] = object.PEElasticCrossSection[1][I - IOFFN[62]]

        # 2V2 BEND MODE HARMONIC
        object.InelasticCrossSectionPerGas[63][I] = 0.0
        object.PEInelasticCrossSectionPerGas[63][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[63][I] = 0.0
        if EN > object.EnergyLevels[63]:
            object.InelasticCrossSectionPerGas[63][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, N2V2, Y2V2, X2V2, APOPGS, 0, 1, object.EnergyLevels[63], 0, 1)
        if EN > 3 * abs(object.EnergyLevels[63]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[63][I] = object.PEElasticCrossSection[1][I - IOFFN[63]]

        # SUPERELASTIC V1 SYMMETRIC STRETCH
        object.InelasticCrossSectionPerGas[64][I] = 0.0
        object.PEInelasticCrossSectionPerGas[64][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[64][I] = 0.0
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[64][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, NV1, YV1, XV1, APOPV1, object.EnergyLevels[65], DEGV1, EN/2,
                                                     0, 1)
            # check if nan
            if object.InelasticCrossSectionPerGas[64][I] != object.InelasticCrossSectionPerGas[64][I]:
                object.InelasticCrossSectionPerGas[64][I] = 0.0
        if EN > 3 * abs(object.EnergyLevels[64]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[64][I] = object.PEElasticCrossSection[1][I - IOFFN[64]]

        # V1 SYMMETRIC STRETCH
        object.InelasticCrossSectionPerGas[65][I] = 0.0
        object.PEInelasticCrossSectionPerGas[65][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[65][I] = 0.0
        if EN > object.EnergyLevels[65]:
            object.InelasticCrossSectionPerGas[65][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, NV1, YV1, XV1, APOPGS, 0, 1, object.EnergyLevels[63], 0, 1)
        if EN > 3 * abs(object.EnergyLevels[65]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[65][I] = object.PEElasticCrossSection[1][I - IOFFN[65]]

        # SUPERELASTIC 3V2 + V12
        object.InelasticCrossSectionPerGas[66][I] = 0.0
        object.PEInelasticCrossSectionPerGas[66][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[66][I] = 0.0
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[66][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, N3V2, Y3V2, X3V2, APOP3V2, object.EnergyLevels[67], DEG3V2,
                                                     EN/2, 0, 1)
            # check if nan
            if object.InelasticCrossSectionPerGas[66][I] != object.InelasticCrossSectionPerGas[66][I]:
                object.InelasticCrossSectionPerGas[66][I] = 0.0
        if EN > 3 * abs(object.EnergyLevels[67]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[66][I] = object.PEElasticCrossSection[1][I - IOFFN[66]]

        # 3V2 + V12
        object.InelasticCrossSectionPerGas[67][I] = 0.0
        object.PEInelasticCrossSectionPerGas[67][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[67][I] = 0.0
        if EN > object.EnergyLevels[67]:
            object.InelasticCrossSectionPerGas[67][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, N3V2, Y3V2, X3V2, APOPGS, 0, 1, object.EnergyLevels[63], 0, 1)
        if EN > 3 * abs(object.EnergyLevels[67]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[67][I] = object.PEElasticCrossSection[1][I - IOFFN[67]]

        # SUPERELASTIC V3 ASYMMETRIC STRETCH
        object.InelasticCrossSectionPerGas[68][I] = 0.0
        object.PEInelasticCrossSectionPerGas[68][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[68][I] = 0.0
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[68][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, NV3, YV3, XV3, APOPV3, object.EnergyLevels[69], DEGV3, object.EnergyLevels[68],
                                                     AMPV3, 0)
        if EN > 3 * abs(object.EnergyLevels[68]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[68][I] = object.PEElasticCrossSection[1][I - IOFFN[68]]

        # V3  ASYMMETRIC STRETCH
        object.InelasticCrossSectionPerGas[69][I] = 0.0
        object.PEInelasticCrossSectionPerGas[69][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[69][I] = 0.0
        if EN > object.EnergyLevels[69]:
            object.InelasticCrossSectionPerGas[69][I] = GasUtil.CALInelasticCrossSectionPerGasVISELA(EN, NV3, YV3, XV3, APOPGS, 0, 1, object.EnergyLevels[69], AMPV3, 1)
        if EN > 3 * abs(object.EnergyLevels[69]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[69][I] = object.PEElasticCrossSection[1][I - IOFFN[69]]

        # 4V2 + 2V1 + V12V2 POLYAD 3
        object.InelasticCrossSectionPerGas[70][I] = 0.0
        object.PEInelasticCrossSectionPerGas[70][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[70][I] = 0.0
        if EN > object.EnergyLevels[70]:
            object.InelasticCrossSectionPerGas[70][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NPD3, YVPD3, XVPD3, 1) * 100
        if EN > 3 * abs(object.EnergyLevels[70]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[70][I] = object.PEElasticCrossSection[1][I - IOFFN[70]]

        # 3V2V1 + 2V1V2
        object.InelasticCrossSectionPerGas[71][I] = 0.0
        object.PEInelasticCrossSectionPerGas[71][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[71][I] = 0.0
        if EN > object.EnergyLevels[71]:
            object.InelasticCrossSectionPerGas[71][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NV130, YV130, XV130, 1) * 100
        if EN > 3 * abs(object.EnergyLevels[71]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[71][I] = object.PEElasticCrossSection[1][I - IOFFN[71]]

        # POLYAD 4
        object.InelasticCrossSectionPerGas[72][I] = 0.0
        object.PEInelasticCrossSectionPerGas[72][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[72][I] = 0.0
        if EN > object.EnergyLevels[72]:
            object.InelasticCrossSectionPerGas[72][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NPD4, YVPD4, XVPD4, 1) * 100
        if EN > 3 * abs(object.EnergyLevels[72]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[72][I] = object.PEElasticCrossSection[1][I - IOFFN[72]]

        # POLYAD 5
        object.InelasticCrossSectionPerGas[73][I] = 0.0
        object.PEInelasticCrossSectionPerGas[73][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[73][I] = 0.0
        if EN > object.EnergyLevels[73]:
            object.InelasticCrossSectionPerGas[73][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NPD5, YVPD5, XVPD5, 1) * 100
        if EN > 3 * abs(object.EnergyLevels[73]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[73][I] = object.PEElasticCrossSection[1][I - IOFFN[73]]

        # POLYAD 6
        object.InelasticCrossSectionPerGas[74][I] = 0.0
        object.PEInelasticCrossSectionPerGas[74][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[74][I] = 0.0
        if EN > object.EnergyLevels[74]:
            object.InelasticCrossSectionPerGas[74][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NPD6, YVPD6, XVPD6, 1) * 100
        if EN > 3 * abs(object.EnergyLevels[74]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[74][I] = object.PEElasticCrossSection[1][I - IOFFN[74]]

        # POLYAD 7
        object.InelasticCrossSectionPerGas[75][I] = 0.0
        object.PEInelasticCrossSectionPerGas[75][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[75][I] = 0.0
        if EN > object.EnergyLevels[75]:
            object.InelasticCrossSectionPerGas[75][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NPD7, YVPD7, XVPD7, 1) * 100
        if EN > 3 * abs(object.EnergyLevels[75]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[75][I] = object.PEElasticCrossSection[1][I - IOFFN[75]]

        # POLYAD 8
        object.InelasticCrossSectionPerGas[76][I] = 0.0
        object.PEInelasticCrossSectionPerGas[76][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[76][I] = 0.0
        if EN > object.EnergyLevels[76]:
            object.InelasticCrossSectionPerGas[76][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NPD8, YVPD8, XVPD8, 1) * 100
        if EN > 3 * abs(object.EnergyLevels[76]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[76][I] = object.PEElasticCrossSection[1][I - IOFFN[76]]

        # POLYAD 9
        object.InelasticCrossSectionPerGas[77][I] = 0.0
        object.PEInelasticCrossSectionPerGas[77][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[77][I] = 0.0
        if EN > object.EnergyLevels[77]:
            object.InelasticCrossSectionPerGas[77][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NPD9, YVPD9, XVPD9, 1) * 100
        if EN > 3 * abs(object.EnergyLevels[77]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[77][I] = object.PEElasticCrossSection[1][I - IOFFN[77]]

        # Sum OF HIGHER POLYADS
        object.InelasticCrossSectionPerGas[78][I] = 0.0
        object.PEInelasticCrossSectionPerGas[78][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[78][I] = 0.0
        if EN > object.EnergyLevels[78]:
            object.InelasticCrossSectionPerGas[78][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NPDH, YVPDH, XVPDH, 1) * 100
        if EN > 3 * abs(object.EnergyLevels[78]):
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[78][I] = object.PEElasticCrossSection[1][I - IOFFN[78]]

        FI = 0
        CONI = 0
        for J in range(79, 89):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2]) * CONS[CONI]
                if object.InelasticCrossSectionPerGas[J][I]< 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    if object.WhichAngularModel > 0:
                        object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        # TRIPLET
        object.InelasticCrossSectionPerGas[89][I] = 0.0
        object.PEInelasticCrossSectionPerGas[89][I] = 0.0
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[89][I] = 0.0
        if EN > object.EnergyLevels[89]:
            object.InelasticCrossSectionPerGas[89][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP1, YTRP1, XTRP1, 2) * 100
            if EN > 2.0 * object.EnergyLevels[89]:
                if object.WhichAngularModel > 0:
                    object.PEInelasticCrossSectionPerGas[89][I] = object.PEElasticCrossSection[1][I - IOFFN[89]]

        for J in range(90, 98):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2]) * CONS[CONI]
                if object.InelasticCrossSectionPerGas[J][I]< 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    if object.WhichAngularModel > 0:
                        object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        # TRIPLET
        object.InelasticCrossSectionPerGas[98][I] = 0.0
        object.PEInelasticCrossSectionPerGas[98][I] = 0.0
        if object.WhichAngularModel == 2:
            object.PEInelasticCrossSectionPerGas[98][I] = 0.0
        if EN > object.EnergyLevels[98]:
            object.InelasticCrossSectionPerGas[98][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTRP2, YTRP2, XTRP2, 2) * 100
        if EN > 2.0 * object.EnergyLevels[98]:
            if object.WhichAngularModel > 0:
                object.PEInelasticCrossSectionPerGas[98][I] = object.PEElasticCrossSection[1][I - IOFFN[98]]

        for J in range(99, 144):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2]) * CONS[CONI]
                if object.InelasticCrossSectionPerGas[J][I]< 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    if object.WhichAngularModel > 0:
                        object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        if EN > 60:
            object.InelasticCrossSectionPerGas[143][I] *= sqrt(60 / EN)
        #LOAD BREMSSTRAHLUNG X-SECTIONS
        object.InelasticCrossSectionPerGas[144][I] = 0.0
        object.InelasticCrossSectionPerGas[145][I] = 0.0
        if EN > 1000:
            object.InelasticCrossSectionPerGas[144][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z6T, EBRM) * 1e-8
            object.InelasticCrossSectionPerGas[145][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z8T, EBRM) * 2e-8

        # Sum ROTATION
        SumR = 0.0
        for J in range(60):
            SumR += object.InelasticCrossSectionPerGas[J][I]

        # Sum VIBRATION
        SumV = 0.0
        for J in range(60, 79):
            SumV += object.InelasticCrossSectionPerGas[J][I]

        # Sum DIPOLE + TRIPLET EXCITATION
        SumE = 0.0
        for J in range(79, 144):
            SumE += object.InelasticCrossSectionPerGas[J][I]

        # Sum TRIPLET EXCITATION
        SumTRP = object.InelasticCrossSectionPerGas[89][I] + object.InelasticCrossSectionPerGas[98][I] + object.InelasticCrossSectionPerGas[143][I]
        # GET Sum DIPOLE
        SumE = SumE - SumTRP
        SumEXC = SumE + SumTRP
        # Sum IONISATION
        SumION = 0.0

        for J in range(11):
            SumION += object.IonizationCrossSection[J][I]
        # GET CORRECT ELASTIC X-SECTION
        object.Q[1][I] -= SumR

        object.Q[0][I] = ElasticCrossSectionA + object.Q[3][I] + SumV + SumE + SumTRP + SumION

    for J in range(1, 74):
        I = (145 - J) - 1
        if object.FinalEnergy <= object.EnergyLevels[I]:
            object.N_Inelastic = I
            break
    return
