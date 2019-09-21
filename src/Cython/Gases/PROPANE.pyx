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
cdef void Gas10(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Propane gas.
    """
    gd = np.load('gases.npy').item()

    cdef double XEN[166], YMT[166], YEL[166], YEPS[166], XION[45], YIONG[45], YIONC[45], XION1[45], YION1[45], XION2[45], YION2[45]
    cdef double XION3[45], YION3[45], XION4[45], YION4[45], XION5[45], YION5[45], XION6[44], YION6[44], XION7[44], YION7[44], XION8[44]
    cdef double YION8[44], XION9[44], YION9[44], XION10[44], YION10[44], XION11[43], YION11[43], XION12[41], YION12[41], XION13[41]
    cdef double YION13[41], XION14[40], YION14[40], XION15[39], YION15[39], XION16[39], YION16[39], XION17[39], YION17[39], XION18[38]
    cdef double YION18[38], XION19[39], YION19[39], XION20[38], YION20[38], XION21[36], YION21[36], XION22[36], YION22[36], XION23[36]
    cdef double YION23[36], XION24[83], YION24[83], XATT1[9], YATT1[9], XATT2[9], YATT2[9], XVIB1[25], YVIB1[25], XVIB2[24], YVIB2[24]
    cdef double XVIB3[25], YVIB3[25], XVIB4[17], YVIB4[17], XTR1[14], YTR1[14], XTR2[11], YTR2[11], XTR3[11], YTR3[11], XTR4[11], YTR4[11]
    cdef double XNUL1[14], YNUL1[14], XNUL2[14], YNUL2[14], Z1T[25], Z6T[25], EBRM[25]
    object.EnergyLevels = gd['gas10/EnergyLevels']
    XEN = gd['gas10/XEN']
    YMT = gd['gas10/YMT']
    YEL = gd['gas10/YEL']
    YEPS = gd['gas10/YEPS']
    XION = gd['gas10/XION']
    YIONG = gd['gas10/YIONG']
    YIONC = gd['gas10/YIONC']
    XION1 = gd['gas10/XION1']
    YION1 = gd['gas10/YION1']
    XION2 = gd['gas10/XION2']
    YION2 = gd['gas10/YION2']
    XION3 = gd['gas10/XION3']
    YION3 = gd['gas10/YION3']
    XION4 = gd['gas10/XION4']
    YION4 = gd['gas10/YION4']
    XION5 = gd['gas10/XION5']
    YION5 = gd['gas10/YION5']
    XION6 = gd['gas10/XION6']
    YION6 = gd['gas10/YION6']
    XION7 = gd['gas10/XION7']
    YION7 = gd['gas10/YION7']
    XION8 = gd['gas10/XION8']
    YION8 = gd['gas10/YION8']
    XION9 = gd['gas10/XION9']
    YION9 = gd['gas10/YION9']
    XION10 = gd['gas10/XION10']
    YION10 = gd['gas10/YION10']
    XION11 = gd['gas10/XION11']
    YION11 = gd['gas10/YION11']
    XION12 = gd['gas10/XION12']
    YION12 = gd['gas10/YION12']
    XION13 = gd['gas10/XION13']
    YION13 = gd['gas10/YION13']
    XION14 = gd['gas10/XION14']
    YION14 = gd['gas10/YION14']
    XION15 = gd['gas10/XION15']
    YION15 = gd['gas10/YION15']
    XION16 = gd['gas10/XION16']
    YION16 = gd['gas10/YION16']
    XION17 = gd['gas10/XION17']
    YION17 = gd['gas10/YION17']
    XION18 = gd['gas10/XION18']
    YION18 = gd['gas10/YION18']
    XION19 = gd['gas10/XION19']
    YION19 = gd['gas10/YION19']
    XION20 = gd['gas10/XION20']
    YION20 = gd['gas10/YION20']
    XION21 = gd['gas10/XION21']
    YION21 = gd['gas10/YION21']
    XION22 = gd['gas10/XION22']
    YION22 = gd['gas10/YION22']
    XION23 = gd['gas10/XION23']
    YION23 = gd['gas10/YION23']
    XION24 = gd['gas10/XION24']
    YION24 = gd['gas10/YION24']
    XATT1 = gd['gas10/XATT1']
    YATT1 = gd['gas10/YATT1']
    XATT2 = gd['gas10/XATT2']
    YATT2 = gd['gas10/YATT2']
    XVIB1 = gd['gas10/XVIB1']
    YVIB1 = gd['gas10/YVIB1']
    XVIB2 = gd['gas10/XVIB2']
    YVIB2 = gd['gas10/YVIB2']
    XVIB3 = gd['gas10/XVIB3']
    YVIB3 = gd['gas10/YVIB3']
    XVIB4 = gd['gas10/XVIB4']
    YVIB4 = gd['gas10/YVIB4']
    XTR1 = gd['gas10/XTR1']
    YTR1 = gd['gas10/YTR1']
    XTR2 = gd['gas10/XTR2']
    YTR2 = gd['gas10/YTR2']
    XTR3 = gd['gas10/XTR3']
    YTR3 = gd['gas10/YTR3']
    XTR4 = gd['gas10/XTR4']
    YTR4 = gd['gas10/YTR4']
    XNUL1 = gd['gas10/XNUL1']
    YNUL1 = gd['gas10/YNUL1']
    XNUL2 = gd['gas10/XNUL2']
    YNUL2 = gd['gas10/YNUL2']
    Z1T = gd['gas10/Z1T']
    Z6T = gd['gas10/Z6T']
    EBRM = gd['gas10/EBRM']

    cdef double A0, RY, CONST, ElectronMass2, API, BBCONST, AM2, C,

    # BORN-BETHE CONSTANTS
    A0 = 0.52917720859e-08
    RY = <float> (13.60569193)
    CONST = 1.873884e-20
    ElectronMass2 = <float> (1021997.804)
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / ElectronMass2

    # BORN BETHE FOR IONISATION
    AM2 = <float> (10.52)
    C = <float> (125.50)
    # ARRAY SIZE
    NASIZE = 4000

    object.N_Ionization = 24
    object.N_Attachment = 2
    object.N_Inelastic = 64
    object.N_Null = 2

    cdef int NBREM, NDATA, N_IonizationG, N_Ionization1, N_Ionization2, N_Ionization3, N_Ionization4, N_Ionization5, N_Ionization6, N_Ionization7, N_Ionization8, N_Ionization9, N_Ionization10, N_Ionization11, N_Ionization12, N_Ionization13, N_Ionization14
    cdef int N_Ionization15, N_Ionization16, N_Ionization17, N_Ionization18, N_Ionization19, N_Ionization20, N_Ionization21, N_Ionization22, N_Ionization23, N_Ionization24, N_Attachment1, N_Attachment2, NVIB1, NVIB2, NVIB3, NVIB4
    cdef int NTR1, NTR2, NTR3, NTR4, NUL1, NUL2, IOFFION[24], IOFFN[64]

    cdef int i, j, I, J, NL
    NBREM = 25
    for i in range(6):
        object.AngularModel[i] = object.WhichAngularModel

    for i in range(8):
        object.KIN[i] = 0

    for i in range(8, object.N_Inelastic):
        object.KIN[i] = 2

    NDATA = 166
    N_IonizationG = 46
    N_Ionization1 = 45
    N_Ionization2 = 45
    N_Ionization3 = 45
    N_Ionization4 = 45
    N_Ionization5 = 45
    N_Ionization6 = 44
    N_Ionization7 = 44
    N_Ionization8 = 44
    N_Ionization9 = 44
    N_Ionization10 = 44
    N_Ionization11 = 43
    N_Ionization12 = 41
    N_Ionization13 = 41
    N_Ionization14 = 40
    N_Ionization15 = 39
    N_Ionization16 = 39
    N_Ionization17 = 39
    N_Ionization18 = 38
    N_Ionization19 = 39
    N_Ionization20 = 38
    N_Ionization21 = 36
    N_Ionization22 = 36
    N_Ionization23 = 36
    N_Ionization24 = 83
    N_Attachment1 = 9
    N_Attachment2 = 9
    NVIB1 = 25
    NVIB2 = 24
    NVIB3 = 25
    NVIB4 = 17
    NTR1 = 14
    NTR2 = 11
    NTR3 = 11
    NTR4 = 11
    NUL1 = 14
    NUL2 = 14

    object.ScaleNull[0] = 1.0
    object.ScaleNull[1] = 1.0

    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, EOBY[30], SCLOBY, APOP1, APOP2, APOP3, APOP4, QCOUNT = 0.0, IonizationCrossSectionC, IonizationCrossSectionG,FAC

    object.E = [0.0, 1.0, <float> (11.05), 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * ElectronMass / (<float> (44.09652) * AMU)

    object.IonizationEnergy[0:24] = [<float> (11.11), <float> (11.55), <float> (11.75), <float> (11.75), <float> (11.91),
                         <float> (13.48), <float> (13.65), <float> (13.79), <float> (14.1), <float> (14.5),
                         <float> (16.5), <float> (20.0), <float> (21.5),
                         25.0,
                         <float> (26.5), 29.0, <float> (30.4), 32.0, 32.0, 36.0, 39.0, 39.0, 39.0, 285.0]

    # OPAL BEATY
    SCLOBY = <float> (0.8)
    for J in range(object.N_Ionization):
        EOBY[J] = object.IonizationEnergy[J] * SCLOBY
    EOBY[object.N_Ionization - 1] *= <float> (0.63)

    for i in range(23):
        object.NC0[i] = 0
        object.EC0[i] = 0.0
        object.WK[i] = 0.0
        object.EFL[i] = 0.0
        object.NG1[i] = 0
        object.EG1[i] = 0.0
        object.NG2[i] = 0
        object.EG2[i] = 0.0
    # DOUBLE CHARGE , ++ ION STATES ( EXTRA ELECTRON )
    object.NC0[17] = 1
    object.EC0[17] = 4.0
    object.NC0[18] = 1
    object.EC0[18] = 4.0
    # FLUORESCENCE DATA (KSHELL)
    object.NC0[23] = 2
    object.EC0[23] = 253.0
    object.WK[23] = <float> (0.0026)
    object.EFL[23] = 273.0
    object.NG1[23] = 1
    object.EG1[23] = 253.0
    object.NG2[23] = 2
    object.EG2[23] = 5.0
    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break

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

    cdef double EN, ENLG, GAMMA1, GAMMA2, BETA, BETA2, QMT, ElasticCrossSection, PQ[3], X1, X2, QBB = 0.0, CrossSectionSum, EFAC, F[52]
    F[0:52] = [<float> (.000339), <float> (.004660), <float> (.012816), <float> (.037747), <float> (.081783),
         <float> (.084248), <float> (.090347), <float> (.098580), <float> (.10415), <float> (.11379), <float> (.12674),
         <float> (.096356), <float> (.10387), <float> (.10183), <float> (.096718), <float> (.090149), <float> (.086661),
         <float> (.086097), <float> (.083324), <float> (.079943), <float> (.077210), <float> (.070368),
         <float> (.061365), <float> (.053208), <float> (.046320), <float> (.042827), <float> (.038898),
         <float> (.035930), <float> (.033632), <float> (.030562), <float> (.028559), <float> (.027052),
         <float> (.048051), <float> (.036375), <float> (.020165), <float> (.010038), <float> (.0054441),
         <float> (.0050790), <float> (.0057699), <float> (.0072715), <float> (.010296), <float> (.014152),
         <float> (.013698), <float> (.010362), <float> (.0088401), <float> (.022195), <float> (.019172),
         <float> (.011553), <float> (.0089679), <float> (.0064815), <float> (.0035484), <float> (.0010872)]
    cdef int FI = 0
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

        # CALCULATE GROSS AND COUNTING IONISATIONS
        # NOT USED: ONLY FOR COMPARISON TO TOTAL OF IONISATION CHMoleculesPerCm3PerGasELS
        #     GROSS IONISATION
        if EN > object.IonizationEnergy[0]:
            IonizationCrossSectionG = GasUtil.CALIonizationCrossSection(EN, N_IonizationG, YIONG, XION)
            IonizationCrossSectionC = GasUtil.CALIonizationCrossSection(EN, N_IonizationG, YIONC, XION)
            if EN > XION[N_IonizationG - 1]:
                X2 = 1 / BETA2
                X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                QBB = CONST * (AM2 * (X1 - object.DEN[I] / 2) + C * X2)
                IonizationCrossSectionC = QBB
                IonizationCrossSectionG = IonizationCrossSectionC / <float>(0.8939)

        # C3H8+
        if EN > object.IonizationEnergy[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization1, YION1, XION1)
            if object.IonizationCrossSection[0][I] == 0:
                if EN > XION1[N_Ionization1 - 1]:
                    object.IonizationCrossSection[0][I] = IonizationCrossSectionC * <float>(0.103628)
            if EN >= 2 * object.IonizationEnergy[0]:
                object.PEIonizationCrossSection[0][I] = object.PEElasticCrossSection[1][I - IOFFION[0]]

        # C3H7+
        if EN > object.IonizationEnergy[1]:
            object.IonizationCrossSection[1][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization2, YION2, XION2)
            if object.IonizationCrossSection[1][I] == 0:
                if EN > XION2[N_Ionization2 - 1]:
                    object.IonizationCrossSection[1][I] = IonizationCrossSectionC * <float>(0.073774)
            if EN >= 2 * object.IonizationEnergy[1]:
                object.PEIonizationCrossSection[1][I] = object.PEElasticCrossSection[1][I - IOFFION[1]]

        # C3H6+
        if EN > object.IonizationEnergy[2]:
            object.IonizationCrossSection[2][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization3, YION3, XION3)
            if object.IonizationCrossSection[2][I] == 0:
                if EN > XION3[N_Ionization3 - 1]:
                    object.IonizationCrossSection[2][I] = IonizationCrossSectionC * <float>(0.017780)
            if EN >= 2 * object.IonizationEnergy[2]:
                object.PEIonizationCrossSection[2][I] = object.PEElasticCrossSection[1][I - IOFFION[2]]
        # C2H4+
        if EN > object.IonizationEnergy[3]:
            object.IonizationCrossSection[3][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization4, YION4, XION4)
            if object.IonizationCrossSection[3][I] == 0:
                if EN > XION4[N_Ionization4 - 1]:
                    object.IonizationCrossSection[3][I] = IonizationCrossSectionC * <float>(0.151263)
            if EN >= 2 * object.IonizationEnergy[3]:
                object.PEIonizationCrossSection[3][I] = object.PEElasticCrossSection[1][I - IOFFION[3]]
        # C2H5+
        if EN > object.IonizationEnergy[4]:
            object.IonizationCrossSection[4][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization5, YION5, XION5)
            if object.IonizationCrossSection[4][I] == 0:
                if EN > XION5[N_Ionization5 - 1]:
                    object.IonizationCrossSection[4][I] = IonizationCrossSectionC * <float>(0.238836)
            if EN >= 2 * object.IonizationEnergy[4]:
                object.PEIonizationCrossSection[4][I] = object.PEElasticCrossSection[1][I - IOFFION[4]]
        # C3H5+
        if EN > object.IonizationEnergy[5]:
            object.IonizationCrossSection[5][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization6, YION6, XION6)
            if object.IonizationCrossSection[5][I] == 0:
                if EN > XION6[N_Ionization6 - 1]:
                    object.IonizationCrossSection[5][I] = IonizationCrossSectionC * <float>(0.040867)
            if EN >= 2 * object.IonizationEnergy[5]:
                object.PEIonizationCrossSection[5][I] = object.PEElasticCrossSection[1][I - IOFFION[5]]
        # CH3+
        if EN > object.IonizationEnergy[6]:
            object.IonizationCrossSection[6][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization7, YION7, XION7)
            if object.IonizationCrossSection[6][I] == 0:
                if EN > XION7[N_Ionization7 - 1]:
                    object.IonizationCrossSection[6][I] = IonizationCrossSectionC * <float>(0.019372)
            if EN >= 2 * object.IonizationEnergy[6]:
                object.PEIonizationCrossSection[6][I] = object.PEElasticCrossSection[1][I - IOFFION[6]]
        # C3H4+
        if EN > object.IonizationEnergy[7]:
            object.IonizationCrossSection[7][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization8, YION8, XION8)
            if object.IonizationCrossSection[7][I] == 0:
                if EN > XION8[N_Ionization8 - 1]:
                    object.IonizationCrossSection[7][I] = IonizationCrossSectionC * <float>(0.007842)
            if EN >= 2 * object.IonizationEnergy[7]:
                object.PEIonizationCrossSection[7][I] = object.PEElasticCrossSection[1][I - IOFFION[7]]
        # C2H2+
        if EN > object.IonizationEnergy[8]:
            object.IonizationCrossSection[8][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization9, YION9, XION9)
            if object.IonizationCrossSection[8][I] == 0:
                if EN > XION9[N_Ionization9 - 1]:
                    object.IonizationCrossSection[8][I] = IonizationCrossSectionC * <float>(0.025343)
            if EN >= 2 * object.IonizationEnergy[8]:
                object.PEIonizationCrossSection[8][I] = object.PEElasticCrossSection[1][I - IOFFION[8]]
        # C2H3+
        if EN > object.IonizationEnergy[9]:
            object.IonizationCrossSection[9][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization10, YION10, XION10)
            if object.IonizationCrossSection[9][I] == 0:
                if EN > XION10[N_Ionization10 - 1]:
                    object.IonizationCrossSection[9][I] = IonizationCrossSectionC * <float>(0.112253)
            if EN >= 2 * object.IonizationEnergy[9]:
                object.PEIonizationCrossSection[9][I] = object.PEElasticCrossSection[1][I - IOFFION[9]]
        # C3H3+
        if EN > object.IonizationEnergy[10]:
            object.IonizationCrossSection[10][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization11, YION11, XION11)
            if object.IonizationCrossSection[10][I] == 0:
                if EN > XION11[N_Ionization11 - 1]:
                    object.IonizationCrossSection[10][I] = IonizationCrossSectionC * <float>(0.049359)
            if EN >= 2 * object.IonizationEnergy[10]:
                object.PEIonizationCrossSection[10][I] = object.PEElasticCrossSection[1][I - IOFFION[10]]
        # H+
        if EN > object.IonizationEnergy[11]:
            object.IonizationCrossSection[11][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization12, YION12, XION12)
            if object.IonizationCrossSection[11][I] == 0:
                if EN > XION12[N_Ionization12 - 1]:
                    object.IonizationCrossSection[11][I] = IonizationCrossSectionC * <float>(0.001884)
            if EN >= 2 * object.IonizationEnergy[11]:
                object.PEIonizationCrossSection[11][I] = object.PEElasticCrossSection[1][I - IOFFION[11]]
        # H2+ AND H3+
        if EN > object.IonizationEnergy[12]:
            object.IonizationCrossSection[12][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization13, YION13, XION13)
            if object.IonizationCrossSection[12][I] == 0:
                if EN > XION13[N_Ionization13 - 1]:
                    object.IonizationCrossSection[12][I] = IonizationCrossSectionC * <float>(0.001015)
            if EN >= 2 * object.IonizationEnergy[12]:
                object.PEIonizationCrossSection[12][I] = object.PEElasticCrossSection[1][I - IOFFION[12]]
        # CH2+
        if EN > object.IonizationEnergy[13]:
            object.IonizationCrossSection[13][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization14, YION14, XION14)
            if object.IonizationCrossSection[13][I] == 0:
                if EN > XION14[N_Ionization14 - 1]:
                    object.IonizationCrossSection[13][I] = IonizationCrossSectionC * <float>(0.006342)
            if EN >= 2 * object.IonizationEnergy[13]:
                object.PEIonizationCrossSection[13][I] = object.PEElasticCrossSection[1][I - IOFFION[13]]

        # C3H2+
        if EN > object.IonizationEnergy[14]:
            object.IonizationCrossSection[14][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization15, YION15, XION15)
            if object.IonizationCrossSection[14][I] == 0:
                if EN > XION15[N_Ionization15 - 1]:
                    object.IonizationCrossSection[14][I] = IonizationCrossSectionC * <float>(0.013401)
            if EN >= 2 * object.IonizationEnergy[14]:
                object.PEIonizationCrossSection[14][I] = object.PEElasticCrossSection[1][I - IOFFION[14]]

        # C3H+
        if EN > object.IonizationEnergy[15]:
            object.IonizationCrossSection[15][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization16, YION16, XION16)
            if object.IonizationCrossSection[15][I] == 0:
                if EN > XION16[N_Ionization16 - 1]:
                    object.IonizationCrossSection[15][I] = IonizationCrossSectionC * <float>(0.008240)
            if EN >= 2 * object.IonizationEnergy[15]:
                object.PEIonizationCrossSection[15][I] = object.PEElasticCrossSection[1][I - IOFFION[15]]

        # C2H+
        if EN > object.IonizationEnergy[16]:
            object.IonizationCrossSection[16][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization17, YION17, XION17)
            if object.IonizationCrossSection[16][I] == 0:
                if EN > XION17[N_Ionization17 - 1]:
                    object.IonizationCrossSection[16][I] = IonizationCrossSectionC * <float>(0.002004)
            if EN >= 2 * object.IonizationEnergy[16]:
                object.PEIonizationCrossSection[16][I] = object.PEElasticCrossSection[1][I - IOFFION[16]]
        # ++ DOUBLE CHARGED STABLE IONS
        if EN > object.IonizationEnergy[17]:
            object.IonizationCrossSection[17][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization18, YION18, XION18)
            if object.IonizationCrossSection[17][I] == 0:
                if EN > XION18[N_Ionization18 - 1]:
                    object.IonizationCrossSection[17][I] = IonizationCrossSectionC * <float>(0.004085)
            if EN >= 2 * object.IonizationEnergy[17]:
                object.PEIonizationCrossSection[17][I] = object.PEElasticCrossSection[1][I - IOFFION[17]]
        # ++ DOUBLE CHARGED UNSTABLE IONS (DISSOCIATIVE)
        if EN > object.IonizationEnergy[18]:
            object.IonizationCrossSection[18][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization19, YION19, XION19)
            if object.IonizationCrossSection[18][I] == 0:
                if EN > XION19[N_Ionization19 - 1]:
                    object.IonizationCrossSection[18][I] = IonizationCrossSectionC * <float>(0.118714)
            if EN >= 2 * object.IonizationEnergy[18]:
                object.PEIonizationCrossSection[18][I] = object.PEElasticCrossSection[1][I - IOFFION[18]]
        # CH+
        if EN > object.IonizationEnergy[19]:
            object.IonizationCrossSection[19][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization20, YION20, XION20)
            if object.IonizationCrossSection[19][I] == 0:
                if EN > XION20[N_Ionization20 - 1]:
                    object.IonizationCrossSection[19][I] = IonizationCrossSectionC * <float>(0.002070)
            if EN >= 2 * object.IonizationEnergy[19]:
                object.PEIonizationCrossSection[19][I] = object.PEElasticCrossSection[1][I - IOFFION[19]]
        # C+
        if EN > object.IonizationEnergy[20]:
            object.IonizationCrossSection[20][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization21, YION21, XION21)
            if object.IonizationCrossSection[20][I] == 0:
                if EN > XION21[N_Ionization21 - 1]:
                    object.IonizationCrossSection[20][I] = IonizationCrossSectionC * <float>(0.000837)
            if EN >= 2 * object.IonizationEnergy[20]:
                object.PEIonizationCrossSection[20][I] = object.PEElasticCrossSection[1][I - IOFFION[20]]
        # C2+
        if EN > object.IonizationEnergy[21]:
            object.IonizationCrossSection[21][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization22, YION22, XION22)
            if object.IonizationCrossSection[21][I] == 0:
                if EN > XION22[N_Ionization22 - 1]:
                    object.IonizationCrossSection[21][I] = IonizationCrossSectionC * <float>(0.000057)
            if EN >= 2 * object.IonizationEnergy[21]:
                object.PEIonizationCrossSection[21][I] = object.PEElasticCrossSection[1][I - IOFFION[21]]
        # C3+
        if EN > object.IonizationEnergy[22]:
            object.IonizationCrossSection[22][I] = GasUtil.CALIonizationCrossSection(EN, N_Ionization23, YION23, XION23)
            if object.IonizationCrossSection[22][I] == 0:
                if EN > XION23[N_Ionization23 - 1]:
                    object.IonizationCrossSection[22][I] = IonizationCrossSectionC * <float>(0.001034)
            if EN >= 2 * object.IonizationEnergy[22]:
                object.PEIonizationCrossSection[22][I] = object.PEElasticCrossSection[1][I - IOFFION[22]]
        # CARBON K-SHEL
        if EN > object.IonizationEnergy[23]:
            object.IonizationCrossSection[23][I] = 3 * GasUtil.CALIonizationCrossSection(EN, N_Ionization24, YION24, XION24)
            if EN >= 2 * object.IonizationEnergy[23]:
                object.PEIonizationCrossSection[23][I] = object.PEElasticCrossSection[1][I - IOFFION[23]]
        # CORRECTION TO TOTAL I0NISATION DUE TO SPLIT OFF KSHELL
        CrossSectionSum = 0.0
        for J in range(23):
            CrossSectionSum += object.IonizationCrossSection[J][I]

        if CrossSectionSum != 0.0:
            FAC = (CrossSectionSum - object.IonizationCrossSection[23][I])/CrossSectionSum
            for J in range(23):
                object.IonizationCrossSection[J][I] = object.IonizationCrossSection[J][I]* FAC

        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = 0.0
        # DISSOCIATIVE ATTACHMENT TO CH3-
        if EN >= XATT1[0]:
            object.AttachmentCrossSection[0][I] = GasUtil.CALIonizationCrossSection(EN, N_Attachment1, YATT1, XATT1)

        object.AttachmentCrossSection[1][I] = 0.0
        # DISSOCIATIVE ATTACHMENT TO H-
        if EN >= XATT2[0]:
            object.AttachmentCrossSection[1][I] = GasUtil.CALIonizationCrossSection(EN, N_Attachment2, YATT2, XATT2)

        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        # set ZEROS

        for J in range(object.N_Inelastic):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J][I] = 0.0

        # SUPERELASTIC VIBRATION-TORSION         AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > 0.0:
            EFAC = sqrt(1.0 - (object.EnergyLevels[0] / EN))
            object.InelasticCrossSectionPerGas[0][I] = <float>(0.00536) * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.InelasticCrossSectionPerGas[0][I] *= APOP1 / (1.0 + APOP1) * 1.e-16
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[0][I] = object.PEElasticCrossSection[1][I - IOFFN[0]]
        #VIBRATION-TORSION                      AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > object.EnergyLevels[1]:
            EFAC = sqrt(1.0 - (object.EnergyLevels[1] / EN))
            object.InelasticCrossSectionPerGas[1][I] = <float>(0.00536) * log((EFAC + 1.0) / (1.0 - EFAC)) / EN
            object.InelasticCrossSectionPerGas[1][I] *= 1.0 / (1.0 + APOP1) * 1.e-16
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[1][I] = object.PEElasticCrossSection[1][I - IOFFN[1]]

        #SUPERELASTIC VIB1                     AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[2][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB1, YVIB1, XVIB1, APOP2 / (1 + APOP2), object.EnergyLevels[3], 1,
                                                  -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[2][I] = object.PEElasticCrossSection[1][I - IOFFN[2]]

        #VIB INELASTIC                          AAnisotropicDetectedTROPIC ABOVE 10 EV
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

        #VIB INELASTIC                          AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > object.EnergyLevels[6]:
            object.InelasticCrossSectionPerGas[6][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB3, YVIB3, XVIB3, 1, 0, 1, -1 * 5 * EN, 0)
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[6][I] = object.PEElasticCrossSection[1][I - IOFFN[6]]

        #VIB INELASTIC                          AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN > object.EnergyLevels[7]:
            object.InelasticCrossSectionPerGas[7][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB4, YVIB4, XVIB4, 1, 0, 1, -1 * 5 * EN, 0)
            if object.InelasticCrossSectionPerGas[7][I] < 0.0:
                object.InelasticCrossSectionPerGas[7][I] = 0.0
            if EN > 10:
                object.PEInelasticCrossSectionPerGas[7][I] = object.PEElasticCrossSection[1][I - IOFFN[7]]

        #
        # EXCITATIONS
        #
        #
        # EXCITATION TO TRIPLET AND SINGLET LEVELS
        #
        # FIRST TRIPLET AT  6.57 EV

        if EN > object.EnergyLevels[8]:
            object.InelasticCrossSectionPerGas[8][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTR1, YTR1, XTR1, 2) * 100
            if EN > 3 * object.EnergyLevels[8]:
                object.PEInelasticCrossSectionPerGas[8][I] = object.PEElasticCrossSection[1][I - IOFFN[8]]

        #SINGLET DISSOCIATION AT  7.65  EV     BEF SCALING F[FI]
        FI = 0
        J = 9

        if EN > object.EnergyLevels[J]:
            object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
            if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                object.InelasticCrossSectionPerGas[J][I] = 0.0
            if EN > 3 * object.EnergyLevels[J]:
                object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
        FI += 1

        # SECOND TRIPLET AT  7.67 EV
        if EN > object.EnergyLevels[10]:
            object.InelasticCrossSectionPerGas[10][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTR2, YTR2, XTR2, 2) * 100
            if EN > 3 * object.EnergyLevels[10]:
                object.PEInelasticCrossSectionPerGas[10][I] = object.PEElasticCrossSection[1][I - IOFFN[10]]

        #SINGLET DISSOCIATION AT  7.65+.3*FI  EV     BEF SCALING F[FI]

        for J in range(11, 17):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 3 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]

            FI += 1

        # THIRD TRIPLET AT  9.59 EV
        if EN > object.EnergyLevels[17]:
            object.InelasticCrossSectionPerGas[17][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTR3, YTR3, XTR3, 2) * 100
            if EN > 3 * object.EnergyLevels[17]:
                object.PEInelasticCrossSectionPerGas[17][I] = object.PEElasticCrossSection[1][I - IOFFN[17]]

        #SINGLET DISSOCIATION AT  7.65+.3*FI  EV     BEF SCALING F[FI]

        for J in range(18, 32):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 3 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]

            FI += 1

        for J in range(32, 61):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]

            FI += 1

        # FOURTH TRIPLET AT  26.0 EV
        if EN > object.EnergyLevels[61]:
            object.InelasticCrossSectionPerGas[61][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NTR4, YTR4, XTR4, 2) * 100
            if EN > 3 * object.EnergyLevels[61]:
                object.PEInelasticCrossSectionPerGas[61][I] = object.PEElasticCrossSection[1][I - IOFFN[61]]

        #SINGLET DISSOCIATION AT  7.65+.3*FI  EV     BEF SCALING F[FI]

        for J in range(62, 64):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]

            FI += 1
        # LOAD BREMSSTRAHLUNG X-SECTION
        object.InelasticCrossSectionPerGas[64][I] = 0.0
        object.InelasticCrossSectionPerGas[65][I] = 0.0

        #  LOAD NULL COLLISIONS
        #
        # LIGHT EMISSION FROM H ALPHA
        # MOHLMMoleculesPerCm3PerGas AND DE HEER CHEM.PHYS.19(1979)233      
        object.NullCrossSection[0][I] = 0.0
        if EN > XNUL1[0]:
            object.NullCrossSection[0][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NUL1, YNUL1, XNUL1, 1) * 100 * <float>(0.9) * object.ScaleNull[0]

        # LIGHT EMISSION FROM CH2(A2DELTA - X2PI)
        #  MOHLMMoleculesPerCm3PerGas AND DE HEER  CHEM.PHYS.19(1979)233

        object.NullCrossSection[1][I] = 0.0
        if EN > XNUL2[0]:
            object.NullCrossSection[1][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NUL2, YNUL2, XNUL2, 1) * 100 * object.ScaleNull[1]

    for J in range(object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break
    return
