from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow,log10
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

    cdef double A0, RY, CONST, ElectronMass2, API, BBCONST, AM2, C,
    cdef int NBREM, i, j, I, J
    A0 = 0.52917720859e-08
    RY = <float> (13.60569193)
    CONST = 1.873884e-20
    ElectronMass2 = <float> (1021997.804)
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / ElectronMass2
    #BORN BETHE VALUES FOR IONISATION
    AM2 = <float> (0.642)
    C = <float> (8.30)

    object.N_Ionization = 2
    object.N_Attachment = 1
    object.N_Inelastic = 107
    object.N_Null = 0

    for i in range(6):
        object.AngularModel[i] = object.WhichAngularModel
    for i in range(4, object.N_Inelastic):
        object.KIN[i] = object.WhichAngularModel

    for i in range(4):
        object.KIN[i] = 0

    cdef int NELM, NROT0, NROT1, NROT2, NROT3, NVIB1, NVIB2, NVIB3, NVIB4, NB3S1, NB3S2, NB3S3, NB3S4, NC3PI, NA3SG, NE3SG, NEFSG, N_IonizationG
    cdef int N_IonizationD, N_Attachment1
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
    N_IonizationG = 92
    N_IonizationD = 61
    N_Attachment1 = 18
    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, ESCOBY, EOBY[2], EATTTH, EATTWD, AMPATT, EATTTH1, EATTWD1, AMPATT1

    object.E = [0.0, 1.0, <float> (15.418), 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * ElectronMass / (<float> (2.01565) * AMU)
    # IONISATION ENERGY FOR PARA =15.42580155 EV
    # IONISATION ENERGY FOR ORTHO=15.41833111 EV
    # USE ORTHO ENERGY FOR ROOM TEMPERATURE GAS
    object.IonizationEnergy[0:2] = [object.E[2], <float> (18.076)]

    # OPAL BEATY FOR LOW ENERGY
    ESCOBY = 0.5
    object.EOBY[0] = object.IonizationEnergy[0] * ESCOBY
    object.EOBY[1] = object.IonizationEnergy[1] * ESCOBY

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

    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break
    #TODO: add EnergyLevels comments

    object.EnergyLevels = gd['gas21/EnergyLevels']

    for J in range(5):
        BEF[J] = object.E[2]

    # ATTACHMENT THRESHOLD EV FOR 2 SIGMAu
    EATTTH = <float> (3.723)
    # ATTACHMENT WIDTH FOR 2 SIGMAu
    EATTWD = <float> (0.45)
    # ATTACHMENT AMPLITUDE FOR 2 SIGMAu
    AMPATT = 3.0e-21
    # ATTACHMENT THRESHOLD EV FOR 2 SIGMAg
    EATTTH1 = <float> (13.922)
    # ATTACHMENT WIDTH FOR 2 SIGMAg
    EATTWD1 = <float> (0.95)
    # ATTACHMENT AMPLITUDE FOR 2 SIGMAg
    AMPATT1 = 3.0e-20

    # ROTATIONAL ENERGY LEVELS: ERLVL(N)
    #  PARA - ORTHO ENERGY DIFFERENCE ( J=0 - J=1 ROT LEVEL) = 0.01469049 EV
    #  REF :ASTROPHYS J.  282(1984)L85
    ERLVL[0] = <float> (0.01469049)
    ERLVL[1] = object.EnergyLevels[4]
    ERLVL[2] = <float> (0.01469049) + object.EnergyLevels[5]
    ERLVL[3] = object.EnergyLevels[4] + object.EnergyLevels[6]
    ERLVL[4] = <float> (0.01469049) + object.EnergyLevels[5] + object.EnergyLevels[7]
    ERLVL[5] = object.EnergyLevels[4] + object.EnergyLevels[6] + <float> (0.15381)
    ERLVL[6] = <float> (0.01469049) + object.EnergyLevels[5] + object.EnergyLevels[7] + <float> (0.1794)

    for J in range(object.N_Inelastic):
        object.PenningFraction[0][J] = 0.0
        object.PenningFraction[1][J] = 1.0
        object.PenningFraction[2][J] = 1.0

    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION
    cdef int NL = 0, FI
    for NL in range(object.N_Inelastic):
        for i in range(4000):
            if object.EG[i] > abs(object.EnergyLevels[NL]):
                IOFFN[NL] = i
                break

    cdef double Sum, FROT[8], GAMMA1, GAMMA2, BETA, BETA2, QMOM, ElasticCrossSectionA, PQ[3], EN, F[84]

    F = [<float> (.0016884), <float> (.005782), <float> (.011536), <float> (.017531), <float> (.022477),
         <float> (.025688), <float> (.027021), <float> (.026731), <float> (.025233), <float> (.022980),
         <float> (.020362), <float> (.017653), <float> (.015054), <float> (.012678), <float> (.010567),
         <float> (.008746), <float> (.007201), <float> (.005909), <float> (.004838), <float> (.003956),
         <float> (.003233), <float> (.002644), <float> (.002165), <float> (.001775), <float> (.001457),
         <float> (.001199), <float> (.0009882), <float> (.0008153), <float> (.0006738), <float> (.0005561),
         <float> (.0004573), <float> (.0003731), <float> (.0002992), <float> (.0002309), <float> (.0001627),
         <float> (8.652e-5), <float> (2.256e-5), <float> (.0476000), <float> (.0728400), <float> (.0698200),
         <float> (.0547200), <float> (.0387400), <float> (.0259800), <float> (.0170000), <float> (.0109900),
         <float> (.0070980), <float> (.0045920), <float> (.0029760), <float> (.0019090), <float> (.0011710),
         <float> (.0005590), <float> (.003970), <float> (.008150), <float> (.009980), <float> (.009520),
         <float> (.007550), <float> (.004230), <float> (.000460), <float> (.000450), <float> (.000300),
         <float> (.007750), <float> (.013100), <float> (.013670), <float> (.011560), <float> (.008730),
         <float> (.006190), <float> (.004280), <float> (.002920), <float> (.001960), <float> (.001330),
         <float> (.000910), <float> (.000630), <float> (.000430), <float> (.000290), <float> (.000200),
         <float> (.000120), <float> (.02230), <float> (.01450), <float> (.01450), <float> (.01010), <float> (.00500),
         <float> (.02680), <float> (.01700), <float> (.00927),
         ]

    Sum = 1.0
    #ROTATIONAL POPULATIONS
    for I in range(1, 8, 2):
        PJ[I - 1] = 3 * (2 * I + 1) * exp(-1 * ERLVL[I - 1] / object.ThermalEnergy)

    for I in range(2, 7, 2):
        PJ[I - 1] = (2 * I + 1) * exp(-1 * ERLVL[I - 1] / object.ThermalEnergy)

    for I in range(7):
        Sum += PJ[I]
    FROT[0] = 1.0 / Sum

    for I in range(1, 8):
        FROT[I] = PJ[I - 1] / Sum

    for I in range(4000):
        EN = object.EG[I]
        if EN > object.EnergyLevels[0]:
            GAMMA1 = (ElectronMass2 + 2 * EN) / ElectronMass2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = sqrt(1.0 - 1.0 / GAMMA2)
            BETA2 = BETA * BETA

        QMOM = GasUtil.CALIonizationCrossSectionREG(EN, NELM, YELM, XELM)
        ElasticCrossSectionA = GasUtil.CALIonizationCrossSectionREG(EN, NELM, YELT, XELM)
        PQ[2] = GasUtil.CALPQ3(EN, NELM, YEPS, XELM)

        PQ = [0.5, 0.5 + (ElasticCrossSectionA - QMOM) / ElasticCrossSectionA, 1 - PQ[2]]

        object.PEElasticCrossSection[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = ElasticCrossSectionA
        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM

        # GROSS IONISATION
        object.IonizationCrossSection[0][I] = 0.0
        object.PEIonizationCrossSection[0][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[0][I] = 0
        if EN >= object.IonizationEnergy[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationG, YION, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[0]:
                object.PEIonizationCrossSection[0][I] = object.PEElasticCrossSection[1][I - IOFFION[0]]

        # DISSOCIATIVE IONISATION
        object.IonizationCrossSection[1][I] = 0.0
        object.PEIonizationCrossSection[1][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[1][I] = 0
        if EN >= object.IonizationEnergy[1]:
            object.IonizationCrossSection[1][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationD, YIOND, XIOND, BETA2, <float>(0.05481), CONST, object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[1]:
                object.PEIonizationCrossSection[1][I] = object.PEElasticCrossSection[1][I - IOFFION[1]]

        # CALCULATE NON_DISSOCIATIVE IONISATION
        if object.IonizationCrossSection[0][I] != 0.0:
            object.IonizationCrossSection[0][I] -= object.IonizationCrossSection[1][I]

        #ATTACHMENT
        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = 0.0
        object.PEElasticCrossSection[3][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEElasticCrossSection[3][I] = 0.0

        #ROTATIONAL DEPENDANCE OF ATTACHMENT TO 2 SIGMAu

        if EN >= (EATTTH - ERLVL[6]):
            object.Q[3][I] = AMPATT * 5.0 * FROT[7] * exp(-1 * (EN - EATTTH + ERLVL[6]) / EATTWD)
        if EN >= (EATTTH - ERLVL[5]):
            object.Q[3][I] += AMPATT * <float>(3.96) * FROT[6] * exp(-1 * (EN - EATTTH + ERLVL[5]) / EATTWD)
        if EN >= (EATTTH - ERLVL[4]):
            object.Q[3][I] += AMPATT * <float>(3.15) * FROT[5] * exp(-1 * (EN - EATTTH + ERLVL[4]) / EATTWD)
        if EN >= (EATTTH - ERLVL[3]):
            object.Q[3][I] += AMPATT * <float>(2.50) * FROT[4] * exp(-1 * (EN - EATTTH + ERLVL[3]) / EATTWD)
        if EN >= (EATTTH - ERLVL[2]):
            object.Q[3][I] += AMPATT * <float>(1.99) * FROT[3] * exp(-1 * (EN - EATTTH + ERLVL[2]) / EATTWD)
        if EN >= (EATTTH - ERLVL[1]):
            object.Q[3][I] += AMPATT * <float>(1.58) * FROT[2] * exp(-1 * (EN - EATTTH + ERLVL[1]) / EATTWD)
        if EN >= (EATTTH - ERLVL[0]):
            object.Q[3][I] += AMPATT * <float>(1.26) * FROT[1] * exp(-1 * (EN - EATTTH + ERLVL[0]) / EATTWD)
        if EN >= EATTTH:
            object.Q[3][I] += AMPATT * FROT[0] * exp(-1 * (EN - EATTTH) / EATTWD)

        if EN > XATT[0]:
            # ATTACHMENT TO 2 SIGMAg
            object.Q[3][I] += GasUtil.CALIonizationCrossSection(EN, N_Attachment1, YATT, XATT)
        if EN > EATTTH1:
            object.Q[3][I] += AMPATT1 * exp(-1 * (EN - EATTTH1) / EATTWD1)
        object.AttachmentCrossSection[0][I] = object.Q[3][I]

        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        for J in range(object.N_Inelastic + 1):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J][I] = 0.0

        # SUPERELASTIC 2-0
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[0][I] = GasUtil.CALIonizationCrossSection(EN + object.EnergyLevels[4], NROT0, YROT0, XROT0)
            object.InelasticCrossSectionPerGas[0][I] *= ((object.EnergyLevels[4] + EN) / EN) * FROT[2] * <float>(0.2)

        # SUPERELASTIC 3-1
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[1][I] = GasUtil.CALIonizationCrossSection(EN + object.EnergyLevels[5], NROT1, YROT1, XROT1)
            object.InelasticCrossSectionPerGas[1][I] *= ((object.EnergyLevels[5] + EN) / EN) * FROT[3] * <float>(3.0 / 7.0)

        # SUPERELASTIC 4-2
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[2][I] = GasUtil.CALIonizationCrossSection(EN + object.EnergyLevels[6], NROT2, YROT2, XROT2)
            object.InelasticCrossSectionPerGas[2][I] *= ((object.EnergyLevels[6] + EN) / EN) * FROT[4] * <float>(5.0 / 9.0)

        # SUPERELASTIC 5-3
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[3][I] = GasUtil.CALIonizationCrossSection(EN + object.EnergyLevels[7], NROT3, YROT3, XROT3)
            object.InelasticCrossSectionPerGas[3][I] *= ((object.EnergyLevels[7] + EN) / EN) * FROT[5] * <float>(7.0 / 11.0)

        # ROTATION 0-2
        if EN > object.EnergyLevels[4]:
            object.InelasticCrossSectionPerGas[4][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NROT0, YROT0, XROT0, 1) * 100 * FROT[0]
            if EN > 2 * object.EnergyLevels[4]:
                object.PEInelasticCrossSectionPerGas[4][I] = object.PEElasticCrossSection[1][I - IOFFN[4]]

        # ROTATION 1-3
        if EN > object.EnergyLevels[5]:
            object.InelasticCrossSectionPerGas[5][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NROT1, YROT1, XROT1, 1) * 100 * FROT[1]
            if EN > 2 * object.EnergyLevels[5]:
                object.PEInelasticCrossSectionPerGas[5][I] = object.PEElasticCrossSection[1][I - IOFFN[5]]

        #                      ROTATION 2-4 + 4-6 + 6-8
        # USED SCALED 2-4 XSECTION FOR 4-6 AND 6-8
        # ALSO SCALED FOR ENERGY LOSS BY 1.5 FOR 4-6 AND BY 2.0 FOR 6-8
        if EN > object.EnergyLevels[6]:
            object.InelasticCrossSectionPerGas[6][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NROT2, YROT2, XROT2, 1) * 100 * (
                    FROT[2] + FROT[4] * <float>(0.8) * 1.5 + FROT[6] * 0.5 * 2.0)
            if EN > 2 * object.EnergyLevels[6]:
                object.PEInelasticCrossSectionPerGas[6][I] = object.PEElasticCrossSection[1][I - IOFFN[6]]

        #                        ROTATION 3-5 + 5-7 + 7-9
        # USED SCALED 3-5 XSECTION FOR 5-7 AND 7-9
        # ALSO SCALED FOR ENERGY LOSS BY 1.4 FOR 5-7 AND 1.8 FOR 7-9
        if EN > object.EnergyLevels[7]:
            object.InelasticCrossSectionPerGas[7][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NROT3, YROT3, XROT3, 1) * 100 * (
                    FROT[3] + FROT[5] * <float>(0.8) * 1.4 + FROT[7] * 0.5 * <float>(1.8))
            if EN > 2 * object.EnergyLevels[7]:
                object.PEInelasticCrossSectionPerGas[7][I] = object.PEElasticCrossSection[1][I - IOFFN[7]]

        #VIBRATION V1 with DJ = 0
        if EN > object.EnergyLevels[8]:
            object.InelasticCrossSectionPerGas[8][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB1, YVIB1, XVIB1, 1) * 100
            if EN > 2 * object.EnergyLevels[8]:
                object.PEInelasticCrossSectionPerGas[8][I] = object.PEElasticCrossSection[1][I - IOFFN[8]]

        #VIBRATION V1 with DJ = 2
        if EN > object.EnergyLevels[9]:
            object.InelasticCrossSectionPerGas[9][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB2, YVIB2, XVIB2, 1) * 100
            if EN > 2 * object.EnergyLevels[9]:
                object.PEInelasticCrossSectionPerGas[9][I] = object.PEElasticCrossSection[1][I - IOFFN[9]]

        #VIBRATION V2
        if EN > object.EnergyLevels[10]:
            object.InelasticCrossSectionPerGas[10][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB3, YVIB3, XVIB3, 1) * 100
            if EN > 2 * object.EnergyLevels[10]:
                object.PEInelasticCrossSectionPerGas[10][I] = object.PEElasticCrossSection[1][I - IOFFN[10]]

        #VIBRATION V3
        if EN > object.EnergyLevels[11]:
            object.InelasticCrossSectionPerGas[11][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB4, YVIB4, XVIB4, 1) * 100
            if EN > 2 * object.EnergyLevels[11]:
                object.PEInelasticCrossSectionPerGas[11][1] = object.PEElasticCrossSection[1][I - IOFFN[11]]

        # B3 SIGMA DISSOCIATION ELOSS=8.0EV
        if EN > object.EnergyLevels[12]:
            object.InelasticCrossSectionPerGas[12][I] = GasUtil.CALIonizationCrossSection(EN, NB3S1, YB3S1, XB3S1)
            if EN > 2 * object.EnergyLevels[12]:
                object.PEInelasticCrossSectionPerGas[12][1] = object.PEElasticCrossSection[1][I - IOFFN[12]]

        # B3 SIGMA DISSOCIATION ELOSS=9.0EV
        if EN > object.EnergyLevels[13]:
            object.InelasticCrossSectionPerGas[13][I] = GasUtil.CALIonizationCrossSection(EN, NB3S2, YB3S2, XB3S2)
            if EN > 2 * object.EnergyLevels[13]:
                object.PEInelasticCrossSectionPerGas[13][1] = object.PEElasticCrossSection[1][I - IOFFN[13]]

        # B3 SIGMA DISSOCIATION ELOSS=9.5EV
        if EN > object.EnergyLevels[14]:
            object.InelasticCrossSectionPerGas[14][I] = GasUtil.CALIonizationCrossSection(EN, NB3S3, YB3S3, XB3S3)
            if EN > 2 * object.EnergyLevels[14]:
                object.PEInelasticCrossSectionPerGas[14][1] = object.PEElasticCrossSection[1][I - IOFFN[14]]

        # B3 SIGMA DISSOCIATION ELOSS=10.0EV
        # SCALED BY 1/E**3 ABOVE XB3S4(NB3S4) EV
        if EN > object.EnergyLevels[15]:
            object.InelasticCrossSectionPerGas[15][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NB3S4, YB3S4, XB3S4, 3) * 100
            if EN > 2 * object.EnergyLevels[15]:
                object.PEInelasticCrossSectionPerGas[15][1] = object.PEElasticCrossSection[1][I - IOFFN[15]]

        #LYMAN BANDS FOR VIB=0 TO 36    B1 SIGMA--- GROUND STATE
        #   DIPOLE ALLOWED
        # V=FI B1 SIGMA
        FI = 0
        for J in range(16, 53):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + BEF[0])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        #V =FI C1 PI
        for J in range(53, 67):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + BEF[1])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        # C3PI V=0-4  METASTABLE LEVEL     FRANCK-CONDON FAC=0.6967
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EnergyLevels[67]:
            object.InelasticCrossSectionPerGas[67][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NC3PI, YC3PI, XC3PI, 3) * 100 * <float>(0.6967)
            if EN > 2 * object.EnergyLevels[67]:
                object.PEInelasticCrossSectionPerGas[67][I] = object.PEElasticCrossSection[1][I - IOFFN[67]]

        # C3PI V=5-18  METASTABLE LEVEL     FRANCK-CONDON FAC=0.3033
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EnergyLevels[68]:
            object.InelasticCrossSectionPerGas[68][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NC3PI, YC3PI, XC3PI, 3) * 100 * <float>(0.3033)
            if EN > 2 * object.EnergyLevels[68]:
                object.PEInelasticCrossSectionPerGas[68][I] = object.PEElasticCrossSection[1][I - IOFFN[68]]

        # A3SG V=0-2  METASTABLE LEVEL     FRANCK-CONDON FAC=0.6668
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EnergyLevels[69]:
            object.InelasticCrossSectionPerGas[69][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NA3SG, YA3SG, XA3SG, 3) * 100 * <float>(0.6668)
            if EN > 2 * object.EnergyLevels[69]:
                object.PEInelasticCrossSectionPerGas[69][I] = object.PEElasticCrossSection[1][I - IOFFN[69]]

        # A3SG V=3-17  METASTABLE LEVEL     FRANCK-CONDON FAC=0.3332
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EnergyLevels[70]:
            object.InelasticCrossSectionPerGas[70][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NA3SG, YA3SG, XA3SG, 3) * 100 * <float>(0.3332)
            if EN > 2 * object.EnergyLevels[70]:
                object.PEInelasticCrossSectionPerGas[70][I] = object.PEElasticCrossSection[1][I - IOFFN[70]]

        # E3SG V=0-9
        # SCALED BY 1/E**3 ABOVE XC3PI(NC3PI) EV
        if EN > object.EnergyLevels[71]:
            object.InelasticCrossSectionPerGas[71][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NE3SG, YE3SG, XE3SG, 3) * 100
            if EN > 2 * object.EnergyLevels[71]:
                object.PEInelasticCrossSectionPerGas[71][I] = object.PEElasticCrossSection[1][I - IOFFN[71]]

        # EF1 SIGMA V=0-5           FRANCK-CONDON FACTOR=0.4
        # USE BORN SCALING ABOVE XEFSG(NEFSG)  EV
        if EN > object.EnergyLevels[72]:
            object.InelasticCrossSectionPerGas[72][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN, NEFSG, YEFSG, XEFSG, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                  BBCONST, object.EnergyLevels[72], BEF[2], <float>(0.0089000))
            if EN <= XEFSG[NEFSG - 1]:
                object.InelasticCrossSectionPerGas[72][I] * 100 * <float>(0.4)
            if EN > 2 * object.EnergyLevels[72]:
                object.PEInelasticCrossSectionPerGas[72][I] = object.PEElasticCrossSection[1][I - IOFFN[72]]

        # EF1 SIGMA V=0-5           FRANCK-CONDON FACTOR=0.6
        # USE BORN SCALING ABOVE XEFSG(NEFSG)  EV
        if EN > object.EnergyLevels[73]:
            object.InelasticCrossSectionPerGas[73][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN, NEFSG, YEFSG, XEFSG, BETA2, GAMMA2, ElectronMass2, object.DEN[I],
                                                  BBCONST, object.EnergyLevels[73], BEF[2], <float>(0.0133000))
            if EN <= XEFSG[NEFSG - 1]:
                object.InelasticCrossSectionPerGas[73][I] * 100 * <float>(0.6)
            if EN > 2 * object.EnergyLevels[73]:
                object.PEInelasticCrossSectionPerGas[73][I] = object.PEElasticCrossSection[1][I - IOFFN[73]]

        #B!1 SIGMA V=FI
        for J in range(74, 83):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + BEF[3])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        #D1 PI  V=FI
        for J in range(83, 99):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + BEF[4])
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        #TODO: add comments
        for J in range(99, 104):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + BEF[4]) * <float>(1.08)
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1

        for J in range(104, 106):
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + BEF[4]) * <float>(1.20)
                if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1
        J = 106
        if EN > object.EnergyLevels[J]:
            object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + BEF[4])
            if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                object.InelasticCrossSectionPerGas[J][I] = 0.0
            if EN > 2 * object.EnergyLevels[J]:
                object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
        FI += 1

        object.Q[0][I] = 0.0
        for J in range(107):
            object.Q[0][I] += object.InelasticCrossSectionPerGas[J][I]

        object.Q[0][I] += object.Q[1][I] + object.Q[3][I] + object.IonizationCrossSection[0][I] + object.IonizationCrossSection[1][I]
    object.N_Inelastic = 12
    if object.FinalEnergy > 8.0 and object.FinalEnergy <= 10.0:
        object.N_Inelastic = 16
    if object.FinalEnergy > 10.0:
        object.N_Inelastic = 107

    return
