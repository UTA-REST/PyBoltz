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
cdef void Gas8(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for CH4 gas.
    """
    gd = np.load('gases.npy').item()
    cdef double XEN[153], YELM[153], YELT[153], YEPS[153], XATT[6], YATT[6], XVBV4[26], YVBV4[26], XVBV2[29], YVBV2[29], XVBV1[30], YVBV1[30],
    cdef double XVBV3[25], YVBV3[25], XVBH1[14], YVBH1[14], XVBH2[14], YVBH2[14], XION[70], YION[70], YINC[70], XINF[70], YINF[70], XINF1[68], YINF1[68],
    cdef double XINF2[66], YINF2[66], XINF3[53], YINF3[53], XINF4[51], YINF4[51], XINF5[50], YINF5[50], XINF6[48], YINF6[48], XINPP[49], YINPP[49],
    cdef double XDET[9], YDET[9], XTR1[12], YTR1[12], XTR2[11], YTR2[11], XTR3[11], YTR3[11], XCHD[32], YCHD[32], XCHB[35], YCHB[35], XHAL[34], YHAL[34],
    cdef double XHBE[34], YHBE[34], XKSH[83], YKSH[83], Z1T[25], Z6T[25], EBRM[25],
    cdef int IOFFN[34], IOFFION[10]
    XEN = gd['gas8/XEN']
    YELM = gd['gas8/YELM']
    YELT = gd['gas8/YELT']
    YEPS = gd['gas8/YEPS']
    XATT = gd['gas8/XATT']
    YATT = gd['gas8/YATT']
    XVBV4 = gd['gas8/XVBV4']
    YVBV4 = gd['gas8/YVBV4']
    XVBV2 = gd['gas8/XVBV2']
    YVBV2 = gd['gas8/YVBV2']
    XVBV1 = gd['gas8/XVBV1']
    YVBV1 = gd['gas8/YVBV1']
    XVBV3 = gd['gas8/XVBV3']
    YVBV3 = gd['gas8/YVBV3']
    XVBH1 = gd['gas8/XVBH1']
    YVBH1 = gd['gas8/YVBH1']
    XVBH2 = gd['gas8/XVBH2']
    YVBH2 = gd['gas8/YVBH2']
    XION = gd['gas8/XION']
    YION = gd['gas8/YION']
    YINC = gd['gas8/YINC']
    XINF = gd['gas8/XINF']
    YINF = gd['gas8/YINF']
    XINF1 = gd['gas8/XINF1']
    YINF1 = gd['gas8/YINF1']
    XINF2 = gd['gas8/XINF2']
    YINF2 = gd['gas8/YINF2']
    XINF3 = gd['gas8/XINF3']
    YINF3 = gd['gas8/YINF3']
    XINF4 = gd['gas8/XINF4']
    YINF4 = gd['gas8/YINF4']
    XINF5 = gd['gas8/XINF5']
    YINF5 = gd['gas8/YINF5']
    XINF6 = gd['gas8/XINF6']
    YINF6 = gd['gas8/YINF6']
    XINPP = gd['gas8/XINPP']
    YINPP = gd['gas8/YINPP']
    XDET = gd['gas8/XDET']
    YDET = gd['gas8/YDET']
    XTR1 = gd['gas8/XTR1']
    YTR1 = gd['gas8/YTR1']
    XTR2 = gd['gas8/XTR2']
    YTR2 = gd['gas8/YTR2']
    XTR3 = gd['gas8/XTR3']
    YTR3 = gd['gas8/YTR3']
    XCHD = gd['gas8/XCHD']
    YCHD = gd['gas8/YCHD']
    XCHB = gd['gas8/XCHB']
    YCHB = gd['gas8/YCHB']
    XHAL = gd['gas8/XHAL']
    YHAL = gd['gas8/YHAL']
    XHBE = gd['gas8/XHBE']
    YHBE = gd['gas8/YHBE']
    XKSH = gd['gas8/XKSH']
    YKSH = gd['gas8/YKSH']
    Z1T = gd['gas8/Z1T']
    Z6T = gd['gas8/Z6T']
    EBRM = gd['gas8/EBRM']

    cdef double A0, RY, CONST, ElectronMass2, API, BBCONST, AM2, C, AM2EXC, CEXC, RAT, DEGV4, DEGV3, DEGV2, DEGV1
    cdef int J, I, i, j, NBREM, NDATA, NVIBV4, NVIBV2, NVIBV1, NVIBV3, NVIBH1, NVIBH2, N_IonizationD, N_IonizationF, N_IonizationF1, N_IonizationF2
    cdef int N_IonizationF3, N_IonizationF4, N_IonizationF5, N_IonizationF6, N_IonizationPP, NKSH, N_Attachment1, NDET, NTRP1, NTRP2, NTRP3, NCHD, NCHB, NHAL, NHBE
    cdef int NASIZE = 4000
    for J in range(6):
        object.AngularModel[J] = object.WhichAngularModel
        #SUPERELASTIC, V2 V1 AND HARMONIC VIBRATIONS ASSumED ISOTROPIC
        object.KIN[J] = 0
    object.KIN[6]=0.0
    object.KIN[7]=0.0
    #V4 AND V3 VIBRATIONS AAnisotropicDetectedTROPIC ( CAPITELLI-LONGO)
    object.KIN[1] = 1
    object.KIN[5] = 1
    for J in range(NBREM):
        EBRM[J] = exp(EBRM[J])
    # ANGULAR DISTRIBUTION FOR DISSOCIATIVE EXCITATION IS OKHRIMOVSKYY TYPE
    for J in range(8, object.N_Inelastic):
        object.KIN[J] = 2

    #RAT IS MOMENTUM TRANSFER TO TOTAL RATIO FOR VIBRATIONS IN THE
    #RESONANCE REGION AND ALSO FOR THE VIBRATIONS V1 AND V2 .
    #USED DIPOLE ANGULAR DISTRIBUTION FOR V3 AND V4 NEAR THRESHOLD.
    RAT = 1.0
    #BORN BETHE CONSTANTS
    A0 = 0.52917720859e-08
    RY = <float> (13.60569193)
    CONST = 1.873884e-20
    ElectronMass2 = <float> (1021997.804)
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / ElectronMass2
    #BORN BETHE VALUES FOR IONISATION
    AM2 = <float> (3.75)
    C = <float> (42.5)
    #BORN BETHE FOR EXCITATION
    AM2EXC = <float> (1.40)
    CEXC = <float> (19.0)
    #ARRAY SIZE
    NASIZE = 4000
    object.N_Ionization = 9
    object.N_Attachment = 1
    object.N_Inelastic = 34
    object.N_Null = 0
    NBREM = 25

    NDATA = 153
    NVIBV4 = 26
    NVIBV2 = 29
    NVIBV1 = 30
    NVIBV3 = 25
    NVIBH1 = 14
    NVIBH2 = 14
    N_IonizationD = 70
    N_IonizationF = 70
    N_IonizationF1 = 68
    N_IonizationF2 = 66
    N_IonizationF3 = 53
    N_IonizationF4 = 51
    N_IonizationF5 = 50
    N_IonizationF6 = 48
    N_IonizationPP = 49
    NKSH = 83
    N_Attachment1 = 6
    NDET = 9
    NTRP1 = 12
    NTRP2 = 11
    NTRP3 = 11
    NCHD = 32
    NCHB = 35
    NHAL = 34
    NHBE = 34

    #VIBRATIONAL DEGENERACY
    DEGV4 = 3.0
    DEGV2 = 2.0
    DEGV1 = 1.0
    DEGV3 = 3.0
    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27

    object.E = [0.0, 1.0, <float> (12.65), 0.0, 0.0, 0.0]
    object.E[1] = <float>(2.0) * ElectronMass / (<float> (16.0426) * AMU)
    object.IonizationEnergy[0:9] = [<float> (12.65), <float> (14.25), <float> (15.2), <float> (22.2), <float> (23.5),
                        <float> (25.2), <float> (27.0), <float> (27.9), <float> (285.0)]
    #OPAL BEATY
    cdef double SCLOBY = <float> (0.475)
    for j in range(9):
        object.EOBY[j] = object.IonizationEnergy[j] * SCLOBY

    object.EOBY[8] = object.IonizationEnergy[8] * <float> (0.63)
    for J in range(8):
        object.NC0[J] = 0.0
        object.EC0[J] = 0.0
        object.WK[J] = 0.0
        object.EFL[J] = 0.0
        object.NG1[J] = 0.0
        object.EG1[J] = 0.0
        object.NG2[J] = 0.0
        object.EG2[J] = 0.0
    #DOUBLE CHARGED, 2+ ION STATES (EXTRA ELECTRON)
    object.NC0[6] = 1
    object.EC0[6] = 6.0
    #FLUORESCENCE DATA
    object.NC0[8] = 2
    object.EC0[8] = 253
    object.WK[8] = <float> (0.0026)
    object.EFL[8] = 273
    object.NG1[8] = 1
    object.EG1[8] = 253
    object.NG2[8] = 2
    object.EG2[8] = 5

    # OFFSET ENERGY FOR IONISATION ELECTRON ANGULAR DISTRIBUTION
    for j in range(0, object.N_Ionization):
        for i in range(0, NASIZE):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break
    object.EnergyLevels = gd['gas8/EnergyLevels']

    cdef int NL = 0
    for NL in range(object.N_Inelastic):
        for i in range(4000):
            if object.EG[i] > abs(object.EnergyLevels[NL]):
                IOFFN[NL] = i
                break
    for NL in range(3):
        for i in range(46):
            object.PenningFraction[NL][i] = 0.0

    cdef double APOPV4, APOPV2, APOPGS, APOPSum, EN, GAMMA1, GAMMA2, BETA, BETA2, ElasticCrossSectionA, QMOM, PQ[3], YXJ, XNJ, YXJ1, XNJ1, A, B, X2, X1, CrossSectionSum, FAC
    cdef double XMT, CON[18], F[18], QSUP, QVIB, QDATT, QSING, QTRIP, QEXC, QTTT, QWINT, InelasticCrossSection, IonizationCrossSectionS
    cdef int FI, CONI
    F[0:18] = [<float> (0.0271), <float> (0.0442), <float> (0.0859), <float> (0.0906), <float> (0.0841),
               <float> (0.1036), <float> (0.1460), <float> (0.1548), <float> (0.1927), <float> (0.1981),
               <float> (0.1628), <float> (0.10930), <float> (0.0628), <float> (0.0297), <float> (0.0074), <float> (0.5),
               <float> (0.0045), <float> (0.0045) ]
    CON[0:18] = [<float> (1.029), <float> (1.027), <float> (1.026), <float> (1.024), <float> (1.023), <float> (1.022),
                 <float> (1.021), <float> (1.020), <float> (1.020), <float> (1.019), <float> (1.018), <float> (1.018),
                 <float> (1.017), <float> (1.016), <float> (1.016), <float> (1), <float> (1.037), <float> (1.034) ]
    #CALC LEVEL POPULATIONS
    APOPV4 = DEGV4 * exp(object.EnergyLevels[0] / object.ThermalEnergy)
    APOPV2 = DEGV2 * exp(object.EnergyLevels[2] / object.ThermalEnergy)
    APOPGS = 1.0
    APOPSum = APOPGS + APOPV4 + APOPV2
    APOPGS = 1.0 / APOPSum
    APOPV4 = APOPV4 / APOPSum
    APOPV2 = APOPV2 / APOPSum
    #  RENORMALISE GROUND STATE TO ALLOW FOR INCREASED EXCITATION X-SEC
    #  FROM EXCITED VIBRATIONAL STATE ( EXACT FOR TWICE GROUND STATE XSEC)
    APOPGS = 1.0
    object.EnergySteps = 4000
    for I in range(object.EnergySteps):
        EN = object.EG[I]
        GAMMA1 = (ElectronMass2 + 2 * EN) / ElectronMass2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA

        #USE LOG INTERPOLATION FOR ELASTIC
        if EN <= XEN[1]:
            ElasticCrossSectionA = 26.7e-16
            QMOM = 26.7e-16
            PQ[2] = 0.0
        else:
            for J in range(1, NDATA):
                if EN <= XEN[J]:
                    break
            YXJ = log(YELT[J])
            YXJ1 = log(YELT[J - 1])
            XNJ = log(XEN[J])
            XNJ1 = log(XEN[J - 1])
            A = (YXJ - YXJ1) / (XNJ - XNJ1)
            B = (XNJ1 * YXJ - XNJ * YXJ1) / (XNJ1 - XNJ)
            ElasticCrossSectionA = exp(A * log(EN) + B) * 1.e-16
            YXJ = log(YELM[J])
            YXJ1 = log(YELM[J - 1])
            A = (YXJ - YXJ1) / (XNJ - XNJ1)
            B = (XNJ1 * YXJ - XNJ * YXJ1) / (XNJ1 - XNJ)
            QMOM = exp(A * log(EN) + B) * 1.e-16
            YXJ = log(YEPS[J])
            YXJ1 = log(YEPS[J - 1])
            A = (YXJ - YXJ1) / (XNJ - XNJ1)
            B = (XNJ1 * YXJ - XNJ * YXJ1) / (XNJ1 - XNJ)
            PQ[2] = exp(A * log(EN) + B)
            #  EPSILON =1-YEPS
            PQ[2] = 1.0e0 - PQ[2]
        PQ[1] = 0.5 + (ElasticCrossSectionA - QMOM) / ElasticCrossSectionA
        PQ[0] = 0.5
        object.PEElasticCrossSection[1][I] = PQ[object.WhichAngularModel]
        object.Q[1][I] = ElasticCrossSectionA
        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM

        # IONISATION TO CH4 +
        object.IonizationCrossSection[0][I] = 0.0
        object.PEIonizationCrossSection[0][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[0][I] = 0
        if EN >= object.IonizationEnergy[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationF, YINF, XINF, BETA2, <float> (0.4594), CONST, object.DEN[I],
                                                 C, AM2)
            if EN > 2 * object.IonizationEnergy[0]:
                object.PEIonizationCrossSection[0][I] = object.PEElasticCrossSection[1][I - IOFFION[0]]

        # IONISATION TO CH3 +
        object.IonizationCrossSection[1][I] = 0.0
        object.PEIonizationCrossSection[1][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[1][I] = 0
        if EN >= object.IonizationEnergy[1]:
            object.IonizationCrossSection[1][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationF1, YINF1, XINF1, BETA2, <float> (0.3716), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[1]:
                object.PEIonizationCrossSection[1][I] = object.PEElasticCrossSection[1][I - IOFFION[1]]

        # IONISATION TO CH2 +
        object.IonizationCrossSection[2][I] = 0.0
        object.PEIonizationCrossSection[2][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[2][I] = 0
        if EN >= object.IonizationEnergy[2]:
            object.IonizationCrossSection[2][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationF2, YINF2, XINF2, BETA2, <float> (0.06312), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[2]:
                object.PEIonizationCrossSection[2][I] = object.PEElasticCrossSection[1][I - IOFFION[2]]

        # IONISATION TO H +
        object.IonizationCrossSection[3][I] = 0.0
        object.PEIonizationCrossSection[3][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[3][I] = 0
        if EN >= object.IonizationEnergy[3]:
            object.IonizationCrossSection[3][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationF3, YINF3, XINF3, BETA2, <float> (0.0664), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[3]:
                object.PEIonizationCrossSection[3][I] = object.PEElasticCrossSection[1][I - IOFFION[3]]

        # IONISATION TO CH +
        object.IonizationCrossSection[4][I] = 0.0
        object.PEIonizationCrossSection[4][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[4][I] = 0
        if EN >= object.IonizationEnergy[4]:
            object.IonizationCrossSection[4][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationF4, YINF4, XINF4, BETA2, <float> (0.02625), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[4]:
                object.PEIonizationCrossSection[4][I] = object.PEElasticCrossSection[1][I - IOFFION[4]]

        # IONISATION TO C +
        object.IonizationCrossSection[5][I] = 0.0
        object.PEIonizationCrossSection[5][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[5][I] = 0
        if EN >= object.IonizationEnergy[5]:
            object.IonizationCrossSection[5][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationF5, YINF5, XINF5, BETA2, <float> (0.00798), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[5]:
                object.PEIonizationCrossSection[5][I] = object.PEElasticCrossSection[1][I - IOFFION[5]]

        # IONISATION TO DOUBLY POSITIVE CHARGED FINAL STATES
        object.IonizationCrossSection[6][I] = 0.0
        object.PEIonizationCrossSection[6][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[6][I] = 0
        if EN >= object.IonizationEnergy[6]:
            object.IonizationCrossSection[6][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationPP, YINPP, XINPP, BETA2, <float> (0.0095969), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[6]:
                object.PEIonizationCrossSection[6][I] = object.PEElasticCrossSection[1][I - IOFFION[6]]

        # IONISATION TO H2+
        object.IonizationCrossSection[7][I] = 0.0
        object.PEIonizationCrossSection[7][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[7][I] = 0
        if EN >= object.IonizationEnergy[7]:
            object.IonizationCrossSection[7][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationF6, YINF6, XINF6, BETA2, <float> (0.00523), CONST,
                                                 object.DEN[I], C, AM2)
            if EN > 2 * object.IonizationEnergy[7]:
                object.PEIonizationCrossSection[7][I] = object.PEElasticCrossSection[1][I - IOFFION[7]]

        # CALCULATE K-SHELL IONISATION
        object.IonizationCrossSection[8][I] = 0.0
        object.PEIonizationCrossSection[8][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[8][I] = 0
        if EN >= object.IonizationEnergy[8]:
            object.IonizationCrossSection[8][I] = GasUtil.CALIonizationCrossSectionREG(EN, NKSH, YKSH, XKSH)
            if EN > 2 * object.IonizationEnergy[8]:
                object.PEIonizationCrossSection[8][I] = object.PEElasticCrossSection[1][I - IOFFION[8]]

        # CORRECT IONISATION FOR SPLIT INTO K-SHELL
        QSUN = 0.0
        for i in range(9):
            CrossSectionSum += object.IonizationCrossSection[i][I]
        if CrossSectionSum != 0:
            FAC = (CrossSectionSum - object.IonizationCrossSection[8][I])/ CrossSectionSum
            for i in range(9):
                object.IonizationCrossSection[i][I] = object.IonizationCrossSection[i][I] * FAC

        #ATTACHMENT
        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = 0.0
        if EN >= XATT[0]:
            object.Q[3][I] = GasUtil.CALIonizationCrossSection(EN, N_Attachment1, YATT, XATT)
            object.AttachmentCrossSection[0][I] = object.Q[3][I]
        # COUNTING IONISATION
        object.Q[4][I] = 0.0

        object.Q[5][I] = 0.0

        #V4  SUPERELASTIC ISOTROPIC
        object.InelasticCrossSectionPerGas[0][I] = 0.0
        object.PEInelasticCrossSectionPerGas[0][I] = 0.5
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[0][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIBV4, YVBV4, XVBV4, APOPV4, object.EnergyLevels[1], DEGV4, object.EnergyLevels[0],
                                                   <float> (0.076))
        #V4 AAnisotropicDetectedTROPIC
        object.InelasticCrossSectionPerGas[1][I] = 0.0
        object.PEInelasticCrossSectionPerGas[1][I] = 0.5
        if EN > object.EnergyLevels[1]:
            object.InelasticCrossSectionPerGas[1][I] = GasUtil.CALInelasticCrossSectionPerGasVAAnisotropicDetected(EN, NVIBV4, YVBV4, XVBV4, object.EnergyLevels[1], APOPGS, RAT,
                                                    <float> (0.076))
            #RATIO OF MT TO TOTAL X-SECT FOR RESONANCE PART =RAT
            XMT = GasUtil.CALXMTVAAnisotropicDetected(EN, NVIBV4, YVBV4, XVBV4, object.EnergyLevels[1], APOPGS, RAT, <float>(0.076))
            object.PEInelasticCrossSectionPerGas[1][I] = 0.5 + (object.InelasticCrossSectionPerGas[1][I] - XMT) / object.InelasticCrossSectionPerGas[1][I]

        #V2  SUPERELASTIC ISOTROPIC
        object.InelasticCrossSectionPerGas[2][I] = 0.0
        object.PEInelasticCrossSectionPerGas[2][I] = 0.5
        if EN > 0.0:
            object.InelasticCrossSectionPerGas[2][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIBV2, YVBV2, XVBV2, APOPV2, object.EnergyLevels[3], DEGV2, object.EnergyLevels[0],
                                                  0.0)

        #V2  ISOTROPIC
        object.InelasticCrossSectionPerGas[3][I] = 0.0
        object.PEInelasticCrossSectionPerGas[3][I] = 0.5
        if EN > object.EnergyLevels[3]:
            object.InelasticCrossSectionPerGas[3][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIBV2, YVBV2, XVBV2, APOPGS, 0, 1, object.EnergyLevels[0], 0.0)

        #V1  ISOTROPIC
        object.InelasticCrossSectionPerGas[4][I] = 0.0
        object.PEInelasticCrossSectionPerGas[4][I] = 0.5
        if EN > object.EnergyLevels[4]:
            object.InelasticCrossSectionPerGas[4][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIBV1, YVBV1, XVBV1, 1, 0, 1, object.EnergyLevels[0], 0.0)

        #V3 AAnisotropicDetectedTROPIC
        object.InelasticCrossSectionPerGas[5][I] = 0.0
        object.PEInelasticCrossSectionPerGas[5][I] = 0.5
        if EN > object.EnergyLevels[5]:
            object.InelasticCrossSectionPerGas[5][I] = GasUtil.CALInelasticCrossSectionPerGasVAAnisotropicDetected(EN, NVIBV3, YVBV3, XVBV3, object.EnergyLevels[5], 1, RAT, <float> (0.076))
            #RATIO OF MT TO TOTAL X-SECT FOR RESONANCE PART =RAT
            XMT = GasUtil.CALXMTVAAnisotropicDetected(EN, NVIBV3, YVBV3, XVBV3, object.EnergyLevels[5], 1, RAT, <float> (0.076))
            object.PEInelasticCrossSectionPerGas[5][I] = 0.5 + (object.InelasticCrossSectionPerGas[5][I] - XMT) / object.InelasticCrossSectionPerGas[5][I]

        #VIBRATION HARMONICS 1 ISOTROPIC
        object.InelasticCrossSectionPerGas[6][I] = 0.0
        object.PEInelasticCrossSectionPerGas[6][I] = 0.5
        if EN > object.EnergyLevels[6]:
            object.InelasticCrossSectionPerGas[6][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIBH1, YVBH1, XVBH1, 1, 0, 1, object.EnergyLevels[0], 0.0)

        #VIBRATION HARMONICS 2 ISOTROPIC
        object.InelasticCrossSectionPerGas[7][I] = 0.0
        object.PEInelasticCrossSectionPerGas[7][I] = 0.5
        if EN > object.EnergyLevels[7]:
            object.InelasticCrossSectionPerGas[7][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIBH2, YVBH2, XVBH2, 1, 0, 1, object.EnergyLevels[0], 0.0)

        #TRIPLET DISSOCIATION 7.5EV
        object.InelasticCrossSectionPerGas[8][I] = 0.0
        object.PEInelasticCrossSectionPerGas[8][I] = 0.0
        if EN > object.EnergyLevels[8]:
            object.InelasticCrossSectionPerGas[8][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NTRP1, YTR1, XTR1, 1, 0, 1, object.EnergyLevels[0], 0.0)
            if EN > 3 * object.EnergyLevels[8]:
                object.PEInelasticCrossSectionPerGas[8][I] = object.PEElasticCrossSection[1][I - IOFFN[8]]

        #ATTACHMENT - DEATTACHMENT RESONANCE VIA H- AT 9.8EV RESONANCE
        object.InelasticCrossSectionPerGas[9][I] = 0.0
        object.PEInelasticCrossSectionPerGas[9][I] = 0.0
        if EN > object.EnergyLevels[9]:
            object.InelasticCrossSectionPerGas[9][I] = GasUtil.CALIonizationCrossSection(EN, NDET, YDET, XDET)
            if EN > 3 * object.EnergyLevels[9] and object.InelasticCrossSectionPerGas[9][I] != 0.0:
                object.PEInelasticCrossSectionPerGas[9][I] = object.PEElasticCrossSection[1][I - IOFFN[9]]

        #TRIPLET DISSOCIATION 8.5EV
        object.InelasticCrossSectionPerGas[10][I] = 0.0
        object.PEInelasticCrossSectionPerGas[10][I] = 0.0
        if EN > object.EnergyLevels[10]:
            object.InelasticCrossSectionPerGas[10][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NTRP2, YTR2, XTR2, 1, 0, 1, object.EnergyLevels[0], 0.0)
            if EN > 3 * object.EnergyLevels[10]:
                object.PEInelasticCrossSectionPerGas[10][I] = object.PEElasticCrossSection[1][I - IOFFN[10]]

        #SINGLET DISSOCIATION AT 8.75+FI*0.5 EV USE BEF SCALING WITH F[J]
        FI = 0
        CONI = 0
        for J in range(11, 14):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (<float>(4.0) * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / <float>(2.0)) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2]) * CON[CONI]
                if object.InelasticCrossSectionPerGas[J][I]<0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 3 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        #TRIPLET DISSOCIATION 10.0EV
        object.InelasticCrossSectionPerGas[14][I] = 0.0
        object.PEInelasticCrossSectionPerGas[14][I] = 0.0
        if EN > object.EnergyLevels[14]:
            object.InelasticCrossSectionPerGas[14][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NTRP3, YTR3, XTR3, 1, 0, 1, object.EnergyLevels[0], 0.0)
            if EN > 3 * object.EnergyLevels[14]:
                object.PEInelasticCrossSectionPerGas[14][I] = object.PEElasticCrossSection[1][I - IOFFN[14]]

        #SINGLET DISSOCIATION AT 8.75+FI*0.5 EV USE BEF SCALING WITH F[J]
        for J in range(15, 22):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (<float>(4.0) * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / <float>(2.0)) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2]) * CON[CONI]

                if object.InelasticCrossSectionPerGas[J][I]<0.0:
                 object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 3 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        #DISSOCIATIVE EXC TO STATES DECAYING VIA CH(A2DELTA TO G.S.)
        object.InelasticCrossSectionPerGas[22][I] = 0.0
        object.PEInelasticCrossSectionPerGas[22][I] = 0.0
        if EN > object.EnergyLevels[22]:
            object.InelasticCrossSectionPerGas[22][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NCHD, YCHD, XCHD, 1) * 100
            if EN > 3 * object.EnergyLevels[22]:
                object.PEInelasticCrossSectionPerGas[22][I] = object.PEElasticCrossSection[1][I - IOFFN[22]]

        #DISSOCIATIVE EXC TO STATES DECAYING VIA CH(B2SIGMA- TO G.S.)
        object.InelasticCrossSectionPerGas[23][I] = 0.0
        object.PEInelasticCrossSectionPerGas[23][I] = 0.0
        if EN > object.EnergyLevels[23]:
            object.InelasticCrossSectionPerGas[23][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NCHB, YCHB, XCHB, 1) * 100
            if EN > 3 * object.EnergyLevels[23]:
                object.PEInelasticCrossSectionPerGas[23][I] = object.PEElasticCrossSection[1][I - IOFFN[23]]

        #SINGLET DISSOCIATION AT 8.75+FI*0.5 EV USE BEF SCALING WITH F[J]
        for J in range(24, 30):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (<float>(4.0) * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / <float>(2.0)) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2]) * CON[CONI]

                if object.InelasticCrossSectionPerGas[J][I]<0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 3 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        #DISSOCIATIVE EXC TO STATES DECAYING VIA H(ALPHA)
        object.InelasticCrossSectionPerGas[30][I] = 0.0
        object.PEInelasticCrossSectionPerGas[30][I] = 0.0
        if EN > object.EnergyLevels[30]:
            object.InelasticCrossSectionPerGas[30][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NHAL, YHAL, XHAL, 1) * 100
            if EN > 3 * object.EnergyLevels[30]:
                object.PEInelasticCrossSectionPerGas[30][I] = object.PEElasticCrossSection[1][I - IOFFN[30]]

        #DISSOCIATIVE EXC TO STATES DECAYING VIA H(BETA)
        object.InelasticCrossSectionPerGas[31][I] = 0.0
        object.PEInelasticCrossSectionPerGas[31][I] = 0.0
        if EN > object.EnergyLevels[31]:
            object.InelasticCrossSectionPerGas[31][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NHBE, YHBE, XHBE, 1) * 100
            if EN > 3 * object.EnergyLevels[31]:
                object.PEInelasticCrossSectionPerGas[31][I] = object.PEElasticCrossSection[1][I - IOFFN[31]]

        #SINGLET DISSOCIATION AT 8.75+FI*0.5 EV USE BEF SCALING WITH F[J]
        for J in range(32, 34):
            object.InelasticCrossSectionPerGas[J][I] = 0.0
            object.PEInelasticCrossSectionPerGas[J][I] = 0.0
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (<float>(4.0) * object.EnergyLevels[J])) - BETA2 - object.DEN[
                    I] / <float>(2.0)) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2]) * CON[CONI]
                if object.InelasticCrossSectionPerGas[J][I]<0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 3 * object.EnergyLevels[J]:
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]
            FI += 1
            CONI += 1

        #LOAD BREMSSTRAHLUNG X-SECTIONS
        object.InelasticCrossSectionPerGas[34][I] = 0.0
        object.InelasticCrossSectionPerGas[35][I] = 0.0
        if EN > 1000:
            object.InelasticCrossSectionPerGas[34][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z6T, EBRM) * 1e-8
            object.InelasticCrossSectionPerGas[35][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z1T, EBRM) * 4e-8

        #skipped the QSUP,QVIB,QDATT,QSING,QTRIP,QEXC,QTTT,QWINT,InelasticCrossSection,IonizationCrossSectionS as they are not used later one.

    for J in range(object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break


    return
