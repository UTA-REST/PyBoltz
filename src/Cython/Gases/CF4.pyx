from libc.math cimport sin, cos, acos,asin, log,sqrt,exp,pow
cimport libc.math
import numpy as np
cimport numpy as np
import sys
from Gas cimport Gas
from cython.parallel import prange

sys.path.append('../hdf5_python')
import cython
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.fast_getattr(True)

cdef void Gas1(Gas* object):
    """
    This function is used to calculate the needed momentum cross sections for CF4 gas.
    """
    gd = np.load('gases.npy').item()
    cdef int i = 0
    object.EnergyLevels = gd['gas1/EnergyLevels']
    cdef double EOBY[12]
    cdef double PQ[3]
    # EnergyLevels=[0 for x in range(250)]#<=== input to this function
    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27
    object.E = [0.0, 1.0, <float>(15.9), 0.0, 0.0, 0.0]
    object.E[1] = <float>(2.0) * ElectronMass / (<float>(88.0043) * AMU)
    object.NC0[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0]
    object.EC0[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 253.0, 625.2]
    cdef double WKLM[12]
    WKLM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, <float>(0.0026), <float>(0.01)]
    object.WK[0:12]=WKLM
    object.EFL[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 273.0, 668.0]
    object.NG1[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 2]
    object.EG1[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 253.0, 625.2]
    object.NG2[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0,     0.0, 0.0, 0.0, 0.0, 0.0, 1, 1]
    object.EG2[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0]
    object.IonizationEnergy[0:12] = [<float>(15.7), <float>(21.47), <float>(29.14), <float>(34.5), <float>(34.77), 36.0, 40.0, 41.0, 43.0, 63.0, 285.0, <float>(685.4)]
    cdef int IOFFION[12]
    cdef int IOFFN[46]


    cdef double ElectronMass2 = <float>(1021997.804)
    cdef double API = acos(-1)
    cdef double A0 = 0.52917720859e-8
    cdef double RY = <float>(13.60569193)
    cdef double BBCONST = 16.0 * API * A0 * A0 * RY * RY / ElectronMass2

    # BORN BETHE VALUES FOR IONISATION
    cdef double CONST = 1.873884e-20
    cdef double AM2 = <float>(9.5)
    cdef double C = <float>(100.9)

    # number of array elements
    cdef int NCF3 = 37
    cdef int NCF2 = 31
    cdef int NCF1 = 28
    cdef int NCF32 = 25
    cdef int NCF0 = 27
    cdef int NC0F = 27
    cdef int NCF22 = 25
    cdef int NCF = 22
    cdef int NCFF = 24
    cdef int NCF2F = 25
    cdef int NCF3F = 26
    object.N_Ionization = 12
    object.N_Attachment = 1
    object.N_Inelastic = 46
    object.N_Null = 0
    cdef int NASIZE = 4000
    cdef int NBREM = 25
    object.EnergySteps = 4000
    for i in range(0, 6):
        object.AngularModel[i] = object.WhichAngularModel
    # ASSumE CAPITELLI LONGO TYPE OF ANGULAR DISTRIBUTION FOR
    # ALL VIBRATIONAL LEVELS AND THE Sum OF HIGHER HARMONICS
    for i in range(0, 10):
        object.KIN[i] = 1
    # ANGULAR DISTRIBUTION FOR DISS.EXCITATION IS GIVEN BY OKHRIMOVSKKY
    for i in range(10, object.N_Inelastic):
        object.KIN[i] = object.WhichAngularModel
    # RATIO OF MOMENTUM TRANSFER TO TOTAL X-SEC FOR RESONANCE
    # PART OF VIBRATIONAL X-SECTIONS
    cdef double RAT = <float>(0.75)
    cdef int NDATA = 163
    cdef int NVBV4 = 11
    cdef int NVBV1 = 11
    cdef int NVBV3 = 11
    cdef int NVIB5 = 12
    cdef int NVIB6 = 12
    cdef int N_Attachment1 = 11
    cdef int NTR1 = 12
    cdef int NTR2 = 11
    cdef int NTR3 = 11
    cdef int NKSHC = 81
    cdef int NKSHF = 79
    cdef int J = 0
    # OPAL BEATY IONISATION ENERGY SPLITTING
    for i in range(0, 10):
        object.EOBY[i] = <float>(0.58) * object.IonizationEnergy[i]

    object.EOBY[10] = 210.0
    object.EOBY[11] = 510.0

    # skipped ISHELL and LEGAS, as they are not used in any calculation
    cdef int j = 0
    for j in range(0, object.N_Ionization):
        for i in range(0, NASIZE):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break

    # OFFSET ENERGY FOR DISSOCIATION ANGULAR DISTRIBUTION
    cdef int NL = 10
    for NL in range(10, 46):
        for i in range(0, NASIZE):
            if object.EG[i] > abs(object.EnergyLevels[NL]):
                IOFFN[NL] = i
                break

    # ENTER PENN_InelasticG TRANSFER FRACTION FOR EACH LEVEL
    # ONLY DISSOCIATION X-SECTION (LEVEL 11) HAS ENOUGH ENERGY TO GIVE
    # POSSIBLE PENN_InelasticG TRANSFER
    for NL in range(3):
        for i in range(46):
            object.PenningFraction[NL][i]=0.0
    # PENN_InelasticG TRANSFER FRACTION FOR LEVEL 11
    object.PenningFraction[0][45] = 0.0
    # PENN_InelasticG TRANSFER DISTANCE IN MICRONS
    object.PenningFraction[1][45] = 1.0
    # PENN_InelasticG TRANSFER TIME IN PICOSECONDS
    object.PenningFraction[2][45] = 1.0

    # PRINT

    # VIBRATIONAL DEGENERACY
    cdef float DEGV4 = 3.0,DEGV3 = 3.0,DEGV2 = 2.0,DEGV1 = 1.0

    # CALC VIB LEVEL POPULATIONS
    cdef double APOPV2 = DEGV2 * exp(object.EnergyLevels[0] / object.ThermalEnergy)
    cdef double APOPV4 = DEGV4 * exp(object.EnergyLevels[2] / object.ThermalEnergy)
    cdef double APOPV1 = DEGV1 * exp(object.EnergyLevels[4] / object.ThermalEnergy)
    cdef double APOPV3 = DEGV3 * exp(object.EnergyLevels[6] / object.ThermalEnergy)
    cdef double APOPGS = 1.0
    cdef double APOPSum = APOPGS + APOPV2 + APOPV4 + APOPV1 + APOPV3
    APOPGS = 1.0 / APOPSum
    APOPV2 = APOPV2 / APOPSum
    APOPV4 = APOPV4 / APOPSum
    APOPV1 = APOPV1 / APOPSum
    APOPV3 = APOPV3 / APOPSum
    cdef double XEN[163], YELM[163], YELT[163], YEPS[163], XVBV4[11], YVBV4[11], XVBV1[11], YVBV1[11], XVBV3[11], YVBV3[11], XVIB5[12]
    cdef double YVIB5[12],XVIB6[12],YVIB6[12],XTR1[12],YTR1[12],XTR2[11],YTR2[11],XTR3[11],YTR3[11],XCF3[37],YCF3[37]
    cdef double XCF2[31],YCF2[31],XCF1[28],YCF1[28],XCF32[25],YCF32[25],XCF0[27],YCF0[27],XCF22[25],YCF22[25],XCF[22]
    cdef double YCF[22],XCFF[24],XATT[11],YATT[11],XKSHC[81],YKSHC[81],XKSHF[79],YKSHF[79],XC0F[27],YC0F[27],XCF2F[25],YCF2F[25]
    cdef double YCFF[24],XCF3F[26],YCF3F[26]
    XEN = gd['gas1/XEN']
    YELM = gd['gas1/YELM']
    YELT = gd['gas1/YELT']
    YEPS = gd['gas1/YEPS']
    XCF3 = gd['gas1/XCF3']
    YCF3 = gd['gas1/YCF3']
    XCF2 = gd['gas1/XCF2']
    YCF2 = gd['gas1/YCF2']
    XCF1 = gd['gas1/XCF1']
    YCF1 = gd['gas1/YCF1']
    XC0F = gd['gas1/XC0F']
    YC0F = gd['gas1/YC0F']
    XCF3F = gd['gas1/XCF3F']
    YCF3F = gd['gas1/YCF3F']
    XCF2F = gd['gas1/XCF2F']
    YCF2F = gd['gas1/YCF2F']
    XCF0 = gd['gas1/XCF0']
    YCF0 = gd['gas1/YCF0']
    XCF32 = gd['gas1/XCF32']
    YCF32 = gd['gas1/YCF32']
    XCF22 = gd['gas1/XCF22']
    YCF22 = gd['gas1/YCF22']
    XCFF = gd['gas1/XCFF']
    YCFF = gd['gas1/YCFF']
    XCF = gd['gas1/XCF']
    YCF = gd['gas1/YCF']
    XKSHC = gd['gas1/XKSHC']
    YKSHC = gd['gas1/YKSHC']
    XKSHF = gd['gas1/XKSHF']
    YKSHF = gd['gas1/YKSHF']
    XATT = gd['gas1/XATT']
    YATT = gd['gas1/YATT']
    XVBV4 = gd['gas1/XVBV4']
    YVBV4 = gd['gas1/YVBV4']
    XVBV1 = gd['gas1/XVBV1']
    YVBV1 = gd['gas1/YVBV1']
    XVBV3 = gd['gas1/XVBV3']
    YVBV3 = gd['gas1/YVBV3']
    XVIB5 = gd['gas1/XVIB5']
    YVIB5 = gd['gas1/YVIB5']
    XVIB6 = gd['gas1/XVIB6']
    YVIB6 = gd['gas1/YVIB6']
    XTR1 = gd['gas1/XTR1']
    XTR2 = gd['gas1/XTR2']
    YTR1 = gd['gas1/YTR1']
    YTR2 = gd['gas1/YTR2']
    XTR3 = gd['gas1/XTR3']
    YTR3 = gd['gas1/YTR3']
    # RENORMALISE GROUND STATE TO ALLOW FOR EXCITATION X-SEC FROM
    # EXCITED VIBRATIONAL STATES (EXACT APPROX IF THE HOT TRANSITIONS HAVE
    # EQUAL X-SEC TO THE GROUND STATE TRANSITIONS)
    APOPGS = 1.0

    cdef double EN,GAMMA1,GAMMA2,BETA,BETA2,A,B,QMOM,ElasticCrossSectionA,X1,X2,EFAC,ELF,ADIP,FWD,BCK
    # EN=-EnergyStep/2.0  #EnergyStep is function input
    for i in range(object.EnergySteps):
        EN = object.EG[i]
        # EN=EN+EnergyStep
        GAMMA1 = (ElectronMass2 + 2.0 * EN) / ElectronMass2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.00 - 1.00 / GAMMA2)
        BETA2 = BETA * BETA

        j = 0
        for j in range(1, NDATA):
            if EN <= XEN[j]:
                break

        A = (YELM[j] - YELM[j - 1]) / (XEN[j] - XEN[j - 1])
        B = (XEN[j - 1] * YELM[j] - XEN[j] * YELM[j - 1]) / (XEN[j - 1] - XEN[j])
        QMOM = (A * EN + B) * 1e-16

        A = (YELT[j] - YELT[j - 1]) / (XEN[j] - XEN[j - 1])
        B = (XEN[j - 1] * YELT[j] - XEN[j] * YELT[j - 1]) / (XEN[j - 1] - XEN[j])
        ElasticCrossSectionA = (A * EN + B) * 1e-16

        A = (YEPS[j] - YEPS[j - 1]) / (XEN[j] - XEN[j - 1])
        B = (XEN[j - 1] * YEPS[j] - XEN[j] * YEPS[j - 1]) / (XEN[j - 1] - XEN[j])
        PQ = [0.5, 0.5 + (ElasticCrossSectionA - QMOM) / ElasticCrossSectionA, 1 - (A * EN + B)]
        # ^^^^^^EPS CORRECTED FOR 1-EPS^^^^^^^^
        object.PEElasticCrossSection[1][i] = PQ[object.WhichAngularModel]
        object.Q[1][i] = ElasticCrossSectionA

        X2 = 1.0 / BETA2
        X1 = X2 * log(BETA2 / (1.0 - BETA2)) - 1.0
        # DISSOCIATIVE IONISATION
        # ION  =  CF3 +
        if object.WhichAngularModel == 0:
            object.Q[1][i] = QMOM
        object.IonizationCrossSection[0][i] = 0.0
        object.PEIonizationCrossSection[0][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[0][i] = 0

        if EN > object.IonizationEnergy[0]:
            if EN <= XCF3[NCF3 - 1]:  # <<<check if -1 or not
                j = 0
                for j in range(1, NCF3):
                    if EN <= XCF3[j]:
                        break
                A = (YCF3[j] - YCF3[j - 1]) / (XCF3[j] - XCF3[j - 1])
                B = (XCF3[j - 1] * YCF3[j] - XCF3[j] * YCF3[j - 1]) / (XCF3[j - 1] - XCF3[j])
                object.IonizationCrossSection[0][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF3([NCF3] EV
                object.IonizationCrossSection[0][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2.0) + C * X2) * <float>(0.7344)
            if EN > 2.0 * object.IonizationEnergy[0]:
                object.PEIonizationCrossSection[0][i] = object.PEElasticCrossSection[1][(i - IOFFION[0])]

        # ION = CF2 +
        object.IonizationCrossSection[1][i] = 0.0
        object.PEIonizationCrossSection[1][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[1][i] = 0.0

        if EN > object.IonizationEnergy[1]:
            if EN <= XCF2[NCF2 - 1]:
                j = 0
                for j in range(1, NCF2):
                    if EN <= XCF2[j]:
                        break
                A = (YCF2[j] - YCF2[j - 1]) / (XCF2[j] - XCF2[j - 1])
                B = (XCF2[j - 1] * YCF2[j] - XCF2[j] * YCF2[j - 1]) / (XCF2[j - 1] - XCF2[j])
                object.IonizationCrossSection[1][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF2[NCF2] EV
                object.IonizationCrossSection[1][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2.0) + C * X2) * <float>(0.0534)
            if EN > 2.0 * object.IonizationEnergy[1]:
                object.PEIonizationCrossSection[1][i] = object.PEElasticCrossSection[1][(i - IOFFION[1])]

        #  ION = CF +
        object.IonizationCrossSection[2][i] = 0.0
        object.PEIonizationCrossSection[2][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[2][i] = 0.0

        if EN > object.IonizationEnergy[2]:
            if EN <= XCF1[NCF1 - 1]:
                j = 0
                for j in range(1, NCF1):
                    if EN <= XCF1[j]:
                        break
                A = (YCF1[j] - YCF1[j - 1]) / (XCF1[j] - XCF1[j - 1])
                B = (XCF1[j - 1] * YCF1[j] - XCF1[j] * YCF1[j - 1]) / (XCF1[j - 1] - XCF1[j])
                object.IonizationCrossSection[2][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF1[NCF1] EV
                X2 = 1 / BETA2
                X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                object.IonizationCrossSection[2][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2.0) + C * X2) * <float>(0.0386)
            if EN > 2.0 * object.IonizationEnergy[2]:
                object.PEIonizationCrossSection[2][i] = object.PEElasticCrossSection[1][(i - IOFFION[2])]

        # ION = F +
        object.IonizationCrossSection[3][i] = 0.0
        object.PEIonizationCrossSection[3][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[3][i] = 0.0

        if EN > object.IonizationEnergy[3]:
            if EN <= XC0F[NC0F - 1]:
                j = 0
                for j in range(1, NC0F):
                    if EN <= XC0F[j]:
                        break
                A = (YC0F[j] - YC0F[j - 1]) / (XC0F[j] - XC0F[j - 1])
                B = (XC0F[j - 1] * YC0F[j] - XC0F[j] * YC0F[j - 1]) / (XC0F[j - 1] - XC0F[j])
                object.IonizationCrossSection[3][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XC0F[NC0F] EV
                X2 = 1 / BETA2
                X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                object.IonizationCrossSection[3][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2.0) + C * X2) * <float>(0.0799)
            if EN > 2.0 * object.IonizationEnergy[3]:
                object.PEIonizationCrossSection[3][i] = object.PEElasticCrossSection[1][(i - IOFFION[3])]

        # ION = C +
        object.IonizationCrossSection[4][i] = 0.0
        object.PEIonizationCrossSection[4][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[4][i] = 0.0

        if EN > object.IonizationEnergy[4]:
            if EN <= XCF0[NCF0 - 1]:
                j = 0
                for j in range(1, NCF0):
                    if EN <= XCF0[j]:
                        break
                A = (YCF0[j] - YCF0[j - 1]) / (XCF0[j] - XCF0[j - 1])
                B = (XCF0[j - 1] * YCF0[j] - XCF0[j] * YCF0[j - 1]) / (XCF0[j - 1] - XCF0[j])
                object.IonizationCrossSection[4][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF0[NCF0] EV
                object.IonizationCrossSection[4][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2.0) + C * X2) * <float>(0.0422)
            if EN > 2.0 * object.IonizationEnergy[4]:
                object.PEIonizationCrossSection[4][i] = object.PEElasticCrossSection[1][(i - IOFFION[4])]

        # DOUBLE IONS  CF3 +  AND F +
        object.IonizationCrossSection[5][i] = 0.0
        object.PEIonizationCrossSection[5][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[5][i] = 0.0

        if EN > object.IonizationEnergy[5]:
            if EN <= XCF3F[NCF3F - 1]:
                j = 0
                for j in range(1, NCF3F):
                    if EN <= XCF3F[j]:
                        break
                A = (YCF3F[j] - YCF3F[j - 1]) / (XCF3F[j] - XCF3F[j - 1])
                B = (XCF3F[j - 1] * YCF3F[j] - XCF3F[j] * YCF3F[j - 1]) / (XCF3F[j - 1] - XCF3F[j])
                object.IonizationCrossSection[5][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF3F[NCF3F] EV
                object.IonizationCrossSection[5][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2.0) + C * X2) * <float>(0.0058)
            if EN > 2.0 * object.IonizationEnergy[5]:
                object.PEIonizationCrossSection[5][i] = object.PEElasticCrossSection[1][(i - IOFFION[5])]
        # DOUBLE IONS  CF2 +  AND F +
        object.IonizationCrossSection[6][i] = 0.0
        object.PEIonizationCrossSection[6][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[6][i] = 0.0

        if EN > object.IonizationEnergy[6]:
            if EN <= XCF2F[NCF2F - 1]:
                j = 0
                for j in range(1, NCF2F):
                    if EN <= XCF2F[j]:
                        break
                A = (YCF2F[j] - YCF2F[j - 1]) / (XCF2F[j] - XCF2F[j - 1])
                B = (XCF2F[j - 1] * YCF2F[j] - XCF2F[j] * YCF2F[j - 1]) / (XCF2F[j - 1] - XCF2F[j])
                object.IonizationCrossSection[6][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF2F[NCF2F] EV
                object.IonizationCrossSection[6][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2.0) + C * X2) * <float>(0.0073)
            if EN > 2.0 * object.IonizationEnergy[6]:
                object.PEIonizationCrossSection[6][i] = object.PEElasticCrossSection[1][(i - IOFFION[6])]

        # DOUBLE CHARGED ION  CF3 ++
        object.IonizationCrossSection[7][i] = 0.0
        object.PEIonizationCrossSection[7][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[7][i] = 0.0

        if EN > object.IonizationEnergy[7]:
            if EN <= XCF32[NCF32 - 1]:
                j = 0
                for j in range(1, NCF32):
                    if EN <= XCF32[j]:
                        break
                A = (YCF32[j] - YCF32[j - 1]) / (XCF32[j] - XCF32[j - 1])
                B = (XCF32[j - 1] * YCF32[j] - XCF32[j] * YCF32[j - 1]) / (XCF32[j - 1] - XCF32[j])
                object.IonizationCrossSection[7][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF32[NCF32] EV
                X2 = 1 / BETA2
                X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                object.IonizationCrossSection[7][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * <float>(0.0031)
            if EN > 2 * object.IonizationEnergy[7]:
                object.PEIonizationCrossSection[7][i] = object.PEElasticCrossSection[1][(i - IOFFION[7])]

        # DOUBLE CHARGED ION  CF2 ++
        # ADD INTO CF3 ++
        if EN > XCF22[0]:
            if EN <= XCF22[NCF22 - 1]:
                j = 0
                for j in range(1, NCF22):
                    if EN <= XCF22[j]:
                        break
                A = (YCF22[j] - YCF22[j - 1]) / (XCF22[j] - XCF22[j - 1])
                B = (XCF22[j - 1] * YCF22[j] - XCF22[j] * YCF22[j - 1]) / (XCF22[j - 1] - XCF22[j])
                object.IonizationCrossSection[7][i] = object.IonizationCrossSection[7][i] + (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF22[NCF22] EV
                X2 = 1 / BETA2
                X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                object.IonizationCrossSection[7][i] = object.IonizationCrossSection[7][i] + CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * <float>(0.0077)

        # DOUBLE IONS    CF +  AND F +
        object.IonizationCrossSection[8][i] = 0.0
        object.PEIonizationCrossSection[8][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[8][i] = 0.0

        if EN > object.IonizationEnergy[8]:
            if EN <= XCFF[NCFF - 1]:
                j = 0
                for j in range(1, NCFF):
                    if EN <= XCFF[j]:
                        break
                A = (YCFF[j] - YCFF[j - 1]) / (XCFF[j] - XCFF[j - 1])
                B = (XCFF[j - 1] * YCFF[j] - XCFF[j] * YCFF[j - 1]) / (XCFF[j - 1] - XCFF[j])
                object.IonizationCrossSection[8][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCFF[NCFF] EV
                X2 = 1 / BETA2
                X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                object.IonizationCrossSection[8][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * <float>(0.0189)
            if EN > 2 * object.IonizationEnergy[8]:
                object.PEIonizationCrossSection[8][i] = object.PEElasticCrossSection[1][(i - IOFFION[8])]

        # DOUBLE IONS    C +  AND F +
        object.IonizationCrossSection[9][i] = 0.0
        object.PEIonizationCrossSection[9][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[9][i] = 0.0

        if EN > object.IonizationEnergy[9]:
            if EN <= XCF[NCF - 1]:
                j = 0
                for j in range(1, NCF):
                    if EN <= XCF[j]:
                        break
                A = (YCF[j] - YCF[j - 1]) / (XCF[j] - XCF[j - 1])
                B = (XCF[j - 1] * YCF[j] - XCF[j] * YCF[j - 1]) / (XCF[j - 1] - XCF[j])
                object.IonizationCrossSection[9][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF[NCF] EV
                X2 = 1 / BETA2
                X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                object.IonizationCrossSection[9][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * <float>(0.0087)
            if EN > 2 * object.IonizationEnergy[9]:
                object.PEIonizationCrossSection[9][i] = object.PEElasticCrossSection[1][(i - IOFFION[9])]

        # CARBON K-SHELL IONISATION
        object.IonizationCrossSection[10][i] = 0.0
        object.PEIonizationCrossSection[10][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[10][i] = 0.0

        if EN > object.IonizationEnergy[10]:
            if EN <= XKSHC[NKSHC - 1]:
                j = 0
                for j in range(1, NKSHC):
                    if EN <= XKSHC[j]:
                        break
                A = (YKSHC[j] - YKSHC[j - 1]) / (XKSHC[j] - XKSHC[j - 1])
                B = (XKSHC[j - 1] * YKSHC[j] - XKSHC[j] * YKSHC[j - 1]) / (XKSHC[j - 1] - XKSHC[j])
                object.IonizationCrossSection[10][i] = (A * EN + B) * 1e-16
            if EN > 2 * object.IonizationEnergy[10]:
                object.PEIonizationCrossSection[10][i] = object.PEElasticCrossSection[1][(i - IOFFION[10])]

        # Fluorine K-SHELL IONISATION
        object.IonizationCrossSection[11][i] = 0.0
        object.PEIonizationCrossSection[11][i] = 0.5

        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[11][i] = 0.0

        if EN > object.IonizationEnergy[11]:
            if EN <= XKSHF[NKSHF - 1]:
                j = 0
                for j in range(1, NKSHF):
                    if EN <= XKSHF[j]:
                        break
                A = (YKSHF[j] - YKSHF[j - 1]) / (XKSHF[j] - XKSHF[j - 1])
                B = (XKSHF[j - 1] * YKSHF[j] - XKSHF[j] * YKSHF[j - 1]) / (XKSHF[j - 1] - XKSHF[j])
                object.IonizationCrossSection[11][i] = 4 * (A * EN + B) * 1e-16
            if EN > 2 * object.IonizationEnergy[11]:
                object.PEIonizationCrossSection[11][i] = object.PEElasticCrossSection[1][(i - IOFFION[11])]

        # ATTACHMENT
        j = 0
        object.Q[3][i] = 0.0
        if EN > XATT[0]:
            if EN <= XATT[N_Attachment1 - 1]:
                for j in range(1, N_Attachment1):
                    if EN <= XATT[j]:
                        break
                A = (YATT[j] - YATT[j - 1]) / (XATT[j] - XATT[j - 1])
                B = (XATT[j - 1] * YATT[j] - XATT[j] * YATT[j - 1]) / (XATT[j - 1] - XATT[j])
                object.Q[3][i] = (A * EN + B) * 1e-16
                object.AttachmentCrossSection[0][i] = object.Q[3][i]
        object.Q[4][i] = 0.0
        object.Q[5][i] = 0.0

        # SCALE FACTOR FOR VIBRATIONAL DIPOLE V3 ABOVE 0.4EV

        VDSC = 1.0
        if EN > <float>(0.4):
            EPR = EN
            if EN > 5.0:
                EPR = 5.0
            VDSC = (<float>(14.4) - EPR) / 14.0
        # SUPERELASTIC OF VIBRATION V2 ISOTROPIC  BELOW 100EV
        object.InelasticCrossSectionPerGas[0][i] = 0.0
        object.PEInelasticCrossSectionPerGas[0][i] = 0.5
        if EN > 0.0:

            EFAC = sqrt(1.0 - (object.EnergyLevels[0] / EN))
            object.InelasticCrossSectionPerGas[0][i] = <float>(0.007) * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.InelasticCrossSectionPerGas[0][i] = object.InelasticCrossSectionPerGas[0][i] * APOPV2 * 1.0e-16 / DEGV2
            if EN > 100.0:
                object.PEInelasticCrossSectionPerGas[0][i] = PQ[1]

        # VIBRATION V2 ISOTROPIC BELOW 100EV
        object.InelasticCrossSectionPerGas[1][i] = 0.0
        object.PEInelasticCrossSectionPerGas[1][i] = 0.5
        if EN > object.EnergyLevels[1]:
            EFAC = sqrt(1.0 - (object.EnergyLevels[1] / EN))
            object.InelasticCrossSectionPerGas[1][i] = <float>(0.007) * log((1.0 + EFAC) / (1.0 - EFAC)) / EN
            object.InelasticCrossSectionPerGas[1][i] = object.InelasticCrossSectionPerGas[1][i] * APOPGS * 1.0e-16
            if EN > 100.0:
                object.PEInelasticCrossSectionPerGas[1][i] = PQ[1]

        # SUPERELASTIC OF VIBRATION V4 ISOTROPIC BELOW 100EV
        object.InelasticCrossSectionPerGas[2][i] = 0.0
        object.PEInelasticCrossSectionPerGas[2][i] = 0.5
        if EN > 0.0:
            if EN - object.EnergyLevels[2] <= XVBV4[NVBV4 - 1]:
                j = 0
                for j in range(1, NVBV4):
                    if EN - object.EnergyLevels[2] <= XVBV4[j]:
                        break
                A = (YVBV4[j] - YVBV4[j - 1]) / (XVBV4[j] - XVBV4[j - 1])
                B = (XVBV4[j - 1] * YVBV4[j] - XVBV4[j] * YVBV4[j - 1]) / (XVBV4[j - 1] - XVBV4[j])
                object.InelasticCrossSectionPerGas[2][i] = (EN - object.EnergyLevels[2]) * (A * (EN - object.EnergyLevels[2]) + B) / EN
            else:
                object.InelasticCrossSectionPerGas[2][i] = YVBV4[NVBV4 - 1] * (XVBV4[NVBV4 - 1] / (EN * pow((EN - object.EnergyLevels[2]) ,2)))
            EFAC = sqrt(1.0 - (object.EnergyLevels[2] / EN))
            object.InelasticCrossSectionPerGas[2][i] = object.InelasticCrossSectionPerGas[2][i] + <float>(0.05) * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.InelasticCrossSectionPerGas[2][i] = object.InelasticCrossSectionPerGas[2][i] * APOPV4 * 1.0e-16 / DEGV4
            if EN > 100.0:
                object.PEInelasticCrossSectionPerGas[2][i] = PQ[1]

        # VIBRATION V4 AAnisotropicDetectedTROPIC
        object.InelasticCrossSectionPerGas[3][i] = 0.0
        object.PEInelasticCrossSectionPerGas[3][i] = 0.5
        if EN > object.EnergyLevels[3]:
            if EN <= XVBV4[NVBV4 - 1]:
                j = 0
                for j in range(1, NVBV4):
                    if EN <= XVBV4[j]:
                        break
                A = (YVBV4[j] - YVBV4[j - 1]) / (XVBV4[j] - XVBV4[j - 1])
                B = (XVBV4[j - 1] * YVBV4[j] - XVBV4[j] * YVBV4[j - 1]) / (XVBV4[j - 1] - XVBV4[j])
                object.InelasticCrossSectionPerGas[3][i] = A * EN + B
            else:
                object.InelasticCrossSectionPerGas[3][i] = YVBV4[NVBV4 - 1] *pow((XVBV4[NVBV4 - 1] / EN) , 3)
            EFAC = sqrt(1.0 - (object.EnergyLevels[3] / EN))
            ADIP = <float>(0.05) * log((1.0 + EFAC) / (1.0 - EFAC)) / EN
            ELF = EN - object.EnergyLevels[3]
            FWD = log((EN + ELF) / (EN + ELF - 2.0 * sqrt(EN * ELF)))
            BCK = log((EN + ELF + 2.0 * sqrt(EN * ELF)) / (EN + ELF))
            # RATIO OF MT TO TOTAL X-SECT FOR RESONANCE PART = RAT
            XMT = ((1.5 - FWD / (FWD + BCK)) * ADIP + RAT * object.InelasticCrossSectionPerGas[3][i]) * APOPGS * 1.0e-16
            object.InelasticCrossSectionPerGas[3][i] = (object.InelasticCrossSectionPerGas[3][i] + ADIP) * APOPGS * 1.0e-16
            if EN <= 100:
                object.PEInelasticCrossSectionPerGas[3][i] = 0.5 + (object.InelasticCrossSectionPerGas[3][i] - XMT) / object.InelasticCrossSectionPerGas[3][i]
            else:
                object.PEInelasticCrossSectionPerGas[3][i] = PQ[1]

        # SUPERELASTIC OF VIBRATION V1 ISOTROPIC BELOW 100EV
        object.InelasticCrossSectionPerGas[4][i] = 0.0
        object.PEInelasticCrossSectionPerGas[4][i] = 0.5
        if EN > 0.0:
            if EN - object.EnergyLevels[4] <= XVBV1[NVBV1 - 1]:
                j = 0
                for j in range(1, NVBV1):
                    if EN - object.EnergyLevels[4] <= XVBV1[j]:
                        break
                A = (YVBV1[j] - YVBV1[j - 1]) / (XVBV1[j] - XVBV1[j - 1])
                B = (XVBV1[j - 1] * YVBV1[j] - XVBV1[j] * YVBV1[j - 1]) / (XVBV1[j - 1] - XVBV1[j])
                object.InelasticCrossSectionPerGas[4][i] = (EN - object.EnergyLevels[4]) * (A * (EN - object.EnergyLevels[4]) + B) / EN
            else:
                object.InelasticCrossSectionPerGas[4][i] = YVBV1[NVBV1 - 1] * (XVBV1[NVBV1 - 1] / (EN * pow((EN - object.EnergyLevels[4]) , 2)))
            EFAC = sqrt(1.0 - (object.EnergyLevels[4] / EN))
            object.InelasticCrossSectionPerGas[4][i] = object.InelasticCrossSectionPerGas[4][i] + <float>(0.0224) * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.InelasticCrossSectionPerGas[4][i] = object.InelasticCrossSectionPerGas[4][i] * APOPV1 * 1.0e-16 / DEGV1
            if EN > 100.0:
                object.PEInelasticCrossSectionPerGas[4][i] = PQ[1]

        # VIBRATION V1  ISOTROPIC BELOW 100EV
        object.InelasticCrossSectionPerGas[5][i] = 0.0
        object.PEInelasticCrossSectionPerGas[5][i] = 0.5
        if EN > object.EnergyLevels[5]:
            if EN <= XVBV1[NVBV1 - 1]:
                j = 0
                for j in range(1, NVBV1):
                    if EN <= XVBV1[j]:
                        break
                A = (YVBV1[j] - YVBV1[j - 1]) / (XVBV1[j] - XVBV1[j - 1])
                B = (XVBV1[j - 1] * YVBV1[j] - XVBV1[j] * YVBV1[j - 1]) / (XVBV1[j - 1] - XVBV1[j])
                object.InelasticCrossSectionPerGas[5][i] = A * EN + B
            else:
                object.InelasticCrossSectionPerGas[5][i] = YVBV1[NVBV1 - 1] * pow((XVBV1[NVBV1 - 1] / EN) , 3)
            EFAC = sqrt(1.0 - (object.EnergyLevels[5] / EN))
            object.InelasticCrossSectionPerGas[5][i] = object.InelasticCrossSectionPerGas[5][i] + <float>(0.0224) * log((EFAC + 1.0) / (1.0 - EFAC)) / EN
            object.InelasticCrossSectionPerGas[5][i] = object.InelasticCrossSectionPerGas[5][i] * APOPGS * 1.0e-16
            if EN > 100.0:
                object.PEInelasticCrossSectionPerGas[5][i] = PQ[1]

        # SUPERELASTIC OF VIBRATION V3 ISOTROPIC BELOW 100EV
        object.InelasticCrossSectionPerGas[6][i] = 0.0
        object.PEInelasticCrossSectionPerGas[6][i] = 0.5
        if EN > 0.0:
            if EN - object.EnergyLevels[6] <= XVBV3[NVBV3 - 1]:
                j = 0
                for j in range(1, NVBV3):
                    if EN - object.EnergyLevels[6] <= XVBV3[j]:
                        break
                A = (YVBV3[j] - YVBV3[j - 1]) / (XVBV3[j] - XVBV3[j - 1])
                B = (XVBV3[j - 1] * YVBV3[j] - XVBV3[j] * YVBV3[j - 1]) / (XVBV3[j - 1] - XVBV3[j])
                object.InelasticCrossSectionPerGas[6][i] = (EN - object.EnergyLevels[6]) * (A * (EN - object.EnergyLevels[6]) + B) / EN
            else:
                object.InelasticCrossSectionPerGas[6][i] = YVBV3[NVBV3 - 1] * (XVBV3[NVBV3 - 1] / (EN * pow((EN - object.EnergyLevels[6]) , 2)))
            EFAC = sqrt(1.0 - (object.EnergyLevels[6] / EN))
            object.InelasticCrossSectionPerGas[6][i] = object.InelasticCrossSectionPerGas[6][i] + VDSC * <float>(1.610) * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.InelasticCrossSectionPerGas[6][i] = object.InelasticCrossSectionPerGas[6][i] * APOPV3 * 1.0e-16 / DEGV3
            if EN > 100.0:
                object.PEInelasticCrossSectionPerGas[6][i] = PQ[1]
        # VIBRATION V3 AAnisotropicDetectedTROPIC
        object.InelasticCrossSectionPerGas[7][i] = 0.0
        object.PEInelasticCrossSectionPerGas[7][i] = 0.5
        if EN > object.EnergyLevels[7]:
            if EN <= XVBV3[NVBV3 - 1]:
                j = 0
                for j in range(1, NVBV3):
                    if EN <= XVBV3[j]:
                        break
                A = (YVBV3[j] - YVBV3[j - 1]) / (XVBV3[j] - XVBV3[j - 1])
                B = (XVBV3[j - 1] * YVBV3[j] - XVBV3[j] * YVBV3[j - 1]) / (XVBV3[j - 1] - XVBV3[j])
                object.InelasticCrossSectionPerGas[7][i] = A * EN + B
            else:
                object.InelasticCrossSectionPerGas[7][i] = YVBV3[NVBV3 - 1] * pow((XVBV3[NVBV3 - 1] / EN) , 3)
            EFAC = sqrt(1.0 - (object.EnergyLevels[7] / EN))
            ADIP = VDSC * <float>(1.610) * log((EFAC + 1.0) / (1.0 - EFAC)) / EN
            ELF = EN - object.EnergyLevels[7]
            FWD = log((EN + ELF) / (EN + ELF - 2.0 * sqrt(EN * ELF)))
            BCK = log((EN + ELF + 2.0 * sqrt(EN * ELF)) / (EN + ELF))
            # ASSumE RATIO MOM T./ TOT X-SECT FOR RESONANCE PART = RAT
            XMT = ((1.5 - FWD / (FWD + BCK)) * ADIP + RAT * object.InelasticCrossSectionPerGas[7][i]) * APOPGS * 1.0e-16
            object.InelasticCrossSectionPerGas[7][i] = (object.InelasticCrossSectionPerGas[7][i] + ADIP) * APOPGS * 1.0e-16
            if EN <= 100:
                object.PEInelasticCrossSectionPerGas[7][i] = 0.5 + (object.InelasticCrossSectionPerGas[7][i] - XMT) / object.InelasticCrossSectionPerGas[7][i]
            else:
                object.PEInelasticCrossSectionPerGas[7][i] = PQ[1]

        # VIBRATION HARMONIC 2V3
        object.InelasticCrossSectionPerGas[8][i] = 0.0
        object.PEInelasticCrossSectionPerGas[8][i] = 0.5
        if EN > object.EnergyLevels[8]:
            if EN <= XVIB5[NVIB5 - 1]:
                j = 0
                for j in range(1, NVIB5):
                    if EN <= XVIB5[j]:
                        break
                A = (YVIB5[j] - YVIB5[j - 1]) / (XVIB5[j] - XVIB5[j - 1])
                B = (XVIB5[j - 1] * YVIB5[j] - XVIB5[j] * YVIB5[j - 1]) / (XVIB5[j - 1] - XVIB5[j])
                object.InelasticCrossSectionPerGas[8][i] = A * EN + B
            else:
                object.InelasticCrossSectionPerGas[8][i] = YVIB5[NVIB5 - 1] * (XVIB5[NVIB5 - 1] / EN)
            object.InelasticCrossSectionPerGas[8][i] = object.InelasticCrossSectionPerGas[8][i] * APOPGS * 1.0e-16
            if EN <= 100:
                object.PEInelasticCrossSectionPerGas[8][i] = 0.5 + (1.0 - RAT)
            else:
                object.PEInelasticCrossSectionPerGas[8][i] = PQ[1]
        # VIBRATION HARMONIC 3V3
        object.InelasticCrossSectionPerGas[9][i] = 0.0
        object.PEInelasticCrossSectionPerGas[9][i] = 0.5
        if EN > object.EnergyLevels[9]:
            if EN <= XVIB6[NVIB6 - 1]:
                j = 0
                for j in range(1, NVIB6):
                    if EN <= XVIB6[j]:
                        break
                A = (YVIB6[j] - YVIB6[j - 1]) / (XVIB6[j] - XVIB6[j - 1])
                B = (XVIB6[j - 1] * YVIB6[j] - XVIB6[j] * YVIB6[j - 1]) / (XVIB6[j - 1] - XVIB6[j])
                object.InelasticCrossSectionPerGas[9][i] = A * EN + B
            else:
                object.InelasticCrossSectionPerGas[9][i] = YVIB6[NVIB6 - 1] * (XVIB6[NVIB6 - 1] / EN)
            object.InelasticCrossSectionPerGas[9][i] = object.InelasticCrossSectionPerGas[9][i] * APOPGS * 1.0e-16
            if EN <= 100:
                object.PEInelasticCrossSectionPerGas[9][i] = 0.5 + (1.0 - RAT)
            else:
                object.PEInelasticCrossSectionPerGas[9][i] = PQ[1]

        # TRIPLET NEUTRAL DISSOCIATION ELOSS=11.5 EV
        object.InelasticCrossSectionPerGas[10][i] = 0.0
        object.PEInelasticCrossSectionPerGas[10][i] = 0.0
        if EN > object.EnergyLevels[10]:
            if EN <= XTR1[NTR1 - 1]:
                j = 0
                for j in range(1, NTR1):
                    if EN <= XTR1[j]:
                        break
                A = (YTR1[j] - YTR1[j - 1]) / (XTR1[j] - XTR1[j - 1])
                B = (XTR1[j - 1] * YTR1[j] - XTR1[j] * YTR1[j - 1]) / (XTR1[j - 1] - XTR1[j])
                object.InelasticCrossSectionPerGas[10][i] = (A * EN + B) * 1.0e-16
            else:
                object.InelasticCrossSectionPerGas[10][i] = YTR1[NTR1 - 1] * pow((XTR1[NTR1 - 1] / EN) , 2) * 1.0e-16
            if EN > 3 * object.EnergyLevels[10]:
                object.PEInelasticCrossSectionPerGas[10][i] = object.PEElasticCrossSection[1][(i - IOFFN[10])]
        # SINGLET NEUTRAL DISSOCIATION  ELOSS=11.63 EV     F=0.0001893
        object.InelasticCrossSectionPerGas[11][i] = 0.0
        object.PEInelasticCrossSectionPerGas[11][i] = 0.0
        if EN > object.EnergyLevels[11]:
            object.InelasticCrossSectionPerGas[11][i] = <float>(0.0001893) / (object.EnergyLevels[11] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0
                                                                                                * object.EnergyLevels[
                                                                                                    11])) - BETA2 -
                                                                 object.DEN[
                                                                     i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[11] + object.E[2]) * <float>(1.0107)
            if object.InelasticCrossSectionPerGas[11][i] < 0.0:
                object.InelasticCrossSectionPerGas[11][i] = 0
            if EN > 3 * object.EnergyLevels[11]:
                object.PEInelasticCrossSectionPerGas[11][i] = object.PEElasticCrossSection[1][i - IOFFN[11]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=11.88 EV     F=0.001085
        object.InelasticCrossSectionPerGas[12][i] = 0.0
        object.PEInelasticCrossSectionPerGas[12][i] = 0.0
        if EN > object.EnergyLevels[12]:
            object.InelasticCrossSectionPerGas[12][i] = <float>(0.001085) / (object.EnergyLevels[12] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0
                                                                                               * object.EnergyLevels[12])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[12] + object.E[2]) * <float>(1.0105)
            if object.InelasticCrossSectionPerGas[12][i] < 0.0:
                object.InelasticCrossSectionPerGas[12][i] = 0
            if EN > 3 * object.EnergyLevels[12]:
                object.PEInelasticCrossSectionPerGas[12][i] = object.PEElasticCrossSection[1][i - IOFFN[12]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=11.88 EV     F=0.004807
        object.InelasticCrossSectionPerGas[13][i] = 0.0
        object.PEInelasticCrossSectionPerGas[13][i] = 0.0
        if EN > object.EnergyLevels[13]:
            object.InelasticCrossSectionPerGas[13][i] = <float>(0.004807) / (object.EnergyLevels[13] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                               object.EnergyLevels[13])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[13] + object.E[2]) * <float>(1.0103)
            if object.InelasticCrossSectionPerGas[13][i] < 0.0:
                object.InelasticCrossSectionPerGas[13][i] = 0
            if EN > 3 * object.EnergyLevels[13]:
                object.PEInelasticCrossSectionPerGas[13][i] = object.PEElasticCrossSection[1][i - IOFFN[13]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=12.38 EV     F=0.008819
        object.InelasticCrossSectionPerGas[14][i] = 0.0
        object.PEInelasticCrossSectionPerGas[14][i] = 0.0
        if EN > object.EnergyLevels[14]:
            object.InelasticCrossSectionPerGas[14][i] = <float>(0.008819) / (object.EnergyLevels[14] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                               object.EnergyLevels[14])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[14] + object.E[2]) * <float>(1.0101)
            if object.InelasticCrossSectionPerGas[14][i] < 0.0:
                object.InelasticCrossSectionPerGas[14][i] = 0
            if EN > 3 * object.EnergyLevels[14]:
                object.PEInelasticCrossSectionPerGas[14][i] = object.PEElasticCrossSection[1][i - IOFFN[14]]

        # TRIPLET NEUTRAL DISSOCIATION ELOSS=12.5 EV
        object.InelasticCrossSectionPerGas[15][i] = 0.0
        object.PEInelasticCrossSectionPerGas[15][i] = 0.0
        if EN > object.EnergyLevels[15]:
            if EN <= XTR2[NTR2 - 1]:
                j = 0
                for j in range(1, NTR2):
                    if EN <= XTR2[j]:
                        break
                A = (YTR2[j] - YTR2[j - 1]) / (XTR2[j] - XTR2[j - 1])
                B = (XTR2[j - 1] * YTR2[j] - XTR2[j] * YTR2[j - 1]) / (XTR2[j - 1] - XTR2[j])
                object.InelasticCrossSectionPerGas[15][i] = (A * EN + B) * 1.0e-16
            else:
                object.InelasticCrossSectionPerGas[15][i] = YTR2[NTR2 - 1] * pow((XTR2[NTR2 - 1] / EN) , 2) * 1.0e-16
            if EN > 3 * object.EnergyLevels[15]:
                object.PEInelasticCrossSectionPerGas[15][i] = object.PEElasticCrossSection[1][i - IOFFN[15]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=12.63 EV     F=0.008918
        object.InelasticCrossSectionPerGas[16][i] = 0.0
        object.PEInelasticCrossSectionPerGas[16][i] = 0.0
        if EN > object.EnergyLevels[16]:
            object.InelasticCrossSectionPerGas[16][i] = <float>(0.008918) / (object.EnergyLevels[16] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                               object.EnergyLevels[16])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[16] + object.E[2]) * <float>(1.0099)
            if object.InelasticCrossSectionPerGas[16][i] < 0.0:
                object.InelasticCrossSectionPerGas[16][i] = 0
            if EN > 3 * object.EnergyLevels[16]:
                object.PEInelasticCrossSectionPerGas[16][i] = object.PEElasticCrossSection[1][i - IOFFN[16]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=12.88 EV     F=0.008420
        object.InelasticCrossSectionPerGas[17][i] = 0.0
        object.PEInelasticCrossSectionPerGas[17][i] = 0.0
        if EN > object.EnergyLevels[17]:
            object.InelasticCrossSectionPerGas[17][i] = <float>(0.008420) / (object.EnergyLevels[17] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                               object.EnergyLevels[17])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[17] + object.E[2]) * <float>(1.0097)
            if object.InelasticCrossSectionPerGas[17][i] < 0.0:
                object.InelasticCrossSectionPerGas[17][i] = 0
            if EN > 3 * object.EnergyLevels[17]:
                object.PEInelasticCrossSectionPerGas[17][i] = object.PEElasticCrossSection[1][i - IOFFN[17]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=13.13 EV     F=0.02531
        object.InelasticCrossSectionPerGas[18][i] = 0.0
        object.PEInelasticCrossSectionPerGas[18][i] = 0.0
        if EN > object.EnergyLevels[18]:
            object.InelasticCrossSectionPerGas[18][i] = <float>(0.02531) / (object.EnergyLevels[18] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[18])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[18] + object.E[2]) * <float>(1.0095)
            if object.InelasticCrossSectionPerGas[18][i] < 0.0:
                object.InelasticCrossSectionPerGas[18][i] = 0
            if EN > 3 * object.EnergyLevels[18]:
                object.PEInelasticCrossSectionPerGas[18][i] = object.PEElasticCrossSection[1][i - IOFFN[18]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=13.38 EV     F=0.09553
        object.InelasticCrossSectionPerGas[19][i] = 0.0
        object.PEInelasticCrossSectionPerGas[19][i] = 0.0
        if EN > object.EnergyLevels[19]:
            object.InelasticCrossSectionPerGas[19][i] = <float>(0.09553) / (object.EnergyLevels[19] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[19])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[19] + object.E[2]) * <float>(1.0093)
            if object.InelasticCrossSectionPerGas[19][i] < 0.0:
                object.InelasticCrossSectionPerGas[19][i] = 0
            if EN > 3 * object.EnergyLevels[19]:
                object.PEInelasticCrossSectionPerGas[19][i] = object.PEElasticCrossSection[1][i - IOFFN[19]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=13.63 EV     F=0.11193
        object.InelasticCrossSectionPerGas[20][i] = 0.0
        object.PEInelasticCrossSectionPerGas[20][i] = 0.0
        if EN > object.EnergyLevels[20]:
            object.InelasticCrossSectionPerGas[20][i] = <float>(0.11193) / (object.EnergyLevels[20] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[20])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[20] + object.E[2]) * <float>(1.0092)
            if object.InelasticCrossSectionPerGas[20][i] < 0.0:
                object.InelasticCrossSectionPerGas[20][i] = 0
            if EN > 3 * object.EnergyLevels[20]:
                object.PEInelasticCrossSectionPerGas[20][i] = object.PEElasticCrossSection[1][i - IOFFN[20]]

        # SINGLET NEUTRAL DISSOCIATION    ELOSS=13.88 EV     F=0.10103
        object.InelasticCrossSectionPerGas[21][i] = 0.0
        object.PEInelasticCrossSectionPerGas[21][i] = 0.0
        if EN > object.EnergyLevels[21]:
            object.InelasticCrossSectionPerGas[21][i] = <float>(0.10103) / (object.EnergyLevels[21] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[21])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[21] + object.E[2]) * <float>(1.0090)
            if object.InelasticCrossSectionPerGas[21][i] < 0.0:
                object.InelasticCrossSectionPerGas[21][i] = 0
            if EN > 3 * object.EnergyLevels[21]:
                object.PEInelasticCrossSectionPerGas[21][i] = object.PEElasticCrossSection[1][i - IOFFN[21]]

        # TRIPLET NEUTRAL DISSOCIATION ELOSS=14.0 EV
        object.InelasticCrossSectionPerGas[22][i] = 0.0
        object.PEInelasticCrossSectionPerGas[22][i] = 0.0
        if EN > object.EnergyLevels[22]:
            if EN <= XTR3[NTR3 - 1]:
                j = 0
                for j in range(1, NTR3):
                    if EN <= XTR3[j]:
                        break
                A = (YTR3[j] - YTR3[j - 1]) / (XTR3[j] - XTR3[j - 1])
                B = (XTR3[j - 1] * YTR3[j] - XTR3[j] * YTR3[j - 1]) / (XTR3[j - 1] - XTR3[j])
                object.InelasticCrossSectionPerGas[22][i] = (A * EN + B) * 1.0e-16
            else:
                object.InelasticCrossSectionPerGas[22][i] = YTR3[NTR3 - 1] * pow((XTR3[NTR3 - 1] / EN) , 2) * 1.0e-16
            if EN > 3 * object.EnergyLevels[22]:
                object.PEInelasticCrossSectionPerGas[22][i] = object.PEElasticCrossSection[1][i - IOFFN[22]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=14.13 EV     F=0.06902
        object.InelasticCrossSectionPerGas[23][i] = 0.0
        object.PEInelasticCrossSectionPerGas[23][i] = 0.0
        if EN > object.EnergyLevels[23]:
            object.InelasticCrossSectionPerGas[23][i] = <float>(0.06902) / (object.EnergyLevels[23] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[23])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[23] + object.E[2]) * <float>(1.0088)
            if object.InelasticCrossSectionPerGas[23][i] < 0.0:
                object.InelasticCrossSectionPerGas[23][i] = 0
            if EN > 3 * object.EnergyLevels[23]:
                object.PEInelasticCrossSectionPerGas[23][i] = object.PEElasticCrossSection[1][i - IOFFN[23]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=14.38 EV     F=0.03968
        object.InelasticCrossSectionPerGas[24][i] = 0.0
        object.PEInelasticCrossSectionPerGas[24][i] = 0.0
        if EN > object.EnergyLevels[24]:
            object.InelasticCrossSectionPerGas[24][i] = <float>(0.03968) / (object.EnergyLevels[24] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[24])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[24] + object.E[2]) * <float>(1.0087)
            if object.InelasticCrossSectionPerGas[24][i] < 0.0:
                object.InelasticCrossSectionPerGas[24][i] = 0
            if EN > 3 * object.EnergyLevels[24]:
                object.PEInelasticCrossSectionPerGas[24][i] = object.PEElasticCrossSection[1][i - IOFFN[24]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=14.63 EV     F=0.02584
        object.InelasticCrossSectionPerGas[25][i] = 0.0
        object.PEInelasticCrossSectionPerGas[25][i] = 0.0
        if EN > object.EnergyLevels[25]:
            object.InelasticCrossSectionPerGas[25][i] = <float>(0.02584) / (object.EnergyLevels[25] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[25])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[25] + object.E[2]) * <float>(1.0085)
            if object.InelasticCrossSectionPerGas[25][i] < 0.0:
                object.InelasticCrossSectionPerGas[25][i] = 0
            if EN > 3 * object.EnergyLevels[25]:
                object.PEInelasticCrossSectionPerGas[25][i] = object.PEElasticCrossSection[1][i - IOFFN[25]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=14.88 EV     F=0.02071
        object.InelasticCrossSectionPerGas[26][i] = 0.0
        object.PEInelasticCrossSectionPerGas[26][i] = 0.0
        if EN > object.EnergyLevels[26]:
            object.InelasticCrossSectionPerGas[26][i] = <float>(0.02071) / (object.EnergyLevels[26] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[26])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[26] + object.E[2]) * <float>(1.0084)
            if object.InelasticCrossSectionPerGas[26][i] < 0.0:
                object.InelasticCrossSectionPerGas[26][i] = 0
            if EN > 3 * object.EnergyLevels[26]:
                object.PEInelasticCrossSectionPerGas[26][i] = object.PEElasticCrossSection[1][i - IOFFN[26]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=15.13 EV     F=0.03122
        object.InelasticCrossSectionPerGas[27][i] = 0.0
        object.PEInelasticCrossSectionPerGas[27][i] = 0.0
        if EN > object.EnergyLevels[27]:
            object.InelasticCrossSectionPerGas[27][i] = <float>(0.03122) / (object.EnergyLevels[27] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[27])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[27] + object.E[2]) * <float>(1.0083)
            if object.InelasticCrossSectionPerGas[27][i] < 0.0:
                object.InelasticCrossSectionPerGas[27][i] = 0
            if EN > 3 * object.EnergyLevels[27]:
                object.PEInelasticCrossSectionPerGas[27][i] = object.PEElasticCrossSection[1][i - IOFFN[27]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=15.38 EV     F=0.05580
        object.InelasticCrossSectionPerGas[28][i] = 0.0
        object.PEInelasticCrossSectionPerGas[28][i] = 0.0
        if EN > object.EnergyLevels[28]:
            object.InelasticCrossSectionPerGas[28][i] = <float>(0.05580) / (object.EnergyLevels[28] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[28])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[28] + object.E[2]) * <float>(1.0081)
            if object.InelasticCrossSectionPerGas[28][i] < 0.0:
                object.InelasticCrossSectionPerGas[28][i] = 0
            if EN > 3 * object.EnergyLevels[28]:
                object.PEInelasticCrossSectionPerGas[28][i] = object.PEElasticCrossSection[1][i - IOFFN[28]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=15.63 EV     F=0.10187
        object.InelasticCrossSectionPerGas[29][i] = 0.0
        object.PEInelasticCrossSectionPerGas[29][i] = 0.0
        if EN > object.EnergyLevels[29]:
            object.InelasticCrossSectionPerGas[29][i] = <float>(0.10187) / (object.EnergyLevels[29] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[29])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[29] + object.E[2]) * <float>(1.0080)
            if object.InelasticCrossSectionPerGas[29][i] < 0.0:
                object.InelasticCrossSectionPerGas[29][i] = 0
            if EN > 3 * object.EnergyLevels[29]:
                object.PEInelasticCrossSectionPerGas[29][i] = object.PEElasticCrossSection[1][i - IOFFN[29]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=15.88 EV     F=0.09427
        object.InelasticCrossSectionPerGas[30][i] = 0.0
        object.PEInelasticCrossSectionPerGas[30][i] = 0.0
        if EN > object.EnergyLevels[30]:
            object.InelasticCrossSectionPerGas[30][i] = <float>(0.09427) / (object.EnergyLevels[30] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[30])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[30] + object.E[2]) * <float>(1.0079)
            if object.InelasticCrossSectionPerGas[30][i] < 0.0:
                object.InelasticCrossSectionPerGas[30][i] = 0
            if EN > 3 * object.EnergyLevels[30]:
                object.PEInelasticCrossSectionPerGas[30][i] = object.PEElasticCrossSection[1][i - IOFFN[30]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=16.13 EV     F=0.05853
        object.InelasticCrossSectionPerGas[31][i] = 0.0
        object.PEInelasticCrossSectionPerGas[31][i] = 0.0
        if EN > object.EnergyLevels[31]:
            object.InelasticCrossSectionPerGas[31][i] = <float>(0.05853) / (object.EnergyLevels[31] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[31])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[31] + object.E[2]) * <float>(1.0077)
            if object.InelasticCrossSectionPerGas[31][i] < 0.0:
                object.InelasticCrossSectionPerGas[31][i] = 0
            if EN > 3 * object.EnergyLevels[31]:
                object.PEInelasticCrossSectionPerGas[31][i] = object.PEElasticCrossSection[1][i - IOFFN[31]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=16.38 EV     F=0.06002
        object.InelasticCrossSectionPerGas[32][i] = 0.0
        object.PEInelasticCrossSectionPerGas[32][i] = 0.0
        if EN > object.EnergyLevels[32]:
            object.InelasticCrossSectionPerGas[32][i] = <float>(0.06002) / (object.EnergyLevels[32] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[32])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[32] + object.E[2]) * <float>(1.0076)
        if object.InelasticCrossSectionPerGas[32][i] < 0.0:
            object.InelasticCrossSectionPerGas[32][i] = 0
        if EN > 3 * object.EnergyLevels[32]:
            object.PEInelasticCrossSectionPerGas[32][i] = object.PEElasticCrossSection[1][i - IOFFN[32]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=16.63 EV     F=0.05647
        object.InelasticCrossSectionPerGas[33][i] = 0.0
        object.PEInelasticCrossSectionPerGas[33][i] = 0.0
        if EN > object.EnergyLevels[33]:
            object.InelasticCrossSectionPerGas[33][i] = <float>(0.05647) / (object.EnergyLevels[33] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[33])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[33] + object.E[2]) * <float>(1.0075)
            if object.InelasticCrossSectionPerGas[33][i] < 0.0:
                object.InelasticCrossSectionPerGas[33][i] = 0
            if EN > 3 * object.EnergyLevels[33]:
                object.PEInelasticCrossSectionPerGas[33][i] = object.PEElasticCrossSection[1][i - IOFFN[33]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=16.88 EV     F=0.04885
        object.InelasticCrossSectionPerGas[34][i] = 0.0
        object.PEInelasticCrossSectionPerGas[34][i] = 0.0
        if EN > object.EnergyLevels[34]:
            object.InelasticCrossSectionPerGas[34][i] = <float>(0.04885) / (object.EnergyLevels[34] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[34])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[34] + object.E[2]) * <float>(1.0074)
            if object.InelasticCrossSectionPerGas[34][i] < 0.0:
                object.InelasticCrossSectionPerGas[34][i] = 0
            if EN > 3 * object.EnergyLevels[34]:
                object.PEInelasticCrossSectionPerGas[34][i] = object.PEElasticCrossSection[1][i - IOFFN[34]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=17.13 EV     F=0.04036
        object.InelasticCrossSectionPerGas[35][i] = 0.0
        object.PEInelasticCrossSectionPerGas[35][i] = 0.0
        if EN > object.EnergyLevels[35]:
            object.InelasticCrossSectionPerGas[35][i] = <float>(0.04036) / (object.EnergyLevels[35] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[35])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[35] + object.E[2]) * <float>(1.0073)
            if object.InelasticCrossSectionPerGas[35][i] < 0.0:
                object.InelasticCrossSectionPerGas[35][i] = 0
            if EN > 3 * object.EnergyLevels[35]:
                object.PEInelasticCrossSectionPerGas[35][i] = object.PEElasticCrossSection[1][i - IOFFN[35]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=17.38 EV     F=0.03298
        object.InelasticCrossSectionPerGas[36][i] = 0.0
        object.PEInelasticCrossSectionPerGas[36][i] = 0.0
        if EN > object.EnergyLevels[36]:
            object.InelasticCrossSectionPerGas[36][i] = <float>(0.03298) / (object.EnergyLevels[36] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[36])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[36] + object.E[2]) * <float>(1.0072)
            if object.InelasticCrossSectionPerGas[36][i] < 0.0:
                object.InelasticCrossSectionPerGas[36][i] = 0
            if EN > 3 * object.EnergyLevels[36]:
                object.PEInelasticCrossSectionPerGas[36][i] = object.PEElasticCrossSection[1][i - IOFFN[36]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=17.63 EV     F=0.02593
        object.InelasticCrossSectionPerGas[37][i] = 0.0
        object.PEInelasticCrossSectionPerGas[37][i] = 0.0
        if EN > object.EnergyLevels[37]:
            object.InelasticCrossSectionPerGas[37][i] = <float>(0.02593) / (object.EnergyLevels[37] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[37])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[37] + object.E[2]) * <float>(1.0071)
            if object.InelasticCrossSectionPerGas[37][i] < 0.0:
                object.InelasticCrossSectionPerGas[37][i] = 0
            if EN > 3 * object.EnergyLevels[37]:
                object.PEInelasticCrossSectionPerGas[37][i] = object.PEElasticCrossSection[1][i - IOFFN[37]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=17.88 EV     F=0.01802
        object.InelasticCrossSectionPerGas[38][i] = 0.0
        object.PEInelasticCrossSectionPerGas[38][i] = 0.0
        if EN > object.EnergyLevels[38]:
            object.InelasticCrossSectionPerGas[38][i] = <float>(0.01802) / (object.EnergyLevels[38] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[38])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[38] + object.E[2]) * <float>(1.0070)
            if object.InelasticCrossSectionPerGas[38][i] < 0.0:
                object.InelasticCrossSectionPerGas[38][i] = 0
            if EN > 3 * object.EnergyLevels[38]:
                object.PEInelasticCrossSectionPerGas[38][i] = object.PEElasticCrossSection[1][i - IOFFN[38]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=18.13 EV     F=0.01287
        object.InelasticCrossSectionPerGas[39][i] = 0.0
        object.PEInelasticCrossSectionPerGas[39][i] = 0.0
        if EN > object.EnergyLevels[39]:
            object.InelasticCrossSectionPerGas[39][i] = <float>(0.01287) / (object.EnergyLevels[39] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[39])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[39] + object.E[2]) * <float>(1.0069)
            if object.InelasticCrossSectionPerGas[39][i] < 0.0:
                object.InelasticCrossSectionPerGas[39][i] = 0
            if EN > 3 * object.EnergyLevels[39]:
                object.PEInelasticCrossSectionPerGas[39][i] = object.PEElasticCrossSection[1][i - IOFFN[39]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=18.38 EV     F=0.00830
        object.InelasticCrossSectionPerGas[40][i] = 0.0
        object.PEInelasticCrossSectionPerGas[40][i] = 0.0
        if EN > object.EnergyLevels[40]:
            object.InelasticCrossSectionPerGas[40][i] = <float>(0.00830) / (object.EnergyLevels[40] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[40])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[40] + object.E[2]) * <float>(1.0068)
            if object.InelasticCrossSectionPerGas[40][i] < 0.0:
                object.InelasticCrossSectionPerGas[40][i] = 0
            if EN > 3 * object.EnergyLevels[40]:
                object.PEInelasticCrossSectionPerGas[40][i] = object.PEElasticCrossSection[1][i - IOFFN[40]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=18.63 EV     F=0.00698
        object.InelasticCrossSectionPerGas[41][i] = 0.0
        object.PEInelasticCrossSectionPerGas[41][i] = 0.0
        if EN > object.EnergyLevels[41]:
            object.InelasticCrossSectionPerGas[41][i] = <float>(0.00698) / (object.EnergyLevels[41] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[41])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[41] + object.E[2]) * <float>(1.0067)
            if object.InelasticCrossSectionPerGas[41][i] < 0.0:
                object.InelasticCrossSectionPerGas[41][i] = 0
            if EN > 3 * object.EnergyLevels[41]:
                object.PEInelasticCrossSectionPerGas[41][i] = object.PEElasticCrossSection[1][i - IOFFN[41]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=18.88 EV     F=0.00581
        object.InelasticCrossSectionPerGas[42][i] = 0.0
        object.PEInelasticCrossSectionPerGas[42][i] = 0.0
        if EN > object.EnergyLevels[42]:
            object.InelasticCrossSectionPerGas[42][i] = <float>(0.00581) / (object.EnergyLevels[42] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[42])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[42] + object.E[2]) * <float>(1.0066)
            if object.InelasticCrossSectionPerGas[42][i] < 0.0:
                object.InelasticCrossSectionPerGas[42][i] = 0
            if EN > 3 * object.EnergyLevels[42]:
                object.PEInelasticCrossSectionPerGas[42][i] = object.PEElasticCrossSection[1][i - IOFFN[42]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=19.13 EV     F=0.00502
        object.InelasticCrossSectionPerGas[43][i] = 0.0
        object.PEInelasticCrossSectionPerGas[43][i] = 0.0
        if EN > object.EnergyLevels[43]:
            object.InelasticCrossSectionPerGas[43][i] = <float>(0.00502) / (object.EnergyLevels[43] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[43])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[43] + object.E[2]) * <float>(1.0065)
            if object.InelasticCrossSectionPerGas[43][i] < 0.0:
                object.InelasticCrossSectionPerGas[43][i] = 0
            if EN > 3 * object.EnergyLevels[43]:
                object.PEInelasticCrossSectionPerGas[43][i] = object.PEElasticCrossSection[1][i - IOFFN[43]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=19.38 EV     F=0.00398
        object.InelasticCrossSectionPerGas[44][i] = 0.0
        object.PEInelasticCrossSectionPerGas[44][i] = 0.0
        if EN > object.EnergyLevels[44]:
            object.InelasticCrossSectionPerGas[44][i] = <float>(0.00398) / (object.EnergyLevels[44] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[44])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[44] + object.E[2]) * <float>(1.0064)
            if object.InelasticCrossSectionPerGas[44][i] < 0.0:
                object.InelasticCrossSectionPerGas[44][i] = 0
            if EN > 3 * object.EnergyLevels[44]:
                object.PEInelasticCrossSectionPerGas[44][i] = object.PEElasticCrossSection[1][i - IOFFN[44]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=19.63 EV     F=0.00189
        object.InelasticCrossSectionPerGas[45][i] = 0.0
        object.PEInelasticCrossSectionPerGas[45][i] = 0.0
        if EN > object.EnergyLevels[45]:
            # magboltz code is 0.00198 while the pattern should go to 0.00189
            object.InelasticCrossSectionPerGas[45][i] = <float>(0.00198) / (object.EnergyLevels[45] * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 *
                                                                                              object.EnergyLevels[45])) - BETA2 -
                                                               object.DEN[
                                                                   i] / 2.0) * BBCONST * EN / (
                                        EN + object.EnergyLevels[45] + object.E[2]) * <float>(1.0064)
            if object.InelasticCrossSectionPerGas[45][i] < 0.0:
                object.InelasticCrossSectionPerGas[45][i] = 0
            if EN > 3 * object.EnergyLevels[45]:
                object.PEInelasticCrossSectionPerGas[45][i] = object.PEElasticCrossSection[1][i - IOFFN[45]]

        IonizationCrossSectionSum = 0.0
        for J in range(0, 12):
            IonizationCrossSectionSum = IonizationCrossSectionSum + object.IonizationCrossSection[J][i]
        QSNGLSum = 0.0
        for J in range(10, 46):
            if J != 10 and J != 15 and J != 22:
                QSNGLSum = QSNGLSum + object.InelasticCrossSectionPerGas[J][i]

        QTRIPSum = object.InelasticCrossSectionPerGas[10][i] + object.InelasticCrossSectionPerGas[15][i] + object.InelasticCrossSectionPerGas[22][i]

        VSum = 0.0
        for J in range(0, 10):
            VSum = VSum + object.InelasticCrossSectionPerGas[J][i]

        IonizationCrossSectionG = IonizationCrossSectionSum
        for J in range(5, 12):
            IonizationCrossSectionG = IonizationCrossSectionG + object.IonizationCrossSection[J][i]

        DISTOT = QSNGLSum + QTRIPSum + IonizationCrossSectionSum
        object.Q[0][i] = object.Q[1][i] + object.Q[3][i] + VSum + DISTOT

    for J in range(10, 46):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break
