from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow
cimport libc.math
import numpy as np
cimport numpy as np
import sys
cimport GasUtil
from Gas cimport Gas
from cython.parallel import prange

sys.path.append('../hdf5_python')
import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.fast_getattr(True)
cdef void Gas3(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Helium4 gas.
    """
    gd = np.load('gases.npy').item()
    cdef double XEN[144], YEM[144], YEL[144], YEPS[144], XION[90], YION[90], YINC[90], X23S[139], Y23S[139], X21S[128], Y21S[128], X23P[128], Y23P[128]
    cdef double X21P[125], Y21P[125], X33S[106], Y33S[106], X31S[87], Y31S[87], X33P[91], Y33P[91], X33D[108], Y33D[108], X31D[94], Y31D[94]
    cdef double X31P[114], Y31P[114], X43S[59], Y43S[59], X41S[55], Y41S[55], X43P[76], Y43P[76], X43D[65], Y43D[65], X41D[53], Y41D[53]
    cdef double X43F[40], Y43F[40], X41F[57], Y41F[57], X41P[96], Y41P[96], Z2T[25], EBRM[25], EOBY[2]
    cdef int IOFFN[49], IOFFION[2]
    XEN = gd['gas3/XEN']
    YEM = gd['gas3/YEM']
    YEL = gd['gas3/YEL']
    YEPS = gd['gas3/YEPS']
    XION = gd['gas3/XION']
    YION = gd['gas3/YION']
    YINC = gd['gas3/YINC']
    X23S = gd['gas3/X23S']
    Y23S = gd['gas3/Y23S']
    X21S = gd['gas3/X21S']
    Y21S = gd['gas3/Y21S']
    X23P = gd['gas3/X23P']
    Y23P = gd['gas3/Y23P']
    X21P = gd['gas3/X21P']
    Y21P = gd['gas3/Y21P']
    X33S = gd['gas3/X33S']
    Y33S = gd['gas3/Y33S']
    X31S = gd['gas3/X31S']
    Y31S = gd['gas3/Y31S']
    X33P = gd['gas3/X33P']
    Y33P = gd['gas3/Y33P']
    X33D = gd['gas3/X33D']
    Y33D = gd['gas3/Y33D']
    X31D = gd['gas3/X31D']
    Y31D = gd['gas3/Y31D']
    X31P = gd['gas3/X31P']
    Y31P = gd['gas3/Y31P']
    X43S = gd['gas3/X43S']
    Y43S = gd['gas3/Y43S']
    X41S = gd['gas3/X41S']
    Y41S = gd['gas3/Y41S']
    X43P = gd['gas3/X43P']
    Y43P = gd['gas3/Y43P']
    X43D = gd['gas3/X43D']
    Y43D = gd['gas3/Y43D']
    X41D = gd['gas3/X41D']
    Y41D = gd['gas3/Y41D']
    X43F = gd['gas3/X43F']
    Y43F = gd['gas3/Y43F']
    X41F = gd['gas3/X41F']
    Y41F = gd['gas3/Y41F']
    X41P = gd['gas3/X41P']
    Y41P = gd['gas3/Y41P']
    Z2T = gd['gas3/Z2T']
    EBRM = gd['gas3/EBRM']

    cdef double ElectronMass2 = <float> (1021997.804)
    cdef double API = acos(-1)
    cdef double A0 = 0.52917720859e-8
    cdef double RY = <float> (13.60569193)
    cdef double BBCONST = 16.0 * API * A0 * A0 * RY * RY / ElectronMass2
    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27
    # BORN BETHE VALUES FOR IONISATION
    cdef double CONST = 1.873884e-20
    cdef double AM2 = <float> (0.489)
    cdef double C = <float> (5.50)
    object.N_Ionization = 2
    object.N_Attachment = 1
    object.N_Inelastic = 49
    object.N_Null = 0
    cdef int NBREM = 25, NDATA = 144, N_IonizationD = 90, N23S = 139, N21S = 128, N23P = 128, N21P = 125, N33S = 106, N31S = 87, N33P = 91, N33D = 108, N31D = 94
    cdef int N31P = 114, N43S = 59, N41S = 55, N43P = 76, N43D = 65, N41D = 53, N43F = 40, N41F = 57, N41P = 96
    object.E = [0.0, <float> (2.0) * ElectronMass / (<float> (4.00260) * AMU), <float> (24.58739), 0.5841e-19, 0.1271e-18,
                <float> (10.5)]
    #IONISATION ENERGIES
    object.IonizationEnergy[0] = <float> (24.58739)
    object.IonizationEnergy[1] = <float> (79.00515)
    # EOBY AT LOW ENERGY
    EOBY[0] = 12.0
    EOBY[1] = 65.0
    cdef double WKLM[2]
    WKLM[0] = 0.0
    WKLM[1] = 0.0
    object.NC0[0] = 0
    object.EC0[0] = 0.0
    object.WK[0:2] = WKLM
    object.EFL[0] = 0.0
    object.NG1[0] = 0
    object.EG1[0] = 0.0
    object.NG2[0] = 0
    object.EG2[0] = 0.0
    object.NC0[1] = 1
    object.EC0[1] = 10.0
    object.EFL[1] = 0.0
    object.NG1[1] = 0
    object.EG1[1] = 0.0
    object.NG2[1] = 0
    object.EG2[1] = 0.0
    cdef int I, j, i
    for i in range(6):
        object.AngularModel[i] = object.WhichAngularModel
    for i in range(object.N_Inelastic):
        object.KIN[i] = object.WhichAngularModel
    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break
    object.EnergyLevels[0:50] = [np.float32(19.81961), np.float32(20.61577), np.float32(20.96409), np.float32(21.21802),
                        np.float32(22.71847), np.float32(22.92032), np.float32(23.00707), np.float32(23.07365),
                        np.float32(23.07407), np.float32(23.08702), np.float32(23.59396), np.float32(23.67357),
                        np.float32(23.70789), np.float32(23.73609), np.float32(23.73633), np.float32(23.73701),
                        np.float32(23.73701), np.float32(23.74207), np.float32(23.97197), np.float32(24.01121),
                        np.float32(24.02822), np.float32(24.04266), np.float32(24.04280), np.float32(24.04315),
                        np.float32(24.04315), np.float32(24.04580), np.float32(24.16900), np.float32(24.19116),
                        np.float32(24.20081), np.float32(24.20916), np.float32(24.20925), np.float32(24.21100),
                        np.float32(24.28456), np.float32(24.29828), np.float32(24.30429), np.float32(24.30954),
                        np.float32(24.30960), np.float32(24.31071), np.float32(24.35810), np.float32(24.36718),
                        np.float32(24.37116), np.float32(24.37468), np.float32(24.37472), np.float32(24.37547),
                        np.float32(24.41989), np.float32(24.45168), np.float32(24.47518), np.float32(24.49308),
                        np.float32(24.50708), np.float32(0.0)]

    for I in range(50, 250):
        object.EnergyLevels[I] = 0.0

    cdef int NL
    for NL in range(object.N_Inelastic):
        object.PenningFraction[0][NL] = 1.0
        # PENN_InelasticG TRANSFER DISTANCE MICRONS
        object.PenningFraction[1][NL] = 1.0
        # PENN_InelasticG TRANSFER TIME PICOSECONDS
        object.PenningFraction[2][NL] = 1.0

    for NL in range(object.N_Inelastic):
        for I in range(4000):
            if object.EG[I] > object.EnergyLevels[NL]:
                IOFFN[NL] = I
                break

    cdef double EN, GAMMA1, GAMMA2, BETA, BETA2, ElasticCrossSectionA, QMOM, A, B, X1, X2, PQ[3], TEMP, QTEMP1, QTEMP2, ER, ENP, QMET, QDIP, QTRP, QSNG, InelasticCrossSection

    for I in range(4000):
        EN = object.EG[I]
        if EN > object.EnergyLevels[0]:
            GAMMA1 = (ElectronMass2 + 2.0 * EN) / ElectronMass2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = sqrt(1.0 - 1.0 / GAMMA2)
            BETA2 = BETA * BETA
        ElasticCrossSectionA = GasUtil.CALIonizationCrossSectionREG(EN, NDATA, YEL, XEN)
        QMOM = GasUtil.CALIonizationCrossSectionREG(EN, NDATA, YEM, XEN)

        TEMP = GasUtil.CALPQ3(EN, NDATA, YEPS, XEN)
        PQ = [0.5, 0.5 + (ElasticCrossSectionA - QMOM) / ElasticCrossSectionA, 1 - TEMP]

        object.PEElasticCrossSection[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = ElasticCrossSectionA

        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM
            object.PEElasticCrossSection[1][I] = 0.5

        #GROSS IONISATION
        object.IonizationCrossSection[0][I] = 0.0
        object.PEIonizationCrossSection[0][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[0][I] = 0
        if EN >= object.IonizationEnergy[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationD, YION, XION, BETA2, 1 / <float> (0.995), CONST,
                                                 object.DEN[I], C, AM2)
            # USE AAnisotropicDetectedTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
            # ENERGIES ABOVE 2 * IONISATION ENERGY
            # AAnisotropicDetectedTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFFSET BY
            # IONISATION ENERGY
            if EN > (2 * object.IonizationEnergy[0]):
                object.PEIonizationCrossSection[0][I] = object.PEElasticCrossSection[1][I - IOFFION[0]]

        #ATTACHMENT
        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = object.Q[3][I]

        #COUNTING IONISATION
        object.Q[4][I] = 0.0
        object.PEIonizationCrossSection[1][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[1][I] = 0
        if EN >= object.IonizationEnergy[0]:
            object.Q[4][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationD, YINC, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
            #USE BORN-BETHE X-SECTION ABOVE XION(N_IonizationD) EV
            if EN > (2 * object.IonizationEnergy[1]):
                object.PEIonizationCrossSection[1][I] = object.PEElasticCrossSection[1][I - IOFFION[1]]

        object.Q[5][I] = 0.0
        QTEMP1 = 2.0 * object.Q[4][I] - object.IonizationCrossSection[0][I]
        QTEMP2 = object.IonizationCrossSection[0][I] - object.Q[4][I]
        object.IonizationCrossSection[0][I] = QTEMP1
        object.IonizationCrossSection[1][I] = QTEMP2
        if object.IonizationCrossSection[1][I] < 0.0:
            object.IonizationCrossSection[1][I] = 0.0

        for NL in range(object.N_Inelastic + 1):
            object.InelasticCrossSectionPerGas[NL][I] = 0.0
            object.PEInelasticCrossSectionPerGas[NL][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[NL][I] = 0.0

        #2 3S
        if EN > object.EnergyLevels[0]:
            object.InelasticCrossSectionPerGas[0][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N23S, Y23S, X23S, 3)
            if EN > (2 * object.EnergyLevels[0]):
                object.PEInelasticCrossSectionPerGas[0][I] = object.PEElasticCrossSection[1][I - IOFFN[0]]

        #2 1S
        if EN > object.EnergyLevels[1]:
            object.InelasticCrossSectionPerGas[1][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N21S, Y21S, X21S, 1)
            if EN > (2 * object.EnergyLevels[1]):
                object.PEInelasticCrossSectionPerGas[1][I] = object.PEElasticCrossSection[1][I - IOFFN[1]]

        #2 3P
        if EN > object.EnergyLevels[2]:
            object.InelasticCrossSectionPerGas[2][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N23P, Y23P, X23P, 3)
            if EN > (2 * object.EnergyLevels[2]):
                object.PEInelasticCrossSectionPerGas[2][I] = object.PEElasticCrossSection[1][I - IOFFN[2]]

        #2 1P
        if EN > object.EnergyLevels[3]:
            object.InelasticCrossSectionPerGas[3][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN,N21P, Y21P, X21P, BETA2, GAMMA2, ElectronMass2, object.DEN[I], BBCONST,
                                                 object.EnergyLevels[3], object.E[2], 0.27608)
            if EN > (2 * object.EnergyLevels[3]):
                object.PEInelasticCrossSectionPerGas[3][I] = object.PEElasticCrossSection[1][I - IOFFN[3]]

        #3 3S
        if EN > object.EnergyLevels[4]:
            object.InelasticCrossSectionPerGas[4][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N33S, Y33S, X33S, 3)
            if EN > (2 * object.EnergyLevels[4]):
                object.PEInelasticCrossSectionPerGas[4][I] = object.PEElasticCrossSection[1][I - IOFFN[4]]

        #3 1S
        if EN > object.EnergyLevels[5]:
            object.InelasticCrossSectionPerGas[5][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N31S, Y31S, X31S, 1)
            if EN > (2 * object.EnergyLevels[5]):
                object.PEInelasticCrossSectionPerGas[5][I] = object.PEElasticCrossSection[1][I - IOFFN[5]]

        #3 3P
        if EN > object.EnergyLevels[6]:
            object.InelasticCrossSectionPerGas[6][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N33P, Y33P, X33P, 1)
            if EN > (2 * object.EnergyLevels[6]):
                object.PEInelasticCrossSectionPerGas[6][I] = object.PEElasticCrossSection[1][I - IOFFN[6]]

        #3 3D
        if EN > object.EnergyLevels[7]:
            object.InelasticCrossSectionPerGas[7][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N33D, Y33D, X33D, 1)
            if EN > (2 * object.EnergyLevels[7]):
                object.PEInelasticCrossSectionPerGas[7][I] = object.PEElasticCrossSection[1][I - IOFFN[7]]

        #3 1D
        if EN > object.EnergyLevels[8]:
            object.InelasticCrossSectionPerGas[8][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N31D, Y31D, X31D, 1)
            if EN > (2 * object.EnergyLevels[8]):
                object.PEInelasticCrossSectionPerGas[8][I] = object.PEElasticCrossSection[1][I - IOFFN[8]]

        #3 1P
        if EN > object.EnergyLevels[9]:
            object.InelasticCrossSectionPerGas[9][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN, EN,N31P, Y31P, X31P, BETA2, GAMMA2, ElectronMass2, object.DEN[I], BBCONST,
                                                 object.EnergyLevels[9], object.E[2], <float> (0.07342))
            if EN > (2 * object.EnergyLevels[9]):
                object.PEInelasticCrossSectionPerGas[9][I] = object.PEElasticCrossSection[1][I - IOFFN[9]]

        #4 3S
        if EN > object.EnergyLevels[10]:
            object.InelasticCrossSectionPerGas[10][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N43S, Y43S, X43S, 3)
            if EN > (2 * object.EnergyLevels[10]):
                object.PEInelasticCrossSectionPerGas[10][I] = object.PEElasticCrossSection[1][I - IOFFN[10]]

        #4 1S
        if EN > object.EnergyLevels[11]:
            object.InelasticCrossSectionPerGas[11][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N41S, Y41S, X41S, 1)
            if EN > (2 * object.EnergyLevels[11]):
                object.PEInelasticCrossSectionPerGas[11][I] = object.PEElasticCrossSection[1][I - IOFFN[11]]

        #4 3P
        if EN > object.EnergyLevels[12]:
            object.InelasticCrossSectionPerGas[12][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N43P, Y43P, X43P, 3)
            if EN > (2 * object.EnergyLevels[12]):
                object.PEInelasticCrossSectionPerGas[12][I] = object.PEElasticCrossSection[1][I - IOFFN[12]]

        #4 3D
        if EN > object.EnergyLevels[13]:
            object.InelasticCrossSectionPerGas[13][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N43D, Y43D, X43D, 3)
            if EN > (2 * object.EnergyLevels[13]):
                object.PEInelasticCrossSectionPerGas[13][I] = object.PEElasticCrossSection[1][I - IOFFN[13]]

        #4 1D
        if EN > object.EnergyLevels[14]:
            object.InelasticCrossSectionPerGas[14][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N41D, Y41D, X41D, 1)
            if EN > (2 * object.EnergyLevels[14]):
                object.PEInelasticCrossSectionPerGas[14][I] = object.PEElasticCrossSection[1][I - IOFFN[14]]

        #4 3F
        if EN > object.EnergyLevels[15]:
            object.InelasticCrossSectionPerGas[15][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N43F, Y43F, X43F, 4)
            if EN > (2 * object.EnergyLevels[15]):
                object.PEInelasticCrossSectionPerGas[15][I] = object.PEElasticCrossSection[1][I - IOFFN[15]]

        #4 1F
        if EN > object.EnergyLevels[16]:
            object.InelasticCrossSectionPerGas[16][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N41F, Y41F, X41F, 1)
            if EN > (2 * object.EnergyLevels[16]):
                object.PEInelasticCrossSectionPerGas[16][I] = object.PEElasticCrossSection[1][I - IOFFN[16]]

        #4 1P
        if EN > object.EnergyLevels[17]:
            object.InelasticCrossSectionPerGas[17][I] = GasUtil.CALInelasticCrossSectionPerGasBEF(EN,EN, N41P, Y41P, X41P, BETA2, GAMMA2, ElectronMass2, object.DEN[I], BBCONST,
                                                  object.EnergyLevels[17], object.E[2], 0.02986)
            if EN > (2 * object.EnergyLevels[17]):
                object.PEInelasticCrossSectionPerGas[17][I] = object.PEElasticCrossSection[1][I - IOFFN[17]]

        #5 3S SCALED FROM 4 3S
        if EN > object.EnergyLevels[18]:
            ER = object.EnergyLevels[18] / object.EnergyLevels[10]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[18][I] = 0.512 * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43S, Y43S, X43S, 3)
            if EN > (2 * object.EnergyLevels[18]):
                object.PEInelasticCrossSectionPerGas[18][I] = object.PEElasticCrossSection[1][I - IOFFN[18]]

        #5 1S SCALED FROM 4 1S
        if EN > object.EnergyLevels[19]:
            ER = object.EnergyLevels[19] / object.EnergyLevels[11]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[19][I] = <float> (0.512) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N41S, Y41S, X41S, 1)
            if EN > (2 * object.EnergyLevels[19]):
                object.PEInelasticCrossSectionPerGas[19][I] = object.PEElasticCrossSection[1][I - IOFFN[19]]

        #5 3P SCALED FROM 5 3P
        if EN > object.EnergyLevels[20]:
            ER = object.EnergyLevels[20] / object.EnergyLevels[12]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[20][I] = <float> (0.512) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43P, Y43P, X43P, 3)
            if EN > (2 * object.EnergyLevels[20]):
                object.PEInelasticCrossSectionPerGas[20][I] = object.PEElasticCrossSection[1][I - IOFFN[20]]

        #5 3D SCALED FROM 4 3D
        if EN > object.EnergyLevels[21]:
            ER = object.EnergyLevels[21] / object.EnergyLevels[13]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[21][I] = <float> (0.512) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43D, Y43D, X43D, 3)
            if EN > (2 * object.EnergyLevels[21]):
                object.PEInelasticCrossSectionPerGas[21][I] = object.PEElasticCrossSection[1][I - IOFFN[21]]

        #5 1D SCALED FROM 4 1D
        if EN > object.EnergyLevels[22]:
            ER = object.EnergyLevels[22] / object.EnergyLevels[14]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[22][I] = <float> (0.512) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N41D, Y41D, X41D, 1)
            if EN > (2 * object.EnergyLevels[22]):
                object.PEInelasticCrossSectionPerGas[22][I] = object.PEElasticCrossSection[1][I - IOFFN[22]]

        #5 3F SCALED FROM 4 3F
        if EN > object.EnergyLevels[23]:
            ER = object.EnergyLevels[23] / object.EnergyLevels[15]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[23][I] = <float> (0.512) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43F, Y43F, X43F, 4)
            if EN > (2 * object.EnergyLevels[23]):
                object.PEInelasticCrossSectionPerGas[23][I] = object.PEElasticCrossSection[1][I - IOFFN[23]]

        #5 1F SCALED FROM 4 1F
        if EN > object.EnergyLevels[24]:
            ER = object.EnergyLevels[24] / object.EnergyLevels[16]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[24][I] = <float> (0.512) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N41F, Y41F, X41F, 1)
            if EN > (2 * object.EnergyLevels[24]):
                object.PEInelasticCrossSectionPerGas[24][I] = object.PEElasticCrossSection[1][I - IOFFN[24]]

        #5 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.01504
        if EN > object.EnergyLevels[25]:
            ER = object.EnergyLevels[25] / object.EnergyLevels[17]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[25][I] = <float> (0.01504) / <float> (0.02986) * GasUtil.CALInelasticCrossSectionPerGasBEF(EN, ENP,N41P, Y41P, X41P, BETA2,
                                                                                          GAMMA2, ElectronMass2,
                                                                                          object.DEN[I], BBCONST,
                                                                                          object.EnergyLevels[25], object.E[2],
                                                                                          <float> (0.02986))
            if EN > (2 * object.EnergyLevels[25]):
                object.PEInelasticCrossSectionPerGas[25][I] = object.PEElasticCrossSection[1][I - IOFFN[25]]

        #6 3S SCALED FROM 4 3S
        if EN > object.EnergyLevels[26]:
            ER = object.EnergyLevels[26] / object.EnergyLevels[10]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[26][I] = <float> (0.296) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43S, Y43S, X43S, 3)
            if EN > (2 * object.EnergyLevels[26]):
                object.PEInelasticCrossSectionPerGas[26][I] = object.PEElasticCrossSection[1][I - IOFFN[26]]

        #6 1S SCALED FROM 4 1S
        if EN > object.EnergyLevels[27]:
            ER = object.EnergyLevels[27] / object.EnergyLevels[11]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[27][I] = <float> (0.296) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N41S, Y41S, X41S, 1)
            if EN > (2 * object.EnergyLevels[27]):
                object.PEInelasticCrossSectionPerGas[27][I] = object.PEElasticCrossSection[1][I - IOFFN[27]]

        #6 3P SCALED FROM 4 3P
        if EN > object.EnergyLevels[28]:
            ER = object.EnergyLevels[28] / object.EnergyLevels[12]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[28][I] = <float> (0.296) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43P, Y43P, X43P, 3)
            if EN > (2 * object.EnergyLevels[28]):
                object.PEInelasticCrossSectionPerGas[28][I] = object.PEElasticCrossSection[1][I - IOFFN[28]]

        #6 3D SCALED FROM 4 3D
        if EN > object.EnergyLevels[29]:
            ER = object.EnergyLevels[29] / object.EnergyLevels[13]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[29][I] = <float> (0.296) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43D, Y43D, X43D, 3)
            if EN > (2 * object.EnergyLevels[29]):
                object.PEInelasticCrossSectionPerGas[29][I] = object.PEElasticCrossSection[1][I - IOFFN[29]]

        #6 1D SCALED FROM 4 1D
        if EN > object.EnergyLevels[30]:
            ER = object.EnergyLevels[30] / object.EnergyLevels[14]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[30][I] = <float> (0.296) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N41D, Y41D, X41D, 1)
            if EN > (2 * object.EnergyLevels[30]):
                object.PEInelasticCrossSectionPerGas[30][I] = object.PEElasticCrossSection[1][I - IOFFN[30]]

        #6 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.01504
        if EN > object.EnergyLevels[31]:
            ER = object.EnergyLevels[31] / object.EnergyLevels[17]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[31][I] = <float> (0.00863) / <float> (0.02986) * GasUtil.CALInelasticCrossSectionPerGasBEF(EN, ENP,N41P, Y41P, X41P, BETA2,
                                                                                          GAMMA2, ElectronMass2,
                                                                                          object.DEN[I], BBCONST,
                                                                                          object.EnergyLevels[31], object.E[2],
                                                                                          <float> (0.02986))
            if EN > (2 * object.EnergyLevels[31]):
                object.PEInelasticCrossSectionPerGas[31][I] = object.PEElasticCrossSection[1][I - IOFFN[31]]

        #7 3S SCALED FROM 4 3S
        if EN > object.EnergyLevels[32]:
            ER = object.EnergyLevels[32] / object.EnergyLevels[10]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[32][I] = <float> (0.187) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43S, Y43S, X43S, 3)
            if EN > (2 * object.EnergyLevels[32]):
                object.PEInelasticCrossSectionPerGas[32][I] = object.PEElasticCrossSection[1][I - IOFFN[32]]

        #7 1S SCALED FROM 4 1S
        if EN > object.EnergyLevels[33]:
            ER = object.EnergyLevels[33] / object.EnergyLevels[11]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[33][I] = <float> (0.187) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N41S, Y41S, X41S, 1)
            if EN > (2 * object.EnergyLevels[33]):
                object.PEInelasticCrossSectionPerGas[33][I] = object.PEElasticCrossSection[1][I - IOFFN[33]]

        #7 3P SCALED FROM 4 3P
        if EN > object.EnergyLevels[34]:
            ER = object.EnergyLevels[34] / object.EnergyLevels[12]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[34][I] = <float> (0.187) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43P, Y43P, X43P, 3)
            if EN > (2 * object.EnergyLevels[34]):
                object.PEInelasticCrossSectionPerGas[34][I] = object.PEElasticCrossSection[1][I - IOFFN[34]]

        #7 3D SCALED FROM 4 3D
        if EN > object.EnergyLevels[35]:
            ER = object.EnergyLevels[35] / object.EnergyLevels[13]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[35][I] = <float> (0.187) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43D, Y43D, X43D, 3)
            if EN > (2 * object.EnergyLevels[35]):
                object.PEInelasticCrossSectionPerGas[35][I] = object.PEElasticCrossSection[1][I - IOFFN[35]]

        #7 1D SCALED FROM 4 1D
        if EN > object.EnergyLevels[36]:
            ER = object.EnergyLevels[36] / object.EnergyLevels[14]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[36][I] = <float> (0.187) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N41D, Y41D, X41D, 1)
            if EN > (2 * object.EnergyLevels[36]):
                object.PEInelasticCrossSectionPerGas[36][I] = object.PEElasticCrossSection[1][I - IOFFN[36]]

        #7 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00540
        if EN > object.EnergyLevels[37]:
            ER = object.EnergyLevels[37] / object.EnergyLevels[17]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[37][I] = <float> (0.00540) / <float> (0.02986) * GasUtil.CALInelasticCrossSectionPerGasBEF(EN, ENP,N41P, Y41P, X41P, BETA2,
                                                                                          GAMMA2, ElectronMass2,
                                                                                          object.DEN[I], BBCONST,
                                                                                          object.EnergyLevels[37], object.E[2],
                                                                                          <float> (0.02986))
            if EN > (2 * object.EnergyLevels[37]):
                object.PEInelasticCrossSectionPerGas[37][I] = object.PEElasticCrossSection[1][I - IOFFN[37]]

        #Sum 3S LEVELS FROM 8 3S HIGHER AND SCALED FROM 4 3S
        if EN > object.EnergyLevels[38]:
            ER = object.EnergyLevels[38] / object.EnergyLevels[10]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[38][I] = <float> (0.553) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43S, Y43S, X43S, 3)
            if EN > (2 * object.EnergyLevels[38]):
                object.PEInelasticCrossSectionPerGas[38][I] = object.PEElasticCrossSection[1][I - IOFFN[38]]

        #Sum 1S LEVELS FROM 8 3S HIGHER AND SCALED FROM 4 1S
        if EN > object.EnergyLevels[39]:
            ER = object.EnergyLevels[39] / object.EnergyLevels[11]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[39][I] = <float> (0.553) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N41S, Y41S, X41S, 1)
            if EN > (2 * object.EnergyLevels[39]):
                object.PEInelasticCrossSectionPerGas[39][I] = object.PEElasticCrossSection[1][I - IOFFN[39]]

        #Sum 3P LEVELS FROM  8 3P HIGHER AND SCALED FROM 4 3P
        if EN > object.EnergyLevels[40]:
            ER = object.EnergyLevels[40] / object.EnergyLevels[12]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[40][I] = <float> (0.553) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43P, Y43P, X43P, 3)
            if EN > (2 * object.EnergyLevels[40]):
                object.PEInelasticCrossSectionPerGas[40][I] = object.PEElasticCrossSection[1][I - IOFFN[40]]

        #Sum 3D LEVELS FROM  8 3D HIGHER AND SCALED FROM 4 3D
        if EN > object.EnergyLevels[41]:
            ER = object.EnergyLevels[41] / object.EnergyLevels[13]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[41][I] = <float> (0.553) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N43D, Y43D, X43D, 3)
            if EN > (2 * object.EnergyLevels[41]):
                object.PEInelasticCrossSectionPerGas[41][I] = object.PEElasticCrossSection[1][I - IOFFN[41]]

        #Sum 1D LEVELS FROM  8 1D HIGHER AND SCALED FROM 4 1D
        if EN > object.EnergyLevels[42]:
            ER = object.EnergyLevels[42] / object.EnergyLevels[14]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[42][I] = <float> (0.553) * GasUtil.CALInelasticCrossSectionPerGasP(ENP, N41D, Y41D, X41D, 1)
            if EN > (2 * object.EnergyLevels[42]):
                object.PEInelasticCrossSectionPerGas[42][I] = object.PEElasticCrossSection[1][I - IOFFN[42]]

        #8 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00362
        if EN > object.EnergyLevels[43]:
            ER = object.EnergyLevels[43] / object.EnergyLevels[17]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[43][I] = <float> (0.00362) / <float> (0.02986) * GasUtil.CALInelasticCrossSectionPerGasBEF(EN,ENP, N41P, Y41P, X41P, BETA2,
                                                                                          GAMMA2, ElectronMass2,
                                                                                          object.DEN[I], BBCONST,
                                                                                          object.EnergyLevels[43], object.E[2],
                                                                                          <float> (0.02986))
            if EN > (2 * object.EnergyLevels[43]):
                object.PEInelasticCrossSectionPerGas[43][I] = object.PEElasticCrossSection[1][I - IOFFN[43]]

        #9 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00253
        if EN > object.EnergyLevels[44]:
            ER = object.EnergyLevels[44] / object.EnergyLevels[17]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[44][I] = <float> (0.00253) / <float> (0.02986) * GasUtil.CALInelasticCrossSectionPerGasBEF(EN,ENP, N41P, Y41P, X41P, BETA2,
                                                                                          GAMMA2, ElectronMass2,
                                                                                          object.DEN[I], BBCONST,
                                                                                          object.EnergyLevels[44], object.E[2],
                                                                                          <float> (0.02986))
            if EN > (2 * object.EnergyLevels[44]):
                object.PEInelasticCrossSectionPerGas[44][I] = object.PEElasticCrossSection[1][I - IOFFN[44]]

        #10 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00184
        if EN > object.EnergyLevels[45]:
            ER = object.EnergyLevels[45] / object.EnergyLevels[17]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[45][I] = <float> (0.00184) / <float> (0.02986) * GasUtil.CALInelasticCrossSectionPerGasBEF(EN, ENP, N41P, Y41P, X41P,
                                                                                          BETA2, GAMMA2, ElectronMass2,
                                                                                          object.DEN[I], BBCONST,
                                                                                          object.EnergyLevels[45], object.E[2],
                                                                                          <float> (0.02986))
            if EN > (2 * object.EnergyLevels[45]):
                object.PEInelasticCrossSectionPerGas[45][I] = object.PEElasticCrossSection[1][I - IOFFN[45]]

        #11 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00138
        if EN > object.EnergyLevels[46]:
            ER = object.EnergyLevels[46] / object.EnergyLevels[17]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[46][I] = <float> (0.00138) / <float> (0.02986) * GasUtil.CALInelasticCrossSectionPerGasBEF(EN, ENP, N41P, Y41P, X41P,
                                                                                          BETA2, GAMMA2, ElectronMass2,
                                                                                          object.DEN[I], BBCONST,
                                                                                          object.EnergyLevels[46], object.E[2],
                                                                                          <float> (0.02986))
            if EN > (2 * object.EnergyLevels[46]):
                object.PEInelasticCrossSectionPerGas[46][I] = object.PEElasticCrossSection[1][I - IOFFN[46]]

        #12 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00106
        if EN > object.EnergyLevels[47]:
            ER = object.EnergyLevels[47] / object.EnergyLevels[17]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[47][I] = <float> (0.00106) / <float> (0.02986) * GasUtil.CALInelasticCrossSectionPerGasBEF(EN, ENP, N41P, Y41P, X41P,
                                                                                          BETA2, GAMMA2, ElectronMass2,
                                                                                          object.DEN[I], BBCONST,
                                                                                          object.EnergyLevels[47], object.E[2],
                                                                                          <float> (0.02986))
            if EN > (2 * object.EnergyLevels[47]):
                object.PEInelasticCrossSectionPerGas[47][I] = object.PEElasticCrossSection[1][I - IOFFN[47]]

        #13 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00440
        if EN > object.EnergyLevels[48]:
            ER = object.EnergyLevels[48] / object.EnergyLevels[17]
            ENP = EN / ER
            object.InelasticCrossSectionPerGas[48][I] = <float> (0.00440) / <float> (0.02986) * GasUtil.CALInelasticCrossSectionPerGasBEF(EN, ENP, N41P, Y41P, X41P,
                                                                                          BETA2, GAMMA2, ElectronMass2,
                                                                                          object.DEN[I], BBCONST,
                                                                                          object.EnergyLevels[48], object.E[2],
                                                                                          <float> (0.02986))
            if EN > (2 * object.EnergyLevels[48]):
                object.PEInelasticCrossSectionPerGas[48][I] = object.PEElasticCrossSection[1][I - IOFFN[48]]

        QMET = object.InelasticCrossSectionPerGas[0][I] + object.InelasticCrossSectionPerGas[1][I]
        QDIP = object.InelasticCrossSectionPerGas[3][I] + object.InelasticCrossSectionPerGas[9][I] + object.InelasticCrossSectionPerGas[17][I] + object.InelasticCrossSectionPerGas[25][I] + object.InelasticCrossSectionPerGas[31][I] + \
               object.InelasticCrossSectionPerGas[37][I] + object.InelasticCrossSectionPerGas[43][I] + object.InelasticCrossSectionPerGas[44][I] + object.InelasticCrossSectionPerGas[45][I] + object.InelasticCrossSectionPerGas[46][I] + \
               object.InelasticCrossSectionPerGas[47][I] + object.InelasticCrossSectionPerGas[48][I]
        QTRP = object.InelasticCrossSectionPerGas[0][I] + object.InelasticCrossSectionPerGas[2][I] + object.InelasticCrossSectionPerGas[4][I] + object.InelasticCrossSectionPerGas[6][I] + object.InelasticCrossSectionPerGas[7][I] + \
               object.InelasticCrossSectionPerGas[10][I] + object.InelasticCrossSectionPerGas[12][I] + object.InelasticCrossSectionPerGas[13][I] + object.InelasticCrossSectionPerGas[15][I] + object.InelasticCrossSectionPerGas[18][I] + \
               object.InelasticCrossSectionPerGas[20][I] + object.InelasticCrossSectionPerGas[21][I] + object.InelasticCrossSectionPerGas[23][I] + object.InelasticCrossSectionPerGas[26][I] + object.InelasticCrossSectionPerGas[28][I] + \
               object.InelasticCrossSectionPerGas[29][I] + object.InelasticCrossSectionPerGas[32][I] + object.InelasticCrossSectionPerGas[34][I] + object.InelasticCrossSectionPerGas[35][I] + object.InelasticCrossSectionPerGas[38][I] + \
               object.InelasticCrossSectionPerGas[40][I] + object.InelasticCrossSectionPerGas[41][I]
        QSNG = object.InelasticCrossSectionPerGas[1][I] + object.InelasticCrossSectionPerGas[3][I] + object.InelasticCrossSectionPerGas[5][I] + object.InelasticCrossSectionPerGas[8][I] + object.InelasticCrossSectionPerGas[9][I] + \
               object.InelasticCrossSectionPerGas[11][I] + object.InelasticCrossSectionPerGas[14][I] + object.InelasticCrossSectionPerGas[16][I] + object.InelasticCrossSectionPerGas[17][I] + object.InelasticCrossSectionPerGas[19][I] + \
               object.InelasticCrossSectionPerGas[22][I] + object.InelasticCrossSectionPerGas[24][I] + object.InelasticCrossSectionPerGas[25][I] + object.InelasticCrossSectionPerGas[27][I] + object.InelasticCrossSectionPerGas[30][I] + \
               object.InelasticCrossSectionPerGas[31][I] + object.InelasticCrossSectionPerGas[33][I] + object.InelasticCrossSectionPerGas[36][I] + object.InelasticCrossSectionPerGas[37][I] + object.InelasticCrossSectionPerGas[39][I] + \
               object.InelasticCrossSectionPerGas[42][I] + object.InelasticCrossSectionPerGas[43][I] + object.InelasticCrossSectionPerGas[44][I] + object.InelasticCrossSectionPerGas[45][I] + object.InelasticCrossSectionPerGas[46][I] + \
               object.InelasticCrossSectionPerGas[47][I] + object.InelasticCrossSectionPerGas[48][I]
        InelasticCrossSection = QSNG + QTRP + object.IonizationCrossSection[0][I] + object.IonizationCrossSection[1][I]
        object.Q[0][I] = ElasticCrossSectionA + InelasticCrossSection


    for J in range(object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break
