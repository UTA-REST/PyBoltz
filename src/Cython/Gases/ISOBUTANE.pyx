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
cdef void Gas11(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Isobutane gas.
    """
    gd = np.load('gases.npy').item()
    cdef double XEN[157], YELM[157], YELT[157], YEPS[157], XION[42], YION[42], YINC[42], XATT[10], YATT[10], XKSH[83], YKSH[83]
    cdef double XVIB1[30], YVIB1[30], XVIB2[24], YVIB2[24], XVIB3[24], YVIB3[24], XVIB4[29], YVIB4[29], XVIB5[15], YVIB5[15]
    cdef double XEXC1[16], YEXC1[16], XEXC2[16], YEXC2[16], Z6T[25], Z1T[25], EBRM[25]
    cdef int IOFFN[24], IOFFION[10]
    XEN = gd['gas11/XEN']
    YELM = gd['gas11/YELM']
    YELT = gd['gas11/YELT']
    YEPS = gd['gas11/YEPS']
    XION = gd['gas11/XION']
    YION = gd['gas11/YION']
    YINC = gd['gas11/YINC']
    XATT = gd['gas11/XATT']
    YATT = gd['gas11/YATT']
    XKSH = gd['gas11/XKSH']
    YKSH = gd['gas11/YKSH']
    XVIB1 = gd['gas11/XVIB1']
    YVIB1 = gd['gas11/YVIB1']
    XVIB2 = gd['gas11/XVIB2']
    YVIB2 = gd['gas11/YVIB2']
    XVIB3 = gd['gas11/XVIB3']
    YVIB3 = gd['gas11/YVIB3']
    XVIB4 = gd['gas11/XVIB4']
    YVIB4 = gd['gas11/YVIB4']
    XVIB5 = gd['gas11/XVIB5']
    YVIB5 = gd['gas11/YVIB5']
    XEXC1 = gd['gas11/XEXC1']
    YEXC1 = gd['gas11/YEXC1']
    XEXC2 = gd['gas11/XEXC2']
    YEXC2 = gd['gas11/YEXC2']
    Z6T = gd['gas11/Z6T']
    Z1T = gd['gas11/Z1T']
    EBRM = gd['gas11/EBRM']
    object.EnergyLevels = gd['gas11/EnergyLevels']
    cdef double A0, RY, CONST, ElectronMass2, API, BBCONST, AM2, C, AUGK, ASING

    # BORN-BETHE CONSTANTS
    A0 = 0.52917720859e-08
    RY = <float>(13.60569193)
    CONST = 1.873884e-20
    ElectronMass2 = <float>(1021997.804)
    API = acos(-1.0e0)
    BBCONST = 16.0 * API * A0 * A0 * RY * RY / ElectronMass2
    # BORN BETHE VALUES FOR IONISATION
    AM2 = <float>(14.8)
    C = <float>(141.9)

    # AVERAGE AUGER EMISSIONS FROM K-SHELL
    AUGK = 2.0

    # SCALE SINGLET X-SECTIONS TO ALLOW FOR INCREASED ENERGY LOSS DUE TO 5%
    #   STEP IN ENERGY BETWEEN SINGLET LEVELS.
    ASING = <float>(1.02)

    object.N_Ionization = 3
    object.N_Attachment = 1
    object.N_Inelastic = 24
    object.N_Null = 0

    cdef int NBREM, i, j, I, J, NL, NDATA, N_IonizationD, N_Attachment1, NVIB1, NVIB2, NVIB3, NVIB4, NVIB5, NEXC1, NEXC2, NKSH
    NBREM = 25

    # USE WhichAngularModel=2 ONLY (OKHRIMOVSKY)
    for i in range(6):
        object.KEL[i] = object.WhichAngularModel
    for i in range(object.N_Inelastic):
        object.KIN[i] = 2

    NDATA = 157
    N_IonizationD = 42
    N_Attachment1 = 10
    NVIB1 = 30
    NVIB2 = 24
    NVIB3 = 24
    NVIB4 = 29
    NVIB5 = 15
    NEXC1 = 16
    NEXC2 = 16
    NKSH = 83

    cdef double ElectronMass = 9.10938291e-31,PENSum
    cdef double AMU = 1.660538921e-27, EOBY[3], SCLOBY, QCOUNT = 0.0, IonizationCrossSectionC, IonizationCrossSectionG
    cdef double APOP1,APOP2,APOP3,APOP4,APOP5,APOP6,APOP7,APOP8,APOP9,APOP10,APOPGST,APOPSum,APOPV2,APOPV3,APOPV4
    cdef double APOPGS


    object.E = [0.0, 1.0, <float>(10.67), 0.0, 0.0, 7.0]
    object.E[1] = 2.0 * ElectronMass / (<float>(58.1234) * AMU)

    object.IonizationEnergy[0:3] = [<float>(10.67), 17.0, 285.0]

    #OPAL BEATY IONISATION  AT LOW ENERGY 0
    #OPAL BEATY FOR DISSOCIATION AND K-SHELL 1,2
    EOBY[0:3] = [6.8, 6.8, 180.0]

    object.NC0[0:3] = [0, 0, 2]
    object.EC0[0:3] = [0.0, 0.0, 253.0]
    object.WK[0:3] = [0.0, 0.0, <float>(0.0026)]
    object.EFL[0:3] = [0.0, 0.0, 273.0]
    object.NG1[0:3] = [0, 0, 1]
    object.EG1[0:3] = [0.0, 0.0, 253.0]
    object.NG2[0:3] = [0, 0, 2]
    object.EG2[0:3] = [0.0, 0.0, 5.0]

    for j in range(0, object.N_Ionization):
        for i in range(0, 4000):
            if (object.EG[i] > object.IonizationEnergy[j]):
                IOFFION[j] = i
                break

    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION
    for NL in range(object.N_Inelastic):
        for i in range(4000):
            if object.EG[i] > abs(object.EnergyLevels[NL]):
                IOFFN[NL] = i
                break

    for i in range(24):
        for j in range(3):

            object.PenningFraction[j][i]

    
    for i in range(9,24):
        object.PenningFraction[0][i]=0.0
        object.PenningFraction[1][i]=1.0
        object.PenningFraction[2][i]=1.0

    APOP1=exp(object.EnergyLevels[0]/object.ThermalEnergy)
    APOP2=exp(2.0*object.EnergyLevels[0]/object.ThermalEnergy)
    APOP3=exp(3.0*object.EnergyLevels[0]/object.ThermalEnergy)
    APOP4=exp(4.0*object.EnergyLevels[0]/object.ThermalEnergy)
    APOP5=exp(5.0*object.EnergyLevels[0]/object.ThermalEnergy)
    APOP6=exp(6.0*object.EnergyLevels[0]/object.ThermalEnergy)
    APOP7=exp(7.0*object.EnergyLevels[0]/object.ThermalEnergy)
    APOP8=exp(8.0*object.EnergyLevels[0]/object.ThermalEnergy)
    APOP9=exp(9.0*object.EnergyLevels[0]/object.ThermalEnergy)
    APOP10=exp(10.0*object.EnergyLevels[0]/object.ThermalEnergy)
    APOPGST=1.0
    APOPSum=APOPGST+APOP1+APOP2+APOP3+APOP4+APOP5+APOP6+APOP7+APOP8+APOP9+APOP10
    APOPGST=1.0/APOPSum
    APOP1=APOP1/APOPSum
    APOP2=APOP2/APOPSum
    APOP3=APOP3/APOPSum
    APOP4=APOP4/APOPSum
    APOP5=APOP5/APOPSum
    APOP6=APOP6/APOPSum
    APOP7=APOP7/APOPSum
    APOP8=APOP8/APOPSum
    APOP9=APOP9/APOPSum
    APOP10=APOP10/APOPSum
    #  USE 2 LEVEL APPROXIMATION FOR TORSION
    APOP1=APOP1+APOP2+APOP3+APOP4+APOP5+APOP6+APOP7+APOP8+APOP9+APOP10
    APOPGST=1.0
    # CALCULATE POPULATION  OF VIBRATIONAL STATES
    # ASSumE ALL STATE DEGENERACIES ARE EQUAL
    APOPV2=exp(object.EnergyLevels[2]/object.ThermalEnergy)
    APOPV3=exp(object.EnergyLevels[4]/object.ThermalEnergy)
    APOPV4=exp(object.EnergyLevels[6]/object.ThermalEnergy)
    APOPGS=1.0+APOPV2+APOPV3+APOPV4
    APOPV2=APOPV2/APOPGS
    APOPV3=APOPV3/APOPGS
    APOPV4=APOPV4/APOPGS
    APOPGS=1.0/APOPGS
    # RENORMALISE GROUND STATE POPULATION ( GIVES CORRECTION THAT
    # ALLOWS FOR VIBRATIONAL EXCITATION FROM EXCITED VIBRATIONAL STATES)
    APOPGS=1.0
    
    cdef double EN, GAMMA1, GAMMA2, BETA, BETA2, QMT, ElasticCrossSection, PQ[3], X1, X2, QBB = 0.0, CrossSectionSum, EFAC,F[13],QSNG,TotalCrossSectionEXC,QTRP

    F = [<float>(0.00131),<float>(0.0150),<float>(0.114),<float>(0.157),<float>(0.171),<float>(0.188),<float>(0.205),<float>(0.193),<float>(0.162),<float>(0.103),<float>(0.067),<float>(0.064),<float>(0.028),]
    cdef int FI
    
    for I in range(4000):
        EN = object.EG[I]
        GAMMA1 = (ElectronMass2 + 2 * EN) / ElectronMass2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA

        QMOM = GasUtil.CALIonizationCrossSectionREG(EN, NDATA, YELM, XEN)
        ElasticCrossSectionA = GasUtil.CALIonizationCrossSectionREG(EN, NDATA, YELT, XEN)
        PQ[2] = GasUtil.CALPQ3(EN, NDATA, YEPS, XEN)

        PQ[2] = 1-PQ[2]
        PQ[1] = 0.5 + (ElasticCrossSectionA-QMOM) / ElasticCrossSectionA
        PQ[0] = 0.5

        object.PEElasticCrossSection[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = ElasticCrossSectionA
        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM

        # GROSS IONISATION
        object.IonizationCrossSection[0][I] = 0.0
        object.PEIonizationCrossSection[0][I] = 0.5
        if object.WhichAngularModel ==2:
            object.PEIonizationCrossSection[0][I] = 0.0
        if EN > object.IonizationEnergy[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationD, YION,XION,BETA2,1,CONST, object.DEN[I],C, AM2)
            # USE AAnisotropicDetectedTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON AT
            # ENERGIES ABOVE  2 * IONISATION ENERGY
            # AAnisotropicDetectedTROPIC ANGULAR DISTRIBUTION EQUAL TO ELASTIC ANGULAR DISTRIBUTION
            # AT AN ENERGY OFFSET BY THE IONISATION ENERGY
            if EN > 2*object.IonizationEnergy[0]:
                object.PEIonizationCrossSection[0][I] = object.PEElasticCrossSection[1][I-IOFFION[0]]
        
        # CALCULATE IONISATION-EXCITATION AND SPLIT IONISATION INTO
        # IONISATION ONLY AND IONISATION +EXCITATION        
        object.IonizationCrossSection[1][I] = 0.0
        object.PEIonizationCrossSection[1][I] = 0.5
        if object.WhichAngularModel ==2:
            object.PEIonizationCrossSection[1][I] = 0.0
        if EN > object.IonizationEnergy[1]:
            object.IonizationCrossSection[1][I] = 12.0 / (object.IonizationEnergy[1] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.IonizationEnergy[1])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.IonizationEnergy[1] + object.E[2])
            if object.IonizationCrossSection[1][I]<0.0:
                object.IonizationCrossSection[1][I] = 0.0
            # FIND IONISATION ONLY
            object.IonizationCrossSection[0][I] -= object.IonizationCrossSection[1][I]
            if EN > 2*object.IonizationEnergy[1]:
                object.PEIonizationCrossSection[1][I] = object.PEElasticCrossSection[1][I-IOFFION[1]]

        # K-shell IONISATION
        object.IonizationCrossSection[2][I] = 0.0
        object.PEIonizationCrossSection[2][I] = 0.5
        if object.WhichAngularModel ==2:
            object.PEIonizationCrossSection[2][I] = 0.0
        if EN > object.IonizationEnergy[2]:
            object.IonizationCrossSection[2][I] = GasUtil.CALIonizationCrossSectionREG(EN, NKSH, YKSH, XKSH) * 4.0
            # USE AAnisotropicDetectedTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON AT
            # ENERGIES ABOVE  2 * IONISATION ENERGY
            # AAnisotropicDetectedTROPIC ANGULAR DISTRIBUTION EQUAL TO ELASTIC ANGULAR DISTRIBUTION
            # AT AN ENERGY OFFSET BY THE IONISATION ENERGY
            if EN > 2*object.IonizationEnergy[2]:
                object.PEIonizationCrossSection[2][I] = object.PEElasticCrossSection[1][I-IOFFION[2]]

        # CORRECT DISSOCIATIVE IONISATION FOR SPLIT INTO K-SHELL
        object.IonizationCrossSection[1][I]-=object.IonizationCrossSection[2][I]
        # ATTACHMENT (NO ATTACHMENT)
        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[0][I] = object.Q[3][I]

        # COUNTING IONISATION
        object.Q[4][I] = 0.0
        object.PEElasticCrossSection[4][I] = 0.5
        if object.WhichAngularModel ==2:
            object.PEElasticCrossSection[4][I] = 0.0
        if EN > object.E[2]:
            # SET COUNTING IONISATION = GROSS IONISATION (LACK OF EXPERIMENTAL DATA)
            object.Q[4][I] = object.IonizationCrossSection[0][I] + AUGK * object.IonizationCrossSection[2][I]
            object.Q[4][I]-=object.IonizationCrossSection[2][I]
            if EN > 2*object.E[2]:
                object.PEElasticCrossSection[4][I] = object.PEElasticCrossSection[1][I-IOFFION[0]]

        object.Q[5][I] = 0.0
        for J in range(11):
            object.InelasticCrossSectionPerGas[J][I]=0.0
            object.PEInelasticCrossSectionPerGas[J][I] =0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[J][I] = 0.0

        # SUPERELASTIC TORSION
        if EN != 0.0:
            EFAC = sqrt(1.0 - (object.EnergyLevels[0] / EN))
            object.InelasticCrossSectionPerGas[0][I] = <float>(0.009) * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.InelasticCrossSectionPerGas[0][I] *= APOP1 * 1.e-16
            if EN > 5* abs(object.EnergyLevels[0]):
                if object.WhichAngularModel ==2:
                    object.PEInelasticCrossSectionPerGas[0][I] = object.PEElasticCrossSection[1][I - IOFFN[0]]

        # TORSION
        if EN > object.EnergyLevels[1]:
            EFAC = sqrt(1.0 - (object.EnergyLevels[1] / EN))
            object.InelasticCrossSectionPerGas[1][I] = <float>(0.009) * log((EFAC + 1.0) / (1.0-EFAC)) / EN
            object.InelasticCrossSectionPerGas[1][I] *= APOPGST * 1.e-16
            if EN > 5* abs(object.EnergyLevels[1]):
                if object.WhichAngularModel ==2:
                    object.PEInelasticCrossSectionPerGas[1][I] = object.PEElasticCrossSection[1][I - IOFFN[1]]

        # SUPERELASTIC VIB BEND MODES
        if EN != 0.0:
            if EN+ object.EnergyLevels[3] <= XVIB2[NVIB2-1]:
                object.InelasticCrossSectionPerGas[2][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB2, YVIB2, XVIB2, APOPV2, object.EnergyLevels[3], 1,
                                                  -1 * 5 * EN, 0)
            else:
                object.InelasticCrossSectionPerGas[2][I] = APOPV2*YVIB2[NVIB2-1]*(XVIB2[NVIB2-1]/EN)*1e-16
            if EN > (3.0 * abs(object.EnergyLevels[2])):
                if object.WhichAngularModel==2:
                    object.PEInelasticCrossSectionPerGas[2][I] = object.PEElasticCrossSection[1][I - IOFFN[2]]

        if EN > object.EnergyLevels[3]:
            object.InelasticCrossSectionPerGas[3][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB2,YVIB2, XVIB2, 1) * APOPGS * 100
            if EN > (3.0 * abs(object.EnergyLevels[3])):
                if object.WhichAngularModel==2:
                    object.PEInelasticCrossSectionPerGas[3][I] = object.PEElasticCrossSection[1][I - IOFFN[3]]

        # SUPERELASTIC VIB STRETCH MODES
        if EN != 0.0:
            if EN+ object.EnergyLevels[5] <= XVIB3[NVIB3-1]:
                object.InelasticCrossSectionPerGas[4][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB3, YVIB3, XVIB3, APOPV3, object.EnergyLevels[5], 1,
                                                  -1 * 5 * EN, 0)
            else:
                object.InelasticCrossSectionPerGas[4][I] = APOPV3*YVIB3[NVIB3-1]*(XVIB3[NVIB3-1]/EN)*1e-16
            if EN > (3.0 * abs(object.EnergyLevels[4])):
                if object.WhichAngularModel==2:
                    object.PEInelasticCrossSectionPerGas[4][I] = object.PEElasticCrossSection[1][I - IOFFN[4]]

        # VIB STRETCH MODES
        if EN > object.EnergyLevels[5]:
            object.InelasticCrossSectionPerGas[5][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB3,YVIB3, XVIB3, 1) * APOPGS * 100
        if EN > (3.0 * abs(object.EnergyLevels[5])):
            if object.WhichAngularModel==2:
                object.PEInelasticCrossSectionPerGas[5][I] = object.PEElasticCrossSection[1][I - IOFFN[5]]

        # SUPERELASTIC VIB STRETCH MODES
        if EN != 0.0:
            if EN + object.EnergyLevels[7] <= XVIB4[NVIB4-1]:
                object.InelasticCrossSectionPerGas[6][I] = GasUtil.CALInelasticCrossSectionPerGasVISO(EN, NVIB4, YVIB4, XVIB4, APOPV4, object.EnergyLevels[7], 1,
                                                  -1 * 5 * EN, 0)
            else:
                object.InelasticCrossSectionPerGas[6][I] = APOPV4*YVIB4[NVIB4-1]*(XVIB4[NVIB4-1]/EN)*1e-16
            if EN > (3.0 * abs(object.EnergyLevels[6])):
                if object.WhichAngularModel==2:
                    object.PEInelasticCrossSectionPerGas[6][I] = object.PEElasticCrossSection[1][I - IOFFN[6]]

        # VIB STRETCH MODES
        if EN > object.EnergyLevels[7]:
            object.InelasticCrossSectionPerGas[7][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB4,YVIB4, XVIB4, 1) * APOPGS * 100
            if EN > (3.0 * abs(object.EnergyLevels[7])):
                if object.WhichAngularModel==2:
                    object.PEInelasticCrossSectionPerGas[7][I] = object.PEElasticCrossSection[1][I - IOFFN[7]]

                # VIB STRETCH MODES
        if EN > object.EnergyLevels[8]:
            object.InelasticCrossSectionPerGas[8][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NVIB5,YVIB5, XVIB5, 1)  * 100
            if EN > (3.0 * abs(object.EnergyLevels[8])):
                if object.WhichAngularModel==2:
                    object.PEInelasticCrossSectionPerGas[8][I] = object.PEElasticCrossSection[1][I - IOFFN[8]]

        # EXCITATION    TRIPLET  ABOVE XEXC1(NEXC1) SCALE BY 1/EN**3
        if EN > object.EnergyLevels[9]:
            object.InelasticCrossSectionPerGas[9][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC1,YEXC1, XEXC1, 3) * 100
            if EN > 2.0 *abs(object.EnergyLevels[9]):
                object.PEInelasticCrossSectionPerGas[9][I] = object.PEElasticCrossSection[1][I - IOFFN[9]]

        # EXCITATION    TRIPLET  ABOVE XEXC2(NEXC2) SCALE BY 1/EN**3
        if EN > object.EnergyLevels[10]:
            object.InelasticCrossSectionPerGas[10][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, NEXC2,YEXC2, XEXC2, 3) * 100
            if EN > 2.0 *abs(object.EnergyLevels[10]):
                object.PEInelasticCrossSectionPerGas[10][I] = object.PEElasticCrossSection[1][I - IOFFN[10]]
        FI = 0
        # EXCITATION  F = F[FI]
        for J in range(11,24):
            object.InelasticCrossSectionPerGas[J][I]=0.0
            object.PEInelasticCrossSectionPerGas[J][I] =0.0
            if EN > object.EnergyLevels[J]:
                object.InelasticCrossSectionPerGas[J][I] = F[FI] / (object.EnergyLevels[J] * BETA2) * (
                            log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[J])) - BETA2 - object.DEN[
                        I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[J] + object.E[2]) * ASING
                if object.InelasticCrossSectionPerGas[J][I]<0.0:
                    object.InelasticCrossSectionPerGas[J][I] = 0.0
                if EN > 2 * abs(object.EnergyLevels[J]):
                    object.PEInelasticCrossSectionPerGas[J][I] = object.PEElasticCrossSection[1][I - IOFFN[J]]

            FI+=1
        QSNG = 0.0
        for J in range(11,24):
            QSNG += object.InelasticCrossSectionPerGas[J][I]

        QTRP = object.InelasticCrossSectionPerGas[9][I] + object.InelasticCrossSectionPerGas[10][I]

        TotalCrossSectionEXC = QTRP+QSNG
        object.Q[0][I]  = 0.0
        # TODO: ERROR IN FORTRAN ?
        #object.Q[0][I] += TotalCrossSectionEXC

        object.Q[0][I] += object.Q[1][I] +object.Q[3][I] + object.Q[4][I]
        for J in range(9):
            object.Q[0][I]+=object.InelasticCrossSectionPerGas[J][I]

    for J in range(6):
        print(object.E[J])
    print("HERE")
    for J in range(object.N_Ionization):
        print(object.IonizationCrossSection[J][3999])
    print(object.FinalEnergy)
    for J in range(13,object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break
    print(object.N_Inelastic)
    sys.exit()
    return
