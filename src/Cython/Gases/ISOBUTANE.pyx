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

    cdef double A0, RY, CONST, EMASS2, API, BBCONST, AM2, C, AUGK, ASING

    # BORN-BETHE CONSTANTS
    A0 = 0.52917720859e-08
    RY = 13.60569193
    CONST = 1.873884e-20
    EMASS2 = 1021997.804
    API = acos(-1.0e0)
    BBCONST = 16.0e0 * API * A0 * A0 * RY * RY / EMASS2
    # BORN BETHE VALUES FOR IONISATION
    AM2 = 14.8
    C = 141.9

    # AVERAGE AUGER EMISSIONS FROM K-SHELL
    AUGK = 2.0

    # SCALE SINGLET X-SECTIONS TO ALLOW FOR INCREASED ENERGY LOSS DUE TO 5%
    #   STEP IN ENERGY BETWEEN SINGLET LEVELS.
    ASING = 1.02

    object.NION = 3
    object.NATT = 1
    object.NIN = 24
    object.NNULL = 0

    cdef int NBREM, i, j, I, J, NL, NDATA, NIOND, NATT1, NVIB1, NVIB2, NVIB3, NVIB4, NVIB5, NEXC1, NEXC2, NKSH
    NBREM = 25

    # USE WhichAngularModel=2 ONLY (OKHRIMOVSKY)
    for i in range(6):
        object.KEL[i] = object.WhichAngularModel
    for i in range(object.NIN):
        object.KIN[i] = 2

    NDATA = 157
    NIOND = 42
    NATT1 = 10
    NVIB1 = 30
    NVIB2 = 24
    NVIB3 = 24
    NVIB4 = 29
    NVIB5 = 15
    NEXC1 = 16
    NEXC2 = 16
    NKSH = 83

    cdef double EMASS = 9.10938291e-31,PENSUM
    cdef double AMU = 1.660538921e-27, EOBY[3], SCLOBY, QCOUNT = 0.0, QIONC, QIONG
    cdef double APOP1,APOP2,APOP3,APOP4,APOP5,APOP6,APOP7,APOP8,APOP9,APOP10,APOPGST,APOPSUM,APOPV2,APOPV3,APOPV4
    cdef double APOPGS


    object.E = [0.0, 1.0, 10.67, 0.0, 0.0, 7.0]
    object.E[1] = 2.0 * EMASS / (58.1234 * AMU)

    object.EION[0:3] = [10.67, 17.0, 285.0]

    #OPAL BEATY IONISATION  AT LOW ENERGY 0
    #OPAL BEATY FOR DISSOCIATION AND K-SHELL 1,2
    EOBY[0:3] = [6.8, 6.8, 180.0]

    object.NC0[0:3] = [0, 0, 2]
    object.EC0[0:3] = [0.0, 0.0, 253.0]
    object.WK[0:3] = [0.0, 0.0, 0.0026]
    object.EFL[0:3] = [0.0, 0.0, 273.0]
    object.NG1[0:3] = [0, 0, 1]
    object.EG1[0:3] = [0.0, 0.0, 253.0]
    object.NG2[0:3] = [0, 0, 2]
    object.EG2[0:3] = [0.0, 0.0, 5.0]
    object.EIN = gd['gas11/EIN']

    for j in range(0, object.NION):
        for i in range(0, 4000):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i - 1
                break

    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION
    for NL in range(object.NIN):
        for i in range(4000):
            if object.EG[i] > object.EIN[NL]:
                IOFFN[NL] = i - 1
                break

    for i in range(object.NIN):
        for j in range(3):
            object.PenningFraction[j][i]
    
    for i in range(9,24):
        object.PenningFraction[0][i]=0.0
        object.PenningFraction[1][i]=1.0
        object.PenningFraction[2][i]=1.0

    APOP1=exp(object.EIN[0]/object.AKT)
    APOP2=exp(2.0*object.EIN[0]/object.AKT)
    APOP3=exp(3.0*object.EIN[0]/object.AKT)
    APOP4=exp(4.0*object.EIN[0]/object.AKT)
    APOP5=exp(5.0*object.EIN[0]/object.AKT)
    APOP6=exp(6.0*object.EIN[0]/object.AKT)
    APOP7=exp(7.0*object.EIN[0]/object.AKT)
    APOP8=exp(8.0*object.EIN[0]/object.AKT)
    APOP9=exp(9.0*object.EIN[0]/object.AKT)
    APOP10=exp(10.0*object.EIN[0]/object.AKT)
    APOPGST=1.0
    APOPSUM=APOPGST+APOP1+APOP2+APOP3+APOP4+APOP5+APOP6+APOP7+APOP8+APOP9+APOP10
    APOPGST=1.0/APOPSUM
    APOP1=APOP1/APOPSUM
    APOP2=APOP2/APOPSUM
    APOP3=APOP3/APOPSUM
    APOP4=APOP4/APOPSUM
    APOP5=APOP5/APOPSUM
    APOP6=APOP6/APOPSUM
    APOP7=APOP7/APOPSUM
    APOP8=APOP8/APOPSUM
    APOP9=APOP9/APOPSUM
    APOP10=APOP10/APOPSUM
    #  USE 2 LEVEL APPROXIMATION FOR TORSION
    APOP1=APOP1+APOP2+APOP3+APOP4+APOP5+APOP6+APOP7+APOP8+APOP9+APOP10
    APOPGST=1.0
    # CALCULATE POPULATION  OF VIBRATIONAL STATES
    # ASSUME ALL STATE DEGENERACIES ARE EQUAL
    APOPV2=exp(object.EIN[2]/object.AKT)
    APOPV3=exp(object.EIN[4]/object.AKT)
    APOPV4=exp(object.EIN[6]/object.AKT)
    APOPGS=1.0+APOPV2+APOPV3+APOPV4
    APOPV2=APOPV2/APOPGS
    APOPV3=APOPV3/APOPGS
    APOPV4=APOPV4/APOPGS
    APOPGS=1.0/APOPGS
    # RENORMALISE GROUND STATE POPULATION ( GIVES CORRECTION THAT
    # ALLOWS FOR VIBRATIONAL EXCITATION FROM EXCITED VIBRATIONAL STATES)
    APOPGS=1.0
    
    cdef double EN, GAMMA1, GAMMA2, BETA, BETA2, QMT, QEL, PQ[3], X1, X2, QBB = 0.0, QSUM, EFAC,F[13],QSNG,QTOTEXC,QTRP

    F = [0.00131,0.0150,0.114,0.157,0.171,0.188,0.205,0.193,0.162,0.103,0.067,0.064,0.028]
    cdef int FI
    
    for I in range(object.EnergySteps):
        EN = object.EG[I]
        ENLG = log(EN)
        GAMMA1 = (EMASS2 + 2 * EN) / EMASS2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA

        QMOM = GasUtil.CALQIONREG(EN, NDATA, YELM, XEN)
        QELA = GasUtil.CALQIONREG(EN, NDATA, YELT, XEN)
        PQ[2] = GasUtil.CALPQ3(EN, NDATA, YEPS, XEN)

        PQ[2] = 1-PQ[2]
        PQ[1] = 0.5 + (QELA-QMOM) / QELA
        PQ[0] = 0.5

        object.PEQEL[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = QELA
        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM

        # GROSS IONISATION
        object.QION[0][I] = 0.0
        object.PEQION[0][I] = 0.5
        if object.WhichAngularModel ==2:
            object.PEQION[0][I] = 0.0
        if EN > object.EION[0]:
            object.QION[0][I] = GasUtil.CALQIONX(EN, NIOND, YION,XION,BETA2,1,CONST, object.DEN[I],C, AM2)
        # USE AAnisotropicDetectedTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON AT
        # ENERGIES ABOVE  2 * IONISATION ENERGY
        # AAnisotropicDetectedTROPIC ANGULAR DISTRIBUTION EQUAL TO ELASTIC ANGULAR DISTRIBUTION
        # AT AN ENERGY OFFSET BY THE IONISATION ENERGY
        if EN > 2*object.EION[0]:
            object.PEQION[0][I] = object.PEQEL[1][I-IOFFION[0]]
        
        # CALCULATE IONISATION-EXCITATION AND SPLIT IONISATION INTO
        # IONISATION ONLY AND IONISATION +EXCITATION        
        object.QION[1][I] = 0.0
        object.PEQION[1][I] = 0.5
        if object.WhichAngularModel ==2:
            object.PEQION[1][I] = 0.0
        if EN > object.EION[1]:
            object.QION[1][I] = 12.0 / (object.EION[1] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EION[1])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EION[1] + object.E[2])
            if object.QION[J][I]<0.0:
                object.QION[J][I] = 0.0
            # FIND IONISATION ONLY
            object.QION[0][I] -= object.QION[1][I]
        if EN > 2*object.EION[1]:
            object.PEQION[1][I] = object.PEQEL[1][I-IOFFION[1]]

        # K-shell IONISATION
        object.QION[2][I] = 0.0
        object.PEQION[2][I] = 0.5
        if object.WhichAngularModel ==2:
            object.PEQION[2][I] = 0.0
        if EN > object.EION[2]:
            object.QION[2][I] = GasUtil.CALQIONREG(EN, NKSH, YKSH, XKSH) * 4.0
        # USE AAnisotropicDetectedTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON AT
        # ENERGIES ABOVE  2 * IONISATION ENERGY
        # AAnisotropicDetectedTROPIC ANGULAR DISTRIBUTION EQUAL TO ELASTIC ANGULAR DISTRIBUTION
        # AT AN ENERGY OFFSET BY THE IONISATION ENERGY
        if EN > 2*object.EION[2]:
            object.PEQION[2][I] = object.PEQEL[1][I-IOFFION[2]]

        # CORRECT DISSOCIATIVE IONISATION FOR SPLIT INTO K-SHELL
        object.QION[1][I]-=object.QION[2][I]
        # ATTACHMENT (NO ATTACHMENT)
        object.Q[3][I] = 0.0
        object.QATT[0][I] = object.Q[3][I]

        # COUNTING IONISATION
        object.Q[4][I] = 0.0
        object.PEQEL[4][I] = 0.5
        if object.WhichAngularModel ==2:
            object.PEQEL[4][I] = 0.0
        if EN > object.E[2]:
            # SET COUNTING IONISATION = GROSS IONISATION (LACK OF EXPERIMENTAL DATA)
            object.Q[4][I] = object.QION[0][I] + AUGK * object.QION[2][I]
            object.Q[4][I]-=object.QION[2][I]
        if EN > 2*object.E[2]:
            object.PEQEL[4][I] = object.PEQEL[1][I-IOFFION[4]]

        object.Q[5][I] = 0.0
        for J in range(10):
            object.QIN[J][I]=0.0
            object.PEQIN[J][I] =0.5
            if object.WhichAngularModel == 2:
                object.PEQIN[J][I] = 0.0

        # SUPERELASTIC TORSION
        if EN != 0.0:
            EFAC = sqrt(1.0 - (object.EIN[0] / EN))
            object.QIN[0][I] = 0.009 * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.QIN[0][I] *= APOP1 * 1.e-16
        if EN > 5* abs(object.EIN[0]):
            if object.WhichAngularModel ==2:
                object.PEQIN[0][I] = object.PEQEL[1][I - IOFFN[0]]

        # TORSION
        if EN > object.EIN[1]:
            EFAC = sqrt(1.0 - (object.EIN[1] / EN))
            object.QIN[1][I] = 0.009 * log((EFAC + 1.0) / (1.0-EFAC)) / EN
            object.QIN[1][I] *= APOPGST * 1.e-16
        if EN > 5* abs(object.EIN[1]):
            if object.WhichAngularModel ==2:
                object.PEQIN[1][I] = object.PEQEL[1][I - IOFFN[1]]

        # SUPERELASTIC VIB BEND MODES
        if EN != 0.0:
            if EN <= XVIB2[NVIB2-1]:
                object.QIN[2][I] = GasUtil.CALQINVISO(EN, NVIB2, YVIB2, XVIB2, APOPV2, object.EIN[3], 1,
                                                  -1 * 5 * EN, 0)
            else:
                object.QIN[2][I] = APOPV2*YVIB2[NVIB2-1]*(XVIB2[NVIB2-1]/EN)*1e-16
        if EN > (3.0 * abs(object.EIN[2])):
            if object.WhichAngularModel==2:
                object.PEQIN[2][I] = object.PEQEL[1][I - IOFFN[2]]

        if EN > object.EIN[3]:
            object.QIN[3][I] = GasUtil.CALQINP(EN, NVIB2,YVIB2, XVIB2, 1) * APOPGS * 100
        if EN > (3.0 * abs(object.EIN[2])):
            if object.WhichAngularModel==2:
                object.PEQIN[3][I] = object.PEQEL[1][I - IOFFN[3]]

        # SUPERELASTIC VIB STRETCH MODES
        if EN != 0.0:
            if EN <= XVIB3[NVIB3-1]:
                object.QIN[4][I] = GasUtil.CALQINVISO(EN, NVIB3, YVIB3, XVIB3, APOPV3, object.EIN[5], 1,
                                                  -1 * 5 * EN, 0)
            else:
                object.QIN[4][I] = APOPV3*YVIB3[NVIB3-1]*(XVIB3[NVIB3-1]/EN)*1e-16
        if EN > (3.0 * abs(object.EIN[5])):
            if object.WhichAngularModel==2:
                object.PEQIN[4][I] = object.PEQEL[1][I - IOFFN[4]]

        # VIB STRETCH MODES
        if EN > object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQINP(EN, NVIB3,YVIB3, XVIB3, 1) * APOPGS * 100
        if EN > (3.0 * abs(object.EIN[5])):
            if object.WhichAngularModel==2:
                object.PEQIN[5][I] = object.PEQEL[1][I - IOFFN[5]]

        # SUPERELASTIC VIB STRETCH MODES
        if EN != 0.0:
            if EN <= XVIB4[NVIB4-1]:
                object.QIN[6][I] = GasUtil.CALQINVISO(EN, NVIB4, YVIB4, XVIB4, APOPV4, object.EIN[7], 1,
                                                  -1 * 5 * EN, 0)
            else:
                object.QIN[6][I] = APOPV4*YVIB4[NVIB4-1]*(XVIB4[NVIB4-1]/EN)*1e-16
        if EN > (3.0 * abs(object.EIN[6])):
            if object.WhichAngularModel==2:
                object.PEQIN[6][I] = object.PEQEL[1][I - IOFFN[6]]

        # VIB STRETCH MODES
        if EN > object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQINP(EN, NVIB4,YVIB4, XVIB4, 1) * APOPGS * 100
        if EN > (3.0 * abs(object.EIN[7])):
            if object.WhichAngularModel==2:
                object.PEQIN[7][I] = object.PEQEL[1][I - IOFFN[7]]

        # EXCITATION    TRIPLET  ABOVE XEXC1(NEXC1) SCALE BY 1/EN**3
        if EN > object.EIN[9]:
            object.QIN[9][I] = GasUtil.CALQINP(EN, NEXC1,YEXC1, XEXC1, 3) * 100
        if EN > 2.0 *object.EIN[9]:
            object.PEQIN[9][I] = object.PEQEL[1][I - IOFFN[9]]

        # EXCITATION    TRIPLET  ABOVE XEXC2(NEXC2) SCALE BY 1/EN**3
        if EN > object.EIN[10]:
            object.QIN[10][I] = GasUtil.CALQINP(EN, NEXC2,YEXC2, XEXC2, 3) * 100
        if EN > 2.0 *object.EIN[10]:
            object.PEQIN[10][I] = object.PEQEL[1][I - IOFFN[10]]
        FI = 0
        # EXCITATION  F = F[FI]
        for J in range(11,24):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                            log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                        I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2]) * ASING
                if object.QIN[J][I]<0.0:
                    object.QIN[J][I] = 0.0
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]

            FI+=1
        QSNG = 0.0
        for J in range(11,24):
            QSNG += object.QIN[J][I]

        QTRP = object.QIN[9][I] + object.QIN[10][I]

        QTOTEXC = QTRP+QSNG
        object.Q[0][I]  = 0.0
        for J in range(9):
            object.Q[0][I] +=object.QIN[J][I]
        # TODO: ERROR IN FORTRAN ?
        #object.Q[0][I] += QTOTEXC

        object.Q[0][I] += object.Q[1][I] +object.Q[3][I] + object.Q[4][I]

    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
    return
