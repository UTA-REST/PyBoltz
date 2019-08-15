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
cdef void Gas4(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Helium3 gas.
    """
    gd = np.load('gases.npy').item()
    cdef double XEN[144], YEM[144], YEL[144], YEPS[144], XION[90], YION[90], YINC[90], X23S[139], Y23S[139], X21S[128], Y21S[128], X23P[128], Y23P[128]
    cdef double X21P[125], Y21P[125], X33S[106], Y33S[106], X31S[87], Y31S[87], X33P[91], Y33P[91], X33D[108], Y33D[108], X31D[94], Y31D[94]
    cdef double X31P[114], Y31P[114], X43S[59], Y43S[59], X41S[55], Y41S[55], X43P[76], Y43P[76], X43D[65], Y43D[65], X41D[53], Y41D[53]
    cdef double X43F[40], Y43F[40], X41F[57], Y41F[57], X41P[96], Y41P[96], Z2T[25], EBRM[25], EOBY[2]
    cdef int IOFFN[49], IOFFION[2]
    XEN = gd['gas4/XEN']
    YEM = gd['gas4/YEM']
    YEL = gd['gas4/YEL']
    YEPS = gd['gas4/YEPS']
    XION = gd['gas4/XION']
    YION = gd['gas4/YION']
    YINC = gd['gas4/YINC']
    X23S = gd['gas4/X23S']
    Y23S = gd['gas4/Y23S']
    X21S = gd['gas4/X21S']
    Y21S = gd['gas4/Y21S']
    X23P = gd['gas4/X23P']
    Y23P = gd['gas4/Y23P']
    X21P = gd['gas4/X21P']
    Y21P = gd['gas4/Y21P']
    X33S = gd['gas4/X33S']
    Y33S = gd['gas4/Y33S']
    X31S = gd['gas4/X31S']
    Y31S = gd['gas4/Y31S']
    X33P = gd['gas4/X33P']
    Y33P = gd['gas4/Y33P']
    X33D = gd['gas4/X33D']
    Y33D = gd['gas4/Y33D']
    X31D = gd['gas4/X31D']
    Y31D = gd['gas4/Y31D']
    X31P = gd['gas4/X31P']
    Y31P = gd['gas4/Y31P']
    X43S = gd['gas4/X43S']
    Y43S = gd['gas4/Y43S']
    X41S = gd['gas4/X41S']
    Y41S = gd['gas4/Y41S']
    X43P = gd['gas4/X43P']
    Y43P = gd['gas4/Y43P']
    X43D = gd['gas4/X43D']
    Y43D = gd['gas4/Y43D']
    X41D = gd['gas4/X41D']
    Y41D = gd['gas4/Y41D']
    X43F = gd['gas4/X43F']
    Y43F = gd['gas4/Y43F']
    X41F = gd['gas4/X41F']
    Y41F = gd['gas4/Y41F']
    X41P = gd['gas4/X41P']
    Y41P = gd['gas4/Y41P']
    Z2T = gd['gas4/Z2T']
    EBRM = gd['gas4/EBRM']

    cdef double EMASS2 = 1021997.804
    cdef double API = acos(-1)
    cdef double A0 = 0.52917720859e-8
    cdef double RY = 13.60569193
    cdef double BBCONST = 16.0 * API * A0 * A0 * RY * RY / EMASS2
    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27
    # BORN BETHE VALUES FOR IONISATION
    cdef double CONST = 1.873884e-20
    cdef double AM2 = 0.489
    cdef double C = 5.50
    object.NION = 2
    object.NATT = 1
    object.NIN = 49
    object.NNULL = 0
    cdef int NBREM = 25, NDATA = 144, NIOND = 90, N23S = 139, N21S = 128, N23P = 128, N21P = 125, N33S = 106, N31S = 87, N33P = 91, N33D = 108, N31D = 94
    cdef int N31P = 114, N43S = 59, N41S = 55, N43P = 76, N43D = 65, N41D = 53, N43F = 40, N41F = 57, N41P = 96
    cdef int I, i, j
    object.E = [0.0, 2.0 * EMASS / (3.0160 * AMU), 24.58739, 0.0, 0.0, 0.0]

    for i in range(6):
        object.KEL[i] = object.NANISO
    for i in range(object.NIN):
        object.KIN[i] = object.NANISO

    #IONISATION ENERGIES
    object.EION[0] = 24.58739
    object.EION[1] = 79.00515

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

    for j in range(0, object.NION):
        for i in range(0, 4000):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i - 1
                break
    object.EIN[0:50] = [19.81961, 20.61577, 20.96409, 21.21802, 22.71847, 22.92032, 23.00707, 23.07365, 23.07407,
                        23.08702, 23.59396, 23.67357, 23.70789, 23.73609, 23.73633, 23.73701, 23.73701, 23.74207,
                        23.97197, 24.01121, 24.02822, 24.04266, 24.04280, 24.04315, 24.04315, 24.04580, 24.16900,
                        24.19116, 24.20081, 24.20916, 24.20925, 24.21100, 24.28456, 24.29828, 24.30429, 24.30954,
                        24.30960, 24.31071, 24.35810, 24.36718, 24.37116, 24.37468, 24.37472, 24.37547, 24.41989,
                        24.45168, 24.47518, 24.49308, 24.50708, 0.0]
    for I in range(50, 250):
        object.EIN[I] = 0.0

    cdef int NL
    for NL in range(object.NIN):
        object.PENFRA[0][NL] = 1.0
        # PENNING TRANSFER DISTANCE MICRONS
        object.PENFRA[1][NL] = 1.0
        # PENNING TRANSFER TIME PICOSECONDS
        object.PENFRA[2][NL] = 1.0

    for NL in range(object.NIN):
        for I in range(4000):
            if object.EG[I] > object.EIN[NL]:
                IOFFN[NL] = I - 1
                break

    cdef double EN, GAMMA1, GAMMA2, BETA, BETA2, QELA, QMOM, A, B, X1, X2, PQ[3], TEMP, QTEMP1, QTEMP2, ER, ENP, QMET, QDIP, QTRP, QSNG, QINEL

    for I in range(4000):
        EN = object.EG[I]
        if EN > object.EIN[0]:
            GAMMA1 = (EMASS2 + 2.0 * EN) / EMASS2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = sqrt(1.0 - 1.0 / GAMMA2)
            BETA2 = BETA * BETA
        QELA = GasUtil.CALQIONREG(EN, NDATA, YEL, XEN)
        QMOM = GasUtil.CALQIONREG(EN, NDATA, YEM, XEN)

        TEMP = GasUtil.CALPQ3(EN, NDATA, YEPS, XEN)
        PQ = [0.5, 0.5 + (QELA - QMOM) / QELA, 1 - TEMP]

        object.PEQEL[1][I] = PQ[object.NANISO]

        object.Q[1][I] = QELA

        if object.NANISO == 0:
            object.Q[1][I] = QMOM
            object.PEQEL[1][I] = 0.5

        #GROSS IONISATION
        object.QION[0][I] = 0.0
        object.PEQION[0][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[0][I] = 0
        if EN >= object.EION[0]:
            object.QION[0][I] = GasUtil.CALQIONX(EN, NIOND, YION, XION, BETA2, 1 / 0.995, CONST, object.DEN[I], C, AM2)
        # USE ANISOTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
        # ENERGIES ABOVE 2 * IONISATION ENERGY
        # ANISOTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFFSET BY
        # IONISATION ENERGY
        if EN > (2 * object.EION[0]):
            object.PEQION[0][I] = object.PEQEL[1][I - IOFFION[0]]

        #ATTACHMENT
        object.Q[3][I] = 0.0
        object.QATT[0][i] = 0.0

        #COUNTING IONISATION
        object.Q[4][i] = 0.0
        object.PEQION[1][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[1][I] = 0
        if EN >= object.EION[1]:
            object.Q[4][I] = GasUtil.CALQIONX(EN, NIOND, YINC, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
        #USE BORN-BETHE X-SECTION ABOVE XION(NIOND) EV
        if EN > (2 * object.EION[1]):
            object.PEQION[1][I] = object.PEQEL[1][I - IOFFION[1]]

        object.Q[5][I] = 0.0
        QTEMP1 = 2 * object.Q[4][I] - object.QION[0][I]
        QTEMP2 = object.QION[0][I] - object.Q[4][I]
        object.QION[0][I] = QTEMP1
        object.QION[1][I] = QTEMP2
        if object.QION[1][I] < 0.0:
            object.QION[1][I] = 0.0

        for NL in range(object.NIN + 1):
            object.QIN[NL][I] = 0.0
            object.PEQIN[NL][I] = 0.5
            if object.NANISO == 2:
                object.PEQIN[NL][I] = 0.0

        #2 3S
        if EN > object.EIN[0]:
            object.QIN[0][I] = GasUtil.CALQINP(EN, N23S, Y23S, X23S, 3)
        if EN > (2 * object.EIN[0]):
            object.PEQIN[0][I] = object.PEQEL[1][I - IOFFN[0]]

        #2 1S
        if EN > object.EIN[1]:
            object.QIN[1][I] = GasUtil.CALQINP(EN, N21S, Y21S, X21S, 1)
        if EN > (2 * object.EIN[1]):
            object.PEQIN[1][I] = object.PEQEL[1][I - IOFFN[1]]

        #2 3P
        if EN > object.EIN[2]:
            object.QIN[2][I] = GasUtil.CALQINP(EN, N23P, Y23P, X23P, 3)
        if EN > (2 * object.EIN[2]):
            object.PEQIN[2][I] = object.PEQEL[1][I - IOFFN[2]]

        #2 1P
        if EN > object.EIN[3]:
            object.QIN[3][I] = GasUtil.CALQINBEF(EN, N21P, Y21P, X21P, BETA2, GAMMA2, EMASS2, object.DEN[I], BBCONST,
                                                 object.EIN[3], object.E[2], 0.27608)
        if EN > (2 * object.EIN[3]):
            object.PEQIN[3][I] = object.PEQEL[1][I - IOFFN[3]]

        #3 3S
        if EN > object.EIN[4]:
            object.QIN[4][I] = GasUtil.CALQINP(EN, N33S, Y33S, X33S, 3)
        if EN > (2 * object.EIN[4]):
            object.PEQIN[4][I] = object.PEQEL[1][I - IOFFN[4]]

        #3 1S
        if EN > object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQINP(EN, N31S, Y31S, X31S, 1)
        if EN > (2 * object.EIN[5]):
            object.PEQIN[5][I] = object.PEQEL[1][I - IOFFN[5]]

        #3 3P
        if EN > object.EIN[6]:
            object.QIN[6][I] = GasUtil.CALQINP(EN, N33P, Y33P, X33P, 1)
        if EN > (2 * object.EIN[6]):
            object.PEQIN[6][I] = object.PEQEL[1][I - IOFFN[6]]

        #3 3D
        if EN > object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQINP(EN, N33D, Y33D, X33D, 1)
        if EN > (2 * object.EIN[7]):
            object.PEQIN[7][I] = object.PEQEL[1][I - IOFFN[7]]

        #3 1D
        if EN > object.EIN[8]:
            object.QIN[8][I] = GasUtil.CALQINP(EN, N31D, Y31D, X31D, 1)
        if EN > (2 * object.EIN[8]):
            object.PEQIN[8][I] = object.PEQEL[1][I - IOFFN[8]]

        #3 1P
        if EN > object.EIN[9]:
            object.QIN[9][I] = GasUtil.CALQINBEF(EN, N31P, Y31P, X31P, BETA2, GAMMA2, EMASS2, object.DEN[I], BBCONST,
                                                 object.EIN[9], object.E[2], 0.07342)
        if EN > (2 * object.EIN[9]):
            object.PEQIN[9][I] = object.PEQEL[1][I - IOFFN[9]]

        #4 3S
        if EN > object.EIN[10]:
            object.QIN[10][I] = GasUtil.CALQINP(EN, N43S, Y43S, X43S, 3)
        if EN > (2 * object.EIN[10]):
            object.PEQIN[10][I] = object.PEQEL[1][I - IOFFN[10]]

        #4 1S
        if EN > object.EIN[11]:
            object.QIN[11][I] = GasUtil.CALQINP(EN, N41S, Y41S, X41S, 1)
        if EN > (2 * object.EIN[11]):
            object.PEQIN[11][I] = object.PEQEL[1][I - IOFFN[11]]

        #4 3P
        if EN > object.EIN[12]:
            object.QIN[12][I] = GasUtil.CALQINP(EN, N43P, Y43P, X43P, 3)
        if EN > (2 * object.EIN[12]):
            object.PEQIN[12][I] = object.PEQEL[1][I - IOFFN[12]]

        #4 3D
        if EN > object.EIN[13]:
            object.QIN[13][I] = GasUtil.CALQINP(EN, N43D, Y43D, X43D, 3)
        if EN > (2 * object.EIN[13]):
            object.PEQIN[13][I] = object.PEQEL[1][I - IOFFN[13]]

        #4 1D
        if EN > object.EIN[14]:
            object.QIN[14][I] = GasUtil.CALQINP(EN, N41D, Y41D, X41D, 1)
        if EN > (2 * object.EIN[14]):
            object.PEQIN[14][I] = object.PEQEL[1][I - IOFFN[14]]

        #4 3F
        if EN > object.EIN[15]:
            object.QIN[15][I] = GasUtil.CALQINP(EN, N43F, Y43F, X43F, 4)
        if EN > (2 * object.EIN[15]):
            object.PEQIN[15][I] = object.PEQEL[1][I - IOFFN[15]]

        #4 1F
        if EN > object.EIN[16]:
            object.QIN[16][I] = GasUtil.CALQINP(EN, N41F, Y41F, X41F, 1)
        if EN > (2 * object.EIN[16]):
            object.PEQIN[16][I] = object.PEQEL[1][I - IOFFN[16]]

        #4 1P
        if EN > object.EIN[17]:
            object.QIN[17][I] = GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2, object.DEN[I], BBCONST,
                                                  object.EIN[17], object.E[2], 0.02986)
        if EN > (2 * object.EIN[17]):
            object.PEQIN[17][I] = object.PEQEL[1][I - IOFFN[17]]

        #5 3S SCALED FROM 4 3S
        if EN > object.EIN[18]:
            ER = object.EIN[18] / object.EIN[10]
            ENP = EN / ER
            object.QIN[18][I] = 0.512 * GasUtil.CALQINP(ENP, N43S, Y43S, X43S, 3)
        if EN > (2 * object.EIN[18]):
            object.PEQIN[18][I] = object.PEQEL[1][I - IOFFN[18]]

        #5 1S SCALED FROM 4 1S
        if EN > object.EIN[19]:
            ER = object.EIN[19] / object.EIN[11]
            ENP = EN / ER
            object.QIN[19][I] = 0.512 * GasUtil.CALQINP(ENP, N41S, Y41S, X41S, 1)
        if EN > (2 * object.EIN[19]):
            object.PEQIN[19][I] = object.PEQEL[1][I - IOFFN[19]]

        #5 3P SCALED FROM 5 3P
        if EN > object.EIN[20]:
            ER = object.EIN[20] / object.EIN[12]
            ENP = EN / ER
            object.QIN[20][I] = 0.512 * GasUtil.CALQINP(ENP, N43P, Y43P, X43P, 3)
        if EN > (2 * object.EIN[20]):
            object.PEQIN[20][I] = object.PEQEL[1][I - IOFFN[20]]

        #5 3D SCALED FROM 4 3D
        if EN > object.EIN[21]:
            ER = object.EIN[21] / object.EIN[13]
            ENP = EN / ER
            object.QIN[21][I] = 0.512 * GasUtil.CALQINP(ENP, N43D, Y43D, X43D, 3)
        if EN > (2 * object.EIN[21]):
            object.PEQIN[21][I] = object.PEQEL[1][I - IOFFN[21]]

        #5 1D SCALED FROM 4 1D
        if EN > object.EIN[22]:
            ER = object.EIN[22] / object.EIN[14]
            ENP = EN / ER
            object.QIN[22][I] = 0.512 * GasUtil.CALQINP(ENP, N41D, Y41D, X41D, 1)
        if EN > (2 * object.EIN[22]):
            object.PEQIN[22][I] = object.PEQEL[1][I - IOFFN[22]]

        #5 3F SCALED FROM 4 3F
        if EN > object.EIN[23]:
            ER = object.EIN[23] / object.EIN[15]
            ENP = EN / ER
            object.QIN[23][I] = 0.512 * GasUtil.CALQINP(ENP, N43F, Y43F, X43F, 4)
        if EN > (2 * object.EIN[23]):
            object.PEQIN[23][I] = object.PEQEL[1][I - IOFFN[23]]

        #5 1F SCALED FROM 4 1F
        if EN > object.EIN[24]:
            ER = object.EIN[24] / object.EIN[16]
            ENP = EN / ER
            object.QIN[24][I] = 0.512 * GasUtil.CALQINP(ENP, N41F, Y41F, X41F, 1)
        if EN > (2 * object.EIN[24]):
            object.PEQIN[24][I] = object.PEQEL[1][I - IOFFN[24]]

        #5 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.01504
        if EN > object.EIN[25]:
            ER = object.EIN[25] / object.EIN[17]
            ENP = EN / ER
            object.QIN[25][I] = 0.01504 / 0.02986 * GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2,
                                                                      object.DEN[I], BBCONST,
                                                                      object.EIN[25], object.E[2], 0.02986)
        if EN > (2 * object.EIN[25]):
            object.PEQIN[25][I] = object.PEQEL[1][I - IOFFN[25]]

        #6 3S SCALED FROM 4 3S
        if EN > object.EIN[26]:
            ER = object.EIN[26] / object.EIN[10]
            ENP = EN / ER
            object.QIN[26][I] = 0.296 * GasUtil.CALQINP(ENP, N43S, Y43S, X43S, 3)
        if EN > (2 * object.EIN[26]):
            object.PEQIN[26][I] = object.PEQEL[1][I - IOFFN[26]]

        #6 1S SCALED FROM 4 1S
        if EN > object.EIN[27]:
            ER = object.EIN[27] / object.EIN[11]
            ENP = EN / ER
            object.QIN[27][I] = 0.296 * GasUtil.CALQINP(ENP, N41S, Y41S, X41S, 1)
        if EN > (2 * object.EIN[27]):
            object.PEQIN[27][I] = object.PEQEL[1][I - IOFFN[27]]

        #6 3P SCALED FROM 4 3P
        if EN > object.EIN[28]:
            ER = object.EIN[28] / object.EIN[12]
            ENP = EN / ER
            object.QIN[28][I] = 0.296 * GasUtil.CALQINP(ENP, N43P, Y43P, X43P, 3)
        if EN > (2 * object.EIN[28]):
            object.PEQIN[28][I] = object.PEQEL[1][I - IOFFN[28]]

        #6 3D SCALED FROM 4 3D
        if EN > object.EIN[29]:
            ER = object.EIN[29] / object.EIN[13]
            ENP = EN / ER
            object.QIN[29][I] = 0.296 * GasUtil.CALQINP(ENP, N43D, Y43D, X43D, 3)
        if EN > (2 * object.EIN[29]):
            object.PEQIN[29][I] = object.PEQEL[1][I - IOFFN[29]]

        #6 1D SCALED FROM 4 1D
        if EN > object.EIN[30]:
            ER = object.EIN[30] / object.EIN[14]
            ENP = EN / ER
            object.QIN[30][I] = 0.296 * GasUtil.CALQINP(ENP, N41D, Y41D, X41D, 1)
        if EN > (2 * object.EIN[30]):
            object.PEQIN[30][I] = object.PEQEL[1][I - IOFFN[30]]

        #6 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.01504
        if EN > object.EIN[31]:
            ER = object.EIN[31] / object.EIN[17]
            ENP = EN / ER
            object.QIN[31][I] = 0.00863 / 0.02986 * GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2,
                                                                      object.DEN[I], BBCONST,
                                                                      object.EIN[31], object.E[2], 0.02986)
        if EN > (2 * object.EIN[31]):
            object.PEQIN[31][I] = object.PEQEL[1][I - IOFFN[31]]

        #7 3S SCALED FROM 4 3S
        if EN > object.EIN[32]:
            ER = object.EIN[32] / object.EIN[10]
            ENP = EN / ER
            object.QIN[32][I] = 0.187 * GasUtil.CALQINP(ENP, N43S, Y43S, X43S, 3)
        if EN > (2 * object.EIN[32]):
            object.PEQIN[32][I] = object.PEQEL[1][I - IOFFN[32]]

        #7 1S SCALED FROM 4 1S
        if EN > object.EIN[33]:
            ER = object.EIN[33] / object.EIN[11]
            ENP = EN / ER
            object.QIN[33][I] = 0.187 * GasUtil.CALQINP(ENP, N41S, Y41S, X41S, 1)
        if EN > (2 * object.EIN[33]):
            object.PEQIN[33][I] = object.PEQEL[1][I - IOFFN[33]]

        #7 3P SCALED FROM 4 3P
        if EN > object.EIN[34]:
            ER = object.EIN[34] / object.EIN[12]
            ENP = EN / ER
            object.QIN[34][I] = 0.187 * GasUtil.CALQINP(ENP, N43P, Y43P, X43P, 3)
        if EN > (2 * object.EIN[34]):
            object.PEQIN[34][I] = object.PEQEL[1][I - IOFFN[34]]

        #7 3D SCALED FROM 4 3D
        if EN > object.EIN[35]:
            ER = object.EIN[35] / object.EIN[13]
            ENP = EN / ER
            object.QIN[35][I] = 0.187 * GasUtil.CALQINP(ENP, N43D, Y43D, X43D, 3)
        if EN > (2 * object.EIN[35]):
            object.PEQIN[35][I] = object.PEQEL[1][I - IOFFN[35]]

        #7 1D SCALED FROM 4 1D
        if EN > object.EIN[36]:
            ER = object.EIN[36] / object.EIN[14]
            ENP = EN / ER
            object.QIN[36][I] = 0.187 * GasUtil.CALQINP(ENP, N41D, Y41D, X41D, 1)
        if EN > (2 * object.EIN[36]):
            object.PEQIN[36][I] = object.PEQEL[1][I - IOFFN[36]]

        #7 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00540
        if EN > object.EIN[37]:
            ER = object.EIN[37] / object.EIN[17]
            ENP = EN / ER
            object.QIN[37][I] = 0.00540 / 0.02986 * GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2,
                                                                      object.DEN[I], BBCONST,
                                                                      object.EIN[37], object.E[2], 0.02986)
        if EN > (2 * object.EIN[37]):
            object.PEQIN[37][I] = object.PEQEL[1][I - IOFFN[37]]
        #SUM 3S LEVELS FROM 8 3S HIGHER AND SCALED FROM 4 3S
        if EN > object.EIN[38]:
            ER = object.EIN[38] / object.EIN[10]
            ENP = EN / ER
            object.QIN[38][I] = 0.553 * GasUtil.CALQINP(ENP, N43S, Y43S, X43S, 3)
        if EN > (2 * object.EIN[38]):
            object.PEQIN[38][I] = object.PEQEL[1][I - IOFFN[38]]

        #SUM 1S LEVELS FROM 8 3S HIGHER AND SCALED FROM 4 1S
        if EN > object.EIN[39]:
            ER = object.EIN[39] / object.EIN[11]
            ENP = EN / ER
            object.QIN[39][I] = 0.553 * GasUtil.CALQINP(ENP, N41S, Y41S, X41S, 1)
        if EN > (2 * object.EIN[39]):
            object.PEQIN[39][I] = object.PEQEL[1][I - IOFFN[39]]

        #SUM 3P LEVELS FROM  8 3P HIGHER AND SCALED FROM 4 3P
        if EN > object.EIN[40]:
            ER = object.EIN[40] / object.EIN[12]
            ENP = EN / ER
            object.QIN[40][I] = 0.553 * GasUtil.CALQINP(ENP, N43P, Y43P, X43P, 3)
        if EN > (2 * object.EIN[40]):
            object.PEQIN[40][I] = object.PEQEL[1][I - IOFFN[40]]

        #SUM 3D LEVELS FROM  8 3D HIGHER AND SCALED FROM 4 3D
        if EN > object.EIN[41]:
            ER = object.EIN[41] / object.EIN[13]
            ENP = EN / ER
            object.QIN[41][I] = 0.553 * GasUtil.CALQINP(ENP, N43D, Y43D, X43D, 3)
        if EN > (2 * object.EIN[41]):
            object.PEQIN[41][I] = object.PEQEL[1][I - IOFFN[41]]

        #SUM 1D LEVELS FROM  8 1D HIGHER AND SCALED FROM 4 1D
        if EN > object.EIN[42]:
            ER = object.EIN[42] / object.EIN[14]
            ENP = EN / ER
            object.QIN[42][I] = 0.553 * GasUtil.CALQINP(ENP, N41D, Y41D, X41D, 1)
        if EN > (2 * object.EIN[42]):
            object.PEQIN[42][I] = object.PEQEL[1][I - IOFFN[42]]

        #8 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00362
        if EN > object.EIN[43]:
            ER = object.EIN[43] / object.EIN[17]
            ENP = EN / ER
            object.QIN[43][I] = 0.00362 / 0.02986 * GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2,
                                                                      object.DEN[I], BBCONST,
                                                                      object.EIN[43], object.E[2], 0.02986)
        if EN > (2 * object.EIN[43]):
            object.PEQIN[43][I] = object.PEQEL[1][I - IOFFN[43]]

        #9 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00253
        if EN > object.EIN[44]:
            ER = object.EIN[44] / object.EIN[17]
            ENP = EN / ER
            object.QIN[44][I] = 0.00253 / 0.02986 * GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2,
                                                                      object.DEN[I], BBCONST,
                                                                      object.EIN[44], object.E[2], 0.02986)
        if EN > (2 * object.EIN[44]):
            object.PEQIN[44][I] = object.PEQEL[1][I - IOFFN[44]]

        #10 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00184
        if EN > object.EIN[45]:
            ER = object.EIN[45] / object.EIN[17]
            ENP = EN / ER
            object.QIN[45][I] = 0.00184 / 0.02986 * GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2,
                                                                      object.DEN[I], BBCONST,
                                                                      object.EIN[45], object.E[2], 0.02986)
        if EN > (2 * object.EIN[45]):
            object.PEQIN[45][I] = object.PEQEL[1][I - IOFFN[45]]

        #11 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00138
        if EN > object.EIN[46]:
            ER = object.EIN[46] / object.EIN[17]
            ENP = EN / ER
            object.QIN[46][I] = 0.00138 / 0.02986 * GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2,
                                                                      object.DEN[I], BBCONST,
                                                                      object.EIN[46], object.E[2], 0.02986)
        if EN > (2 * object.EIN[46]):
            object.PEQIN[46][I] = object.PEQEL[1][I - IOFFN[46]]

        #12 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00106
        if EN > object.EIN[47]:
            ER = object.EIN[47] / object.EIN[17]
            ENP = EN / ER
            object.QIN[47][I] = 0.00106 / 0.02986 * GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2,
                                                                      object.DEN[I], BBCONST,
                                                                      object.EIN[47], object.E[2], 0.02986)
        if EN > (2 * object.EIN[47]):
            object.PEQIN[47][I] = object.PEQEL[1][I - IOFFN[47]]

        #13 1P   SCALED FROM 4 1P   OSC STRENGTH  F=0.00440
        if EN > object.EIN[48]:
            ER = object.EIN[48] / object.EIN[17]
            ENP = EN / ER
            object.QIN[48][I] = 0.00440 / 0.02986 * GasUtil.CALQINBEF(EN, N41P, Y41P, X41P, BETA2, GAMMA2, EMASS2,
                                                                      object.DEN[I], BBCONST,
                                                                      object.EIN[48], object.E[2], 0.02986)
        if EN > (2 * object.EIN[48]):
            object.PEQIN[48][I] = object.PEQEL[1][I - IOFFN[48]]
        QMET = object.QIN[0][I] + object.QIN[1][I]
        QDIP = object.QIN[3][I] + object.QIN[9][I] + object.QIN[17][I] + object.QIN[25][I] + object.QIN[31][I] + \
               object.QIN[37][I] + object.QIN[43][I] + object.QIN[44][I] + object.QIN[45][I] + object.QIN[46][I] + \
               object.QIN[47][I] + object.QIN[48][I]
        QTRP = object.QIN[0][I] + object.QIN[2][I] + object.QIN[4][I] + object.QIN[6][I] + object.QIN[7][I] + \
               object.QIN[10][I] + object.QIN[12][I] + object.QIN[13][I] + object.QIN[15][I] + object.QIN[18][I] + \
               object.QIN[20][I] + object.QIN[21][I] + object.QIN[23][I] + object.QIN[26][I] + object.QIN[28][I] + \
               object.QIN[29][I] + object.QIN[32][I] + object.QIN[34][I] + object.QIN[35][I] + object.QIN[38][I] + \
               object.QIN[40][I] + object.QIN[41][I]
        QSNG = object.QIN[1][I] + object.QIN[3][I] + object.QIN[5][I] + object.QIN[8][I] + object.QIN[9][I] + \
               object.QIN[11][I] + object.QIN[14][I] + object.QIN[16][I] + object.QIN[17][I] + object.QIN[19][I] + \
               object.QIN[22][I] + object.QIN[24][I] + object.QIN[25][I] + object.QIN[27][I] + object.QIN[30][I] + \
               object.QIN[31][I] + object.QIN[33][I] + object.QIN[36][I] + object.QIN[37][I] + object.QIN[39][I] + \
               object.QIN[42][I] + object.QIN[43][I] + object.QIN[44][I] + object.QIN[45][I] + object.QIN[46][I] + \
               object.QIN[47][I] + object.QIN[48][I]
        QINEL = QSNG + QTRP + object.QION[0][I] + object.QION[1][I]
        object.Q[0][I] = QELA + QINEL

    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
