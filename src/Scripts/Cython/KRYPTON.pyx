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
cdef void Gas6(Gas*object):
    gd = np.load('gases.npy').item()
    cdef double XEN[162], YXSEC[162], XEL[151], YEL[151], XEPS[186], YEPS[186], XION[65], YION[65], YINC[65], YIN1[65], XIN2[41]
    cdef double YIN2[41], XIN3[35], YIN3[35], XIN4[32], YIN4[32], XKSH[74], YKSH[74], XL1S[83], YL1S[83], XL2S[82], YL2S[82], XL3S[84], YL3S[84]
    cdef double XM1S[91], YM1S[91], XM2S[98], YM2S[98], XM3S[99], YM3S[99], XM4S[105], YM4S[105], XM5S[106], YM5S[106]
    cdef double XEXC1[105], YEXC1[105], XEXC2[86], YEXC2[86], X2P2[75], Y2P2[75], YP2P2[75], X3D4[59], Y3D4[59], YP3D4[59]
    cdef double X1S5[169], Y1S5[169], YP1S5[169], X1S4[130], Y1S4[130], YP1S4[130], X1S3[168], Y1S3[168], YP1S3[168], X1S2[150], Y1S2[150], YP1S2[150]
    cdef double X2P10[142], Y2P10[142], YP2P10[142], X2P9[117], Y2P9[117], YP2P9[117], X3D6[69], Y3D6[69], YP3D6[69], X3D5[75], Y3D5[75], YP3D5[75]
    cdef double X2P8[120], Y2P8[120], YP2P8[120], X2P7[111], Y2P7[111], YP2P7[111], X2P6[100], Y2P6[100], YP2P6[100], X2P5[102], Y2P5[102], YP2P5[102]
    cdef double X2P4[64], Y2P4[64], YP2P4[64], X3D3[74], Y3D3[74], YP3D3[74], X3D4P[73], Y3D4P[73], YP3D4P[73], X2P3[73], Y2P3[73], YP2P3[73]
    cdef double X2P1[51], Y2P1[51], YP2P1[51], X3D1PP[48], Y3D1PP[48], YP3D1PP[48], X3D1P[41], Y3D1P[41], YP3D1P[41], X2S5[44], Y2S5[44], YP2S5[44]
    cdef double X3P10[20], Y3P10[20], YP3P10[20], X3P9[20], Y3P9[20], YP3P9[20], X3P8[20], Y3P8[20], YP3P8[20], X3S1PP[20], Y3S1PP[20], YP3S1PP[20]
    cdef double X3P7[20], Y3P7[20], YP3P7[20], X3P6[20], Y3P6[20], YP3P6[20], X4D1P[19], Y4D1P[19], YP4D1P[19], X3S5[19], Y3S5[19], YP3S5[19]
    cdef double X3S1PPPP[20], Y3S1PPPP[20], YP3S1PPPP[20], X3S1PPP[20], Y3S1PPP[20], YP3S1PPP[20], X3P5[20], Y3P5[20], YP3P5[20]
    cdef double X4D6[20], Y4D6[20], YP4D6[20], X4D4P[20], Y4D4P[20], YP4D4P[20], X2S3[19], Y2S3[19], YP2S3[19], X4D1PP[19], Y4D1PP[19], YP4D1PP[19]
    cdef double X4D4[19], Y4D4[19], YP4D4[19], X4D3[19], Y4D3[19], YP4D3[19], X4FS[19], Y4FS[19], YP4FS[19], Z36T[25], EBRM[25]
    cdef int IOFFN[51], IOFFION[11]

    XEN = gd['gas6/XEN']
    YXSEC = gd['gas6/YXSEC']
    XEL = gd['gas6/XEL']
    YEL = gd['gas6/YEL']
    XEPS = gd['gas6/XEPS']
    YEPS = gd['gas6/YEPS']
    XION = gd['gas6/XION']
    YION = gd['gas6/YION']
    YINC = gd['gas6/YINC']
    YIN1 = gd['gas6/YIN1']
    XIN2 = gd['gas6/XIN2']
    YIN2 = gd['gas6/YIN2']
    XIN3 = gd['gas6/XIN3']
    YIN3 = gd['gas6/YIN3']
    XIN4 = gd['gas6/XIN4']
    YIN4 = gd['gas6/YIN4']
    XKSH = gd['gas6/XKSH']
    YKSH = gd['gas6/YKSH']
    XL1S = gd['gas6/XL1S']
    YL1S = gd['gas6/YL1S']
    XL2S = gd['gas6/XL2S']
    YL2S = gd['gas6/YL2S']
    XL3S = gd['gas6/XL3S']
    YL3S = gd['gas6/YL3S']
    XM1S = gd['gas6/XM1S']
    YM1S = gd['gas6/YM1S']
    XM2S = gd['gas6/XM2S']
    YM2S = gd['gas6/YM2S']
    XM3S = gd['gas6/XM3S']
    YM3S = gd['gas6/YM3S']
    XM4S = gd['gas6/XM4S']
    YM4S = gd['gas6/YM4S']
    XM5S = gd['gas6/XM5S']
    YM5S = gd['gas6/YM5S']
    XEXC1 = gd['gas6/XEXC1']
    YEXC1 = gd['gas6/YEXC1']
    XEXC2 = gd['gas6/XEXC2']
    YEXC2 = gd['gas6/YEXC2']
    X1S5 = gd['gas6/X1S5']
    Y1S5 = gd['gas6/Y1S5']
    YP1S5 = gd['gas6/YP1S5']
    X1S4 = gd['gas6/X1S4']
    Y1S4 = gd['gas6/Y1S4']
    YP1S4 = gd['gas6/YP1S4']
    X1S3 = gd['gas6/X1S3']
    Y1S3 = gd['gas6/Y1S3']
    YP1S3 = gd['gas6/YP1S3']
    X1S2 = gd['gas6/X1S2']
    Y1S2 = gd['gas6/Y1S2']
    YP1S2 = gd['gas6/YP1S2']
    X2P10 = gd['gas6/X2P10']
    Y2P10 = gd['gas6/Y2P10']
    YP2P10 = gd['gas6/YP2P10']
    X2P9 = gd['gas6/X2P9']
    Y2P9 = gd['gas6/Y2P9']
    YP2P9 = gd['gas6/YP2P9']
    X2P8 = gd['gas6/X2P8']
    Y2P8 = gd['gas6/Y2P8']
    YP2P8 = gd['gas6/YP2P8']
    X2P7 = gd['gas6/X2P7']
    Y2P7 = gd['gas6/Y2P7']
    YP2P7 = gd['gas6/YP2P7']
    X2P6 = gd['gas6/X2P6']
    Y2P6 = gd['gas6/Y2P6']
    YP2P6 = gd['gas6/YP2P6']
    X2P5 = gd['gas6/X2P5']
    Y2P5 = gd['gas6/Y2P5']
    YP2P5 = gd['gas6/YP2P5']
    X3D6 = gd['gas6/X3D6']
    Y3D6 = gd['gas6/Y3D6']
    YP3D6 = gd['gas6/YP3D6']
    X3D5 = gd['gas6/X3D5']
    Y3D5 = gd['gas6/Y3D5']
    YP3D5 = gd['gas6/YP3D5']
    X2P4 = gd['gas6/X2P4']
    Y2P4 = gd['gas6/Y2P4']
    YP2P4 = gd['gas6/YP2P4']
    X3D3 = gd['gas6/X3D3']
    Y3D3 = gd['gas6/Y3D3']
    YP3D3 = gd['gas6/YP3D3']
    X3D4P = gd['gas6/X3D4P']
    Y3D4P = gd['gas6/Y3D4P']
    YP3D4P = gd['gas6/YP3D4P']
    X2P3 = gd['gas6/X2P3']
    Y2P3 = gd['gas6/Y2P3']
    YP2P3 = gd['gas6/YP2P3']
    X2P2 = gd['gas6/X2P2']
    Y2P2 = gd['gas6/Y2P2']
    YP2P2 = gd['gas6/YP2P2']
    X3D4 = gd['gas6/X3D4']
    Y3D4 = gd['gas6/Y3D4']
    YP3D4 = gd['gas6/YP3D4']
    X2P1 = gd['gas6/X2P1']
    Y2P1 = gd['gas6/Y2P1']
    YP2P1 = gd['gas6/YP2P1']
    X3D1PP = gd['gas6/X3D1PP']
    Y3D1PP = gd['gas6/Y3D1PP']
    YP3D1PP = gd['gas6/YP3D1PP']
    X3D1P = gd['gas6/X3D1P']
    Y3D1P = gd['gas6/Y3D1P']
    YP3D1P = gd['gas6/YP3D1P']
    X2S5 = gd['gas6/X2S5']
    Y2S5 = gd['gas6/Y2S5']
    YP2S5 = gd['gas6/YP2S5']
    X3P10 = gd['gas6/X3P10']
    Y3P10 = gd['gas6/Y3P10']
    YP3P10 = gd['gas6/YP3P10']
    X3P9 = gd['gas6/X3P9']
    Y3P9 = gd['gas6/Y3P9']
    YP3P9 = gd['gas6/YP3P9']
    X3P8 = gd['gas6/X3P8']
    Y3P8 = gd['gas6/Y3P8']
    YP3P8 = gd['gas6/YP3P8']
    X3S1PP = gd['gas6/X3S1PP']
    Y3S1PP = gd['gas6/Y3S1PP']
    YP3S1PP = gd['gas6/YP3S1PP']
    X3P7 = gd['gas6/X3P7']
    Y3P7 = gd['gas6/Y3P7']
    YP3P7 = gd['gas6/YP3P7']
    X3P6 = gd['gas6/X3P6']
    Y3P6 = gd['gas6/Y3P6']
    YP3P6 = gd['gas6/YP3P6']
    X3S1PPPP = gd['gas6/X3S1PPPP']
    Y3S1PPPP = gd['gas6/Y3S1PPPP']
    YP3S1PPPP = gd['gas6/YP3S1PPPP']
    X3S1PPP = gd['gas6/X3S1PPP']
    Y3S1PPP = gd['gas6/Y3S1PPP']
    YP3S1PPP = gd['gas6/YP3S1PPP']
    X3P5 = gd['gas6/X3P5']
    Y3P5 = gd['gas6/Y3P5']
    YP3P5 = gd['gas6/YP3P5']
    X4D6 = gd['gas6/X4D6']
    Y4D6 = gd['gas6/Y4D6']
    YP4D6 = gd['gas6/YP4D6']
    X4D4P = gd['gas6/X4D4P']
    Y4D4P = gd['gas6/Y4D4P']
    YP4D4P = gd['gas6/YP4D4P']
    X4D4 = gd['gas6/X4D4']
    Y4D4 = gd['gas6/Y4D4']
    YP4D4 = gd['gas6/YP4D4']
    X4D3 = gd['gas6/X4D3']
    Y4D3 = gd['gas6/Y4D3']
    YP4D3 = gd['gas6/YP4D3']
    X2S3 = gd['gas6/X2S3']
    Y2S3 = gd['gas6/Y2S3']
    YP2S3 = gd['gas6/YP2S3']
    X4D1PP = gd['gas6/X4D1PP']
    Y4D1PP = gd['gas6/Y4D1PP']
    YP4D1PP = gd['gas6/YP4D1PP']
    X4D1P = gd['gas6/X4D1P']
    Y4D1P = gd['gas6/Y4D1P']
    YP4D1P = gd['gas6/YP4D1P']
    X3S5 = gd['gas6/X3S5']
    Y3S5 = gd['gas6/Y3S5']
    YP3S5 = gd['gas6/YP3S5']
    X4FS = gd['gas6/X4FS']
    Y4FS = gd['gas6/Y4FS']
    YP4FS = gd['gas6/YP4FS']
    Z36T = gd['gas6/Z36T']
    EBRM = gd['gas6/EBRM']

    cdef double CONST, EMASS2, AM2, C, A0, RY, API, BBCONST, AN1S, AN2P10, AN2P5, AN2P1, AN2P, AN3P, AN3P5, AN3D, AN4D, AUGM5, AUGM4, AUGM3
    cdef double AUGM2, AUGM1, AUGL3, AUGL2, AUGL1, AUGK
    cdef int NBREM, i, j, I, J, NL, NEL, NDATA, NEPSI, NIONG, NION2, NION3, NION4, NKSH, NL1S, NL2S, NL3S, NM1S, NM2S, NM3S, NM4S, NM5S, N1S5
    cdef int N1S4, N1S3, N1S2, N2P10, N2P9, N2P8, N2P7, N2P6, N2P5, N3D6, N3D5, N2P4, N3D3, N3D4P, N2P3, N2P2, N3D4, N2P1, N3D1PP, N3D1P
    cdef int N2S5, N3P10, N3P9, N3P8, N3S1PP, N3P7, N3P6, N3S1PPPP, N3S1PPP, N3P5, N4D6, N4D4P, N4D4, N4D3, N2S3, N4D1PP, N4D1P, N3S5, N4FS
    CONST = 1.873884e-20
    EMASS2 = 1021997.804
    AM2 = 4.65
    C = 52.7
    # BORN BETHE CONSTANT
    A0 = 0.52917720859e-8
    RY = 13.60569193
    API = acos(-1.0)
    BBCONST = 16.0 * API * A0 * A0 * RY * RY / EMASS2
    # SCALING CONSTANTS
    AN1S = 0.87
    AN2P10 = 0.4
    AN2P5 = 0.4
    AN2P1 = 0.4
    AN2P = 0.75
    AN3P = 0.60
    AN3P5 = 0.4
    AN3D = 0.65
    AN4D = 0.4
    # AVERAGE AUGER EMISSIONS FROM EACH SHELL
    AUGM5 = 2.0
    AUGM4 = 2.0
    AUGM3 = 3.43
    AUGM2 = 2.0
    AUGM1 = 3.81
    AUGL3 = 4.85
    AUGL2 = 4.41
    AUGL1 = 6.47
    AUGK = 5.91

    object.NION = 11
    object.NATT = 1
    object.NIN = 51
    object.NNULL = 0

    for i in range(6):
        object.KEL[i] = object.NANISO
    for i in range(object.NIN):
        object.KIN[i] = object.NANISO
    NEL = 151
    NDATA = 162
    NEPSI = 186
    NIONG = 65
    NION2 = 41
    NION3 = 35
    NION4 = 32
    NKSH = 74
    NL1S = 83
    NL2S = 82
    NL3S = 84
    NM1S = 91
    NM2S = 98
    NM3S = 99
    NM4S = 105
    NM5S = 106
    N1S5 = 169
    N1S4 = 130
    N1S3 = 168
    N1S2 = 150
    N2P10 = 142
    N2P9 = 117
    N2P8 = 120
    N2P7 = 111
    N2P6 = 100
    N2P5 = 102
    N3D6 = 69
    N3D5 = 75
    N2P4 = 64
    N3D3 = 74
    N3D4P = 73
    N2P3 = 73
    N2P2 = 75
    N3D4 = 59
    N2P1 = 51
    N3D1PP = 48
    N3D1P = 41
    N2S5 = 44
    N3P10 = 20
    N3P9 = 20
    N3P8 = 20
    N3S1PP = 20
    N3P7 = 20
    N3P6 = 20
    N3S1PPPP = 20
    N3S1PPP = 20
    N3P5 = 20
    N4D6 = 20
    N4D4P = 20
    N4D4 = 19
    N4D3 = 19
    N2S3 = 19
    N4D1PP = 19
    N4D1P = 19
    N3S5 = 19
    N4FS = 19

    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, EOBY[11]

    object.E = [0.0, 1.0, 13.9996, 0.0, 0.0, 23.0]
    object.E[1] = 2.0 * EMASS / (83.798 * AMU)
    # EOBY at low energy 0-3, EOBY for shells 4-10
    EOBY[0:11] = [10.0, 30.0, 60.0, 100., 175.0, 180.0, 250.0, 1678.4, 1730.9, 1921.0, 14327.26]

    # AUGER AND FLUORESCENCE DATA
    object.NC0[0:11] = [0, 1, 2, 3, 3, 2, 4, 5, 4, 7, 10]
    object.EC0[0:11] = [0.0, 5.0, 10.0, 15.0, 135.1, 186.8, 200.9, 1555, 1619.9, 1698.4, 13993]
    object.WK[0:11] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0216, 0.0211, 0.0022, 0.65]
    object.EFL[0:11] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1584, 1636, 1700, 12649]
    object.NG1[0:11] = [0, 0, 0, 0, 0, 0, 0, 3, 3, 5, 5]
    object.EG1[0:11] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1490, 1580, 1538, 12438]
    object.NG2[0:11] = [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 5]
    object.EG2[0:11] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 54, 60, 160, 1555]
    object.EION[0:11] = [13.99960, 38.35944, 74.029, 124.88, 214.4, 222.2, 292.8, 1678.4, 1730.9, 1921.0, 14327.26]

    for j in range(0, object.NION):
        for i in range(0, 4000):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i - 1
                break
    object.EIN = gd['gas6/EIN']
    # ENTER PENNING TRANSFER FRACTION FOR EACH LEVEL
    # USE PENNING TRANSFER FRACTION BETWEEN 0.0 AND 0.2
    for NL in range(object.NIN):
        object.PENFRA[0][NL] = 0.0
        #PENNING TRANSFER DISTANCE MICRONS
        object.PENFRA[1][NL] = 1.0
        #PENNING TRANSFER TIME
        object.PENFRA[3][NL] = 1.0

    for NL in range(object.NIN):
        for i in range(4000):
            if object.EG[i] > object.EIN[NL]:
                IOFFN[NL] = i -1
                break
    cdef double F[11]
    cdef double FI

    F = [0.082,0.154,0.0140,0.0435,0.0105,0.0970,0.0808,0.0015,0.0439,0.0203,0.1680]

    cdef double EN,GAMMA1,GAMMA2,BETA,BETA2,QELA,QMOM,PQ[3],QTEMP,QCORR
    object.NSTEP = 4000
    for I in range(object.NSTEP):
        EN = object.EG[I]
        if EN > object.EIN[0]:
            GAMMA1 = (EMASS2 + 2 * EN) / EMASS2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = sqrt(1.0 - 1.0 / GAMMA2)
            BETA2 = BETA * BETA
        if EN == 0:
            QELA = 37.8e-16
            QMOM = 37.8e-16
        else:
            QELA = GasUtil.QLSCALE(EN, NEL, YEL, XEL)
            QMOM = GasUtil.QLSCALE(EN, NDATA, YXSEC, XEN)
        PQ=[0,0.5+(QELA-QMOM)/QELA,1-GasUtil.CALPQ3(EN,NEPSI, YEPS, XEPS)]

        object.PEQEL[1][I] = PQ[object.NANISO]
        object.Q[1][I] = QELA
        if object.NANISO==0:
            object.Q[1][I] = QMOM


        #IONISATION FOR CHARGE STATE =1
        object.QION[0][I] = 0.0
        object.PEQION[0][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[0][I] = 0.0
        if EN > object.EION[0]:
            object.QION[0][I] = GasUtil.CALQIONX(EN, NIONG, YIN1, XION, BETA2, 0.9009, CONST, object.DEN[I], C, AM2)
        #USE ANISOTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
        #ENERGIES ABOVE 2 * IONISATION ENERGY
        # ANISOTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFF SET BY
        # THE IONISATION ENERGY
        if EN > 2 * object.EION[0]:
            object.PEQION[0][I] = object.PEQEL[1][I - IOFFION[0]]
            
        #IONISATION FOR CHARGE STATE =2
        object.QION[1][I] = 0.0
        object.PEQION[1][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[1][I] = 0.0
        if EN > object.EION[1]:
            object.QION[1][I] = GasUtil.CALQIONX(EN, NION2, YIN2, XIN2, BETA2, 0.0613, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[1]:
            object.PEQION[1][I] = object.PEQEL[1][I - IOFFION[1]]

        #IONISATION FOR CHARGE STATE =3
        object.QION[2][I] = 0.0
        object.PEQION[2][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[2][I] = 0.0
        if EN > object.EION[2]:
            object.QION[2][I] = GasUtil.CALQIONX(EN, NION3, YIN3, XIN3, BETA2, 0.0291, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[2]:
            object.PEQION[2][I] = object.PEQEL[1][I - IOFFION[2]]
            
        #IONISATION FOR CHARGE STATE =4
        object.QION[3][I] = 0.0
        object.PEQION[3][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[3][I] = 0.0
        if EN > object.EION[3]:
            object.QION[3][I] = GasUtil.CALQIONX(EN, NION4, YIN4, XIN4, BETA2, 0.0082, CONST, object.DEN[I], C, AM2)
        if EN > 2 * object.EION[3]:
            object.PEQION[3][I] = object.PEQEL[1][I - IOFFION[3]]

        # M3 SHELL IONISATION
        object.QION[4][I] = 0.0
        object.PEQION[4][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[4][I] = 0.0
        if EN > object.EION[4]:
            object.QION[4][I] = GasUtil.CALQION(EN, NM3S, YM3S, XM3S)
            object.PEQION[4][I] = object.PEQEL[1][I - IOFFION[4]]

        # M2 SHELL IONISATION
        object.QION[5][I] = 0.0
        object.PEQION[5][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[5][I] = 0.0
        if EN > object.EION[5]:
            object.QION[5][I] = GasUtil.CALQION(EN, NM2S, YM2S, XM2S)
            object.PEQION[5][I] = object.PEQEL[1][I - IOFFION[5]]


        # M1 SHELL IONISATION
        object.QION[6][I] = 0.0
        object.PEQION[6][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[6][I] = 0.0
        if EN > object.EION[6]:
            object.QION[6][I] = GasUtil.CALQION(EN, NM1S, YM1S, XM1S)
            object.PEQION[6][I] = object.PEQEL[1][I - IOFFION[6]]

        # L3 SHELL IONISATION
        object.QION[7][I] = 0.0
        object.PEQION[7][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[7][I] = 0.0
        if EN > object.EION[7]:
            object.QION[7][I] = GasUtil.CALQION(EN, NL3S, YL3S, XL3S)
            object.PEQION[7][I] = object.PEQEL[1][I - IOFFION[7]]

        # L2 SHELL IONISATION
        object.QION[8][I] = 0.0
        object.PEQION[8][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[8][I] = 0.0
        if EN > object.EION[8]:
            object.QION[8][I] = GasUtil.CALQION(EN, NL2S, YL2S, XL2S)
            object.PEQION[8][I] = object.PEQEL[1][I - IOFFION[8]]

        # L1 SHELL IONISATION
        object.QION[9][I] = 0.0
        object.PEQION[9][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[9][I] = 0.0
        if EN > object.EION[9]:
            object.QION[9][I] = GasUtil.CALQION(EN, NL1S, YL1S, XL1S)
            object.PEQION[9][I] = object.PEQEL[1][I - IOFFION[9]]

        # K-SHELL IONISATION
        object.QION[10][I] = 0.0
        object.PEQION[10][I] = 0.5
        if object.NANISO == 2:
            object.PEQION[10][I] = 0.0
        if EN > object.EION[10]:
            object.QION[10][I] = GasUtil.CALQION(EN, NKSH, YKSH, XKSH)
            object.PEQION[10][I] = object.PEQEL[1][I - IOFFION[10]]
        object.Q[3][I] = 0.0
        object.QATT[0][I] =0.0
        
        #COUNTING IONISATION
        object.Q[4][I] = 0.0
        object.PEQEL[4][I] = 0.5
        if object.NANISO == 2:
            object.PEQEL[4][I] = 0.0
        if EN > object.E[2]:
            object.Q[4][I] = GasUtil.CALQIONX(EN, NIONG, YINC, XION, BETA2, 1, CONST, object.DEN[I], C, AM2)
        QTEMP = 0.0
        for i in range(4,11):
            QTEMP +=object.QION[i][I]

        if object.Q[4][I] ==0:
            QCORR = 1.0
        else:
            QCORR = (object.Q[4][I]-QTEMP)/object.Q[4][I]

        object.QION[0][I] *=QCORR
        object.QION[1][I] *=QCORR
        object.QION[2][I] *=QCORR
        object.QION[3][I] *=QCORR

        object.Q[5][I] = 0.0

        for NL in range(object.NIN+1):
            object.QIN[NL][I] = 0.0
            object.PEQIN[NL][I] = 0.5
            if object.NANISO==2:
                object.PEQIN[NL][I] = 0.0

        #1S5
        if EN > object.EIN[0]:
            object.QIN[0][I] = GasUtil.CALQINP(EN, N1S5, Y1S5, X1S5, 3)*AN1S
        if EN > 2 * object.EIN[0]:
            object.PEQIN[0][I] = object.PEQEL[1][I - IOFFN[0]]

        #1S4  F=0.203
        if EN > object.EIN[1]:
            object.QIN[1][I] = GasUtil.CALQINBEF(EN, N1S4, Y1S4, X1S4, BETA2, GAMMA2, EMASS2, object.DEN[I], BBCONST,
                                                 object.EIN[1], object.E[2], 0.203)
            if EN<=X1S4[N1S4-1]:
                object.QIN[1][I]*=AN1S
        if EN > 2 * object.EIN[1]:
            object.PEQIN[1][I] = object.PEQEL[1][I - IOFFN[1]]

        #1S3
        if EN > object.EIN[2]:
            object.QIN[2][I] = GasUtil.CALQINP(EN, N1S3, Y1S3, X1S3, 3)*AN1S
        if EN > 2 * object.EIN[2]:
            object.PEQIN[2][I] = object.PEQEL[1][I - IOFFN[2]]

        #1S2  F=0.182
        if EN > object.EIN[3]:
            object.QIN[3][I] = GasUtil.CALQINBEF(EN, N1S2, Y1S2, X1S2, BETA2, GAMMA2, EMASS2, object.DEN[I], BBCONST,
                                                 object.EIN[1], object.E[2], 0.182)
            if EN<=X1S2[N1S2-1]:
                object.QIN[3][I]*=AN1S
        if EN > 2 * object.EIN[3]:
            object.PEQIN[3][I] = object.PEQEL[1][I - IOFFN[3]]

        #2P10
        if EN > object.EIN[4]:
            object.QIN[4][I] = GasUtil.CALQINP(EN, N2P10, Y2P10, X2P10, 3)*AN2P10
        if EN > 2 * object.EIN[4]:
            object.PEQIN[4][I] = object.PEQEL[1][I - IOFFN[4]]

        #2P9
        if EN > object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQINP(EN, N2P9, Y2P9, X2P9, 3)*AN2P
        if EN > 2 * object.EIN[5]:
            object.PEQIN[5][I] = object.PEQEL[1][I - IOFFN[5]]

        #2P8
        if EN > object.EIN[6]:
            object.QIN[6][I] = GasUtil.CALQINP(EN, N2P8, Y2P8, X2P8, 1)*AN2P
        if EN > 2 * object.EIN[6]:
            object.PEQIN[6][I] = object.PEQEL[1][I - IOFFN[6]]

        #2P7
        if EN > object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQINP(EN, N2P7, Y2P7, X2P7, 3)*AN2P
        if EN > 2 * object.EIN[7]:
            object.PEQIN[7][I] = object.PEQEL[1][I - IOFFN[7]]

        #2P6
        if EN > object.EIN[8]:
            object.QIN[8][I] = GasUtil.CALQINP(EN, N2P6, Y2P6, X2P6, 1)*AN2P
        if EN > 2 * object.EIN[8]:
            object.PEQIN[8][I] = object.PEQEL[1][I - IOFFN[8]]

        #2P5
        if EN > object.EIN[9]:
            object.QIN[9][I] = GasUtil.CALQINP(EN, N2P5, Y2P5, X2P5, 1)*AN2P5
        if EN > 2 * object.EIN[9]:
            object.PEQIN[9][I] = object.PEQEL[1][I - IOFFN[9]]

        #3D6
        if EN > object.EIN[10]:
            object.QIN[10][I] = GasUtil.CALQINP(EN, N3D6, Y3D6, X3D6, 3)*AN3D
        if EN > 2 * object.EIN[10]:
            object.PEQIN[10][I] = object.PEQEL[1][I - IOFFN[10]]

        #3D5  BEF SCALING
        if EN > object.EIN[11]:
            object.QIN[11][I] = 0.0053 / (object.EIN[11] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[11])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[11] + object.E[2])
            if object.QIN[11][I]<0.0:
                object.QIN[11][I]=0.0
        if EN > 2 * object.EIN[11]:
            object.PEQIN[11][I] = object.PEQEL[1][I - IOFFN[11]]

        #2P4
        if EN > object.EIN[12]:
            object.QIN[12][I] = GasUtil.CALQINP(EN, N2P4, Y2P4, X2P4, 3)*AN2P
        if EN > 2 * object.EIN[12]:
            object.PEQIN[12][I] = object.PEQEL[1][I - IOFFN[12]]
            
        #3D3
        if EN > object.EIN[13]:
            object.QIN[13][I] = GasUtil.CALQINP(EN, N3D3, Y3D3, X3D3, 3)*AN3D
        if EN > 2 * object.EIN[13]:
            object.PEQIN[13][I] = object.PEQEL[1][I - IOFFN[13]]
        
        #3D4P
        if EN > object.EIN[14]:
            object.QIN[14][I] = GasUtil.CALQINP(EN, N3D4P, Y3D4P, X3D4P, 3)*AN3D
        if EN > 2 * object.EIN[14]:
            object.PEQIN[14][I] = object.PEQEL[1][I - IOFFN[14]]

        #2P3
        if EN > object.EIN[15]:
            object.QIN[15][I] = GasUtil.CALQINP(EN, N2P3, Y2P3, X2P3, 3)*AN2P
        if EN > 2 * object.EIN[15]:
            object.PEQIN[15][I] = object.PEQEL[1][I - IOFFN[15]]

        #2P2
        if EN > object.EIN[16]:
            object.QIN[16][I] = GasUtil.CALQINP(EN, N2P2, Y2P2, X2P2, 1)*AN2P
        if EN > 2 * object.EIN[16]:
            object.PEQIN[16][I] = object.PEQEL[1][I - IOFFN[16]]
            
        #3D4
        if EN > object.EIN[17]:
            object.QIN[17][I] = GasUtil.CALQINP(EN, N3D4, Y3D4, X3D4, 1)*AN3D
        if EN > 2 * object.EIN[17]:
            object.PEQIN[17][I] = object.PEQEL[1][I - IOFFN[17]]

        #2P1
        if EN > object.EIN[18]:
            object.QIN[18][I] = GasUtil.CALQINP(EN, N2P1, Y2P1, X2P1, 1)*AN2P1
        if EN > 2 * object.EIN[18]:
            object.PEQIN[18][I] = object.PEQEL[1][I - IOFFN[18]]
        
        #3D41PP
        if EN > object.EIN[19]:
            object.QIN[19][I] = GasUtil.CALQINP(EN, N3D1PP, Y3D1PP, X3D1PP, 3)*AN3D
        if EN > 2 * object.EIN[19]:
            object.PEQIN[19][I] = object.PEQEL[1][I - IOFFN[19]]
            
        #3D41P
        if EN > object.EIN[20]:
            object.QIN[20][I] = GasUtil.CALQINP(EN, N3D1P, Y3D1P, X3D1P, 1)*AN3D
        if EN > 2 * object.EIN[20]:
            object.PEQIN[20][I] = object.PEQEL[1][I - IOFFN[2]]

        #2S5
        if EN > object.EIN[21]:
            object.QIN[21][I] = GasUtil.CALQINP(EN, N2S5, Y2S5, X2S5, 3)*AN1S
        if EN > 2 * object.EIN[21]:
            object.PEQIN[21][I] = object.PEQEL[1][I - IOFFN[21]]
            
            
        FI = 0
        for J in range(22,24):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                            log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                        I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI+=1


        #3P10
        if EN > object.EIN[24]:
            object.QIN[24][I] = GasUtil.CALQINP(EN, N3P10, Y3P10, X3P10, 3)*AN3P
        if EN > 2 * object.EIN[24]:
            object.PEQIN[24][I] = object.PEQEL[1][I - IOFFN[24]]
        
        
        #3P9
        if EN > object.EIN[25]:
            object.QIN[25][I] = GasUtil.CALQINP(EN, N3P9, Y3P9, X3P9, 3)*AN3P
        if EN > 2 * object.EIN[25]:
            object.PEQIN[25][I] = object.PEQEL[1][I - IOFFN[25]]
            
        #3P8
        if EN > object.EIN[26]:
            object.QIN[26][I] = GasUtil.CALQINP(EN, N3P8, Y3P8, X3P8, 1)*AN3P
        if EN > 2 * object.EIN[26]:
            object.PEQIN[26][I] = object.PEQEL[1][I - IOFFN[26]]

        #3S1PP
        if EN > object.EIN[27]:
            object.QIN[27][I] = GasUtil.CALQINP(EN, N3S1PP, Y3S1PP, X3S1PP, 3)*AN3D
        if EN > 2 * object.EIN[27]:
            object.PEQIN[27][I] = object.PEQEL[1][I - IOFFN[27]]

        #3P7
        if EN > object.EIN[28]:
            object.QIN[28][I] = GasUtil.CALQINP(EN, N3P7, Y3P7, X3P7, 3)*AN3P
        if EN > 2 * object.EIN[28]:
            object.PEQIN[28][I] = object.PEQEL[1][I - IOFFN[28]]

        #3P6
        if EN > object.EIN[29]:
            object.QIN[29][I] = GasUtil.CALQINP(EN, N3P6, Y3P6, X3P6, 1)*AN3P
        if EN > 2 * object.EIN[29]:
            object.PEQIN[29][I] = object.PEQEL[1][I - IOFFN[29]]

        #3S1PPPP
        if EN > object.EIN[30]:
            object.QIN[30][I] = GasUtil.CALQINP(EN, N3S1PPPP, Y3S1PPPP, X3S1PPPP, 3)*AN3D
        if EN > 2 * object.EIN[30]:
            object.PEQIN[30][I] = object.PEQEL[1][I - IOFFN[30]]

        #3S1PPP
        if EN > object.EIN[31]:
            object.QIN[31][I] = GasUtil.CALQINP(EN, N3S1PPP, Y3S1PPP, X3S1PPP, 1)*AN3D
        if EN > 2 * object.EIN[31]:
            object.PEQIN[31][I] = object.PEQEL[1][I - IOFFN[3]]

        #3P5
        if EN > object.EIN[32]:
            object.QIN[32][I] = GasUtil.CALQINP(EN, N3P5, Y3P5, X3P5, 1)*AN3P5
        if EN > 2 * object.EIN[32]:
            object.PEQIN[32][I] = object.PEQEL[1][I - IOFFN[32]]
        
        J=33
        if EN > object.EIN[J]:
            object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
        if EN > 2 * object.EIN[J]:
            object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
        FI+=1

        #4D6
        if EN > object.EIN[34]:
            object.QIN[34][I] = GasUtil.CALQINP(EN, N4D6, Y4D6, X4D6, 3)*AN4D
        if EN > 2 * object.EIN[34]:
            object.PEQIN[34][I] = object.PEQEL[1][I - IOFFN[34]]

        #4D4P
        if EN > object.EIN[35]:
            object.QIN[35][I] = GasUtil.CALQINP(EN, N4D4P, Y4D4P, X4D4P, 3)*AN4D
        if EN > 2 * object.EIN[35]:
            object.PEQIN[35][I] = object.PEQEL[1][I - IOFFN[35]]

        J=36
        if EN > object.EIN[J]:
            object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
        if EN > 2 * object.EIN[J]:
            object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
        FI+=1
        
        #4D4
        if EN > object.EIN[37]:
            object.QIN[37][I] = GasUtil.CALQINP(EN, N4D4, Y4D4, X4D4, 1)*AN4D
        if EN > 2 * object.EIN[37]:
            object.PEQIN[37][I] = object.PEQEL[1][I - IOFFN[37]]

        #4D3
        if EN > object.EIN[38]:
            object.QIN[38][I] = GasUtil.CALQINP(EN, N4D3, Y4D3, X4D3, 3)*AN4D
        if EN > 2 * object.EIN[38]:
            object.PEQIN[38][I] = object.PEQEL[1][I - IOFFN[38]]

        #2S3
        if EN > object.EIN[39]:
            object.QIN[39][I] = GasUtil.CALQINP(EN, N2S3, Y2S3, X2S3, 3)*AN1S
        if EN > 2 * object.EIN[39]:
            object.PEQIN[39][I] = object.PEQEL[1][I - IOFFN[39]]

        J=40
        if EN > object.EIN[J]:
            object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
        if EN > 2 * object.EIN[J]:
            object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
        FI+=1

        #4D1PP
        if EN > object.EIN[41]:
            object.QIN[41][I] = GasUtil.CALQINP(EN, N4D1PP, Y4D1PP, X4D1PP, 3)*AN4D
        if EN > 2 * object.EIN[41]:
            object.PEQIN[41][I] = object.PEQEL[1][I - IOFFN[41]]
            
        #4D1P
        if EN > object.EIN[42]:
            object.QIN[42][I] = GasUtil.CALQINP(EN, N4D1P, Y4D1P, X4D1P, 3)*AN4D
        if EN > 2 * object.EIN[42]:
            object.PEQIN[42][I] = object.PEQEL[1][I - IOFFN[42]]

        #3S5
        if EN > object.EIN[43]:
            object.QIN[43][I] = GasUtil.CALQINP(EN, N3S5, Y3S5, X3S5, 3)*AN1S
        if EN > 2 * object.EIN[43]:
            object.PEQIN[43][I] = object.PEQEL[1][I - IOFFN[43]]


        for J in range(44,46):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                            log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                        I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI+=1

        #4FS
        if EN > object.EIN[46]:
            object.QIN[46][I] = GasUtil.CALQINP(EN, N4FS, Y4FS, X4FS, 1)*AN4D
        if EN > 2 * object.EIN[46]:
            object.PEQIN[46][I] = object.PEQEL[1][I - IOFFN[46]]


        for J in range(47,51):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                            log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                        I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
            if EN > 2 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI+=1

        object.Q[0][I] = QELA
        for J in range(51):
            object.Q[0][I]+=object.QIN[J][I]

        for J in range(11):
            object.Q[0][I]+=object.QION[J][I]

    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
    return