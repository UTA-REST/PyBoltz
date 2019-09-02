from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow
cimport GasUtil
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
cdef void Gas7(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Xenon gas.
    """
    print("XENON")
    gd = np.load('gases.npy').item()
    cdef double EN, GAMMA1, GAMMA2, BETA, BETA2, QELA, QMOM, A, B, X1, X2, C, PQ[3], TEMP, Q456, QCORR, QTEMP,QEXC
    cdef double XEN[182], YMOM[182], XEL[153], YEL[153], XEPS[182], YEPS[182]
    cdef double XION[76], YION[76], YINC[76], YIN1[76], XIN2[54], YIN2[54],
    cdef double XIN3[47], YIN3[47], XIN4[42], YIN4[42], XIN5[37], YIN5[37], XIN6[35],
    cdef double YIN6[35], XKSH[60], YKSH[60],
    cdef double XL1S[76], YL1S[76], XL2S[76], YL2S[76], XL3S[76], YL3S[76],
    cdef double XM1S[79], YM1S[79], XM2S[80], YM2S[80], XM3S[80], YM3S[80],
    cdef double XM4S[82], YM4S[82], XM5S[83], YM5S[83]
    cdef double X1S5[70], Y1S5[70], YP1S5[70], X1S4[38], Y1S4[38], YP1S4[38],
    cdef double X1S3[46], Y1S3[46], YP1S3[46], X1S2[20], Y1S2[20], YP1S2[20],
    cdef double X2P10[22], Y2P10[22], YP2P10[22], X2P9[21], Y2P9[21], YP2P9[21],
    cdef double X2P8[22], Y2P8[22], YP2P8[22], X2P7[22], Y2P7[22], YP2P7[22],
    cdef double X2P6[22], Y2P6[22], YP2P6[22], X3D6[24], Y3D6[24], YP3D6[24],
    cdef double X2P5[15], Y2P5[15], YP2P5[15], X3D4P[24], Y3D4P[24], YP3D4P[24],
    cdef double X3D3[24], Y3D3[24], YP3D3[24], X3D4[26], Y3D4[26], YP3D4[26],
    cdef double X3D1PP[23], Y3D1PP[23], YP3D1PP[23], X3D1P[22], Y3D1P[22], YP3D1P[22],
    cdef double X2S5[18], Y2S5[18], YP2S5[18], X3P105[18], Y3P105[18], YP3P105[18],
    cdef double X2P4[14], Y2P4[14], YP2P4[14], X4DSUM[16], Y4DSUM[16], YP4DSUM[16],
    cdef double X2P3[14], Y2P3[14], YP2P3[14], X2P2[14], Y2P2[14], YP2P2[14],
    cdef double X2P1[15], Y2P1[15], YP2P1[15],
    cdef int IOFFN[50], IOFFION[12]
    cdef double Z54T[25], EBRM[25]
    XEN = gd['gas7/XEN']
    YMOM = gd['gas7/YMOM']
    XEL = gd['gas7/XEL']
    YEL = gd['gas7/YEL']
    XEPS = gd['gas7/XEPS']
    YEPS = gd['gas7/YEPS']
    XION = gd['gas7/XION']
    YION = gd['gas7/YION']
    YINC = gd['gas7/YINC']
    YIN1 = gd['gas7/YIN1']
    XIN2 = gd['gas7/XIN2']
    YIN2 = gd['gas7/YIN2']
    XIN3 = gd['gas7/XIN3']
    YIN3 = gd['gas7/YIN3']
    XIN4 = gd['gas7/XIN4']
    YIN4 = gd['gas7/YIN4']
    XIN5 = gd['gas7/XIN5']
    YIN5 = gd['gas7/YIN5']
    XIN6 = gd['gas7/XIN6']
    YIN6 = gd['gas7/YIN6']
    XKSH = gd['gas7/XKSH']
    YKSH = gd['gas7/YKSH']
    XL1S = gd['gas7/XL1S']
    YL1S = gd['gas7/YL1S']
    XL2S = gd['gas7/XL2S']
    YL2S = gd['gas7/YL2S']
    XL3S = gd['gas7/XL3S']
    YL3S = gd['gas7/YL3S']
    XM1S = gd['gas7/XM1S']
    YM1S = gd['gas7/YM1S']
    XM2S = gd['gas7/XM2S']
    YM2S = gd['gas7/YM2S']
    XM3S = gd['gas7/XM3S']
    YM3S = gd['gas7/YM3S']
    XM4S = gd['gas7/XM4S']
    YM4S = gd['gas7/YM4S']
    XM5S = gd['gas7/XM5S']
    YM5S = gd['gas7/YM5S']
    X1S5 = gd['gas7/X1S5']
    Y1S5 = gd['gas7/Y1S5']
    YP1S5 = gd['gas7/YP1S5']
    X1S4 = gd['gas7/X1S4']
    Y1S4 = gd['gas7/Y1S4']
    YP1S4 = gd['gas7/YP1S4']
    X1S3 = gd['gas7/X1S3']
    Y1S3 = gd['gas7/Y1S3']
    YP1S3 = gd['gas7/YP1S3']
    X1S2 = gd['gas7/X1S2']
    Y1S2 = gd['gas7/Y1S2']
    YP1S2 = gd['gas7/YP1S2']
    X2P10 = gd['gas7/X2P10']
    Y2P10 = gd['gas7/Y2P10']
    YP2P10 = gd['gas7/YP2P10']
    X2P9 = gd['gas7/X2P9']
    Y2P9 = gd['gas7/Y2P9']
    YP2P9 = gd['gas7/YP2P9']
    X2P8 = gd['gas7/X2P8']
    Y2P8 = gd['gas7/Y2P8']
    YP2P8 = gd['gas7/YP2P8']
    X2P7 = gd['gas7/X2P7']
    Y2P7 = gd['gas7/Y2P7']
    YP2P7 = gd['gas7/YP2P7']
    X2P6 = gd['gas7/X2P6']
    Y2P6 = gd['gas7/Y2P6']
    YP2P6 = gd['gas7/YP2P6']
    X3D6 = gd['gas7/X3D6']
    Y3D6 = gd['gas7/Y3D6']
    YP3D6 = gd['gas7/YP3D6']
    X2P5 = gd['gas7/X2P5']
    Y2P5 = gd['gas7/Y2P5']
    YP2P5 = gd['gas7/YP2P5']
    X3D4P = gd['gas7/X3D4P']
    Y3D4P = gd['gas7/Y3D4P']
    YP3D4P = gd['gas7/YP3D4P']
    X3D3 = gd['gas7/X3D3']
    Y3D3 = gd['gas7/Y3D3']
    YP3D3 = gd['gas7/YP3D3']
    X3D4 = gd['gas7/X3D4']
    Y3D4 = gd['gas7/Y3D4']
    YP3D4 = gd['gas7/YP3D4']
    X3D1PP = gd['gas7/X3D1PP']
    Y3D1PP = gd['gas7/Y3D1PP']
    YP3D1PP = gd['gas7/YP3D1PP']
    X3D1P = gd['gas7/X3D1P']
    Y3D1P = gd['gas7/Y3D1P']
    YP3D1P = gd['gas7/YP3D1P']
    X2S5 = gd['gas7/X2S5']
    Y2S5 = gd['gas7/Y2S5']
    YP2S5 = gd['gas7/YP2S5']
    X3P105 = gd['gas7/X3P105']
    Y3P105 = gd['gas7/Y3P105']
    YP3P105 = gd['gas7/YP3P105']
    X2P4 = gd['gas7/X2P4']
    Y2P4 = gd['gas7/Y2P4']
    YP2P4 = gd['gas7/YP2P4']
    X4DSUM = gd['gas7/X4DSUM']
    Y4DSUM = gd['gas7/Y4DSUM']
    YP4DSUM = gd['gas7/YP4DSUM']
    X2P3 = gd['gas7/X2P3']
    Y2P3 = gd['gas7/Y2P3']
    YP2P3 = gd['gas7/YP2P3']
    X2P2 = gd['gas7/X2P2']
    Y2P2 = gd['gas7/Y2P2']
    YP2P2 = gd['gas7/YP2P2']
    X2P1 = gd['gas7/X2P1']
    Y2P1 = gd['gas7/Y2P1']
    YP2P1 = gd['gas7/YP2P1']
    Z54T = gd['gas7/Z54T']
    EBRM = gd['gas7/EBRM']
    #   BORN BETHE VALUES FOR IONISATION
    CONST = 1.873884e-20
    EMASS2 = 1021997.804
    API = acos(-1)
    A0 = 0.52917720859e-8
    RY = 13.60569193
    BBCONST = 16.0 * API * A0 * A0 * RY * RY / EMASS2

    AM2 = 8.04
    C = 75.25

    # AVERAGE AUGER EMISSIONS FROM EACH SHELL
    AUGM5 = 4.34
    AUGM4 = 4.43
    AUGM3 = 6.79
    AUGM2 = 6.85
    AUGM1 = 7.94
    AUGL3 = 8.21
    AUGL2 = 8.45
    AUGL1 = 9.39
    AUGK = 8.49

    object.NION = 12
    object.NATT = 1
    object.NIN = 50
    object.NNULL = 0
    NBREM = 25

    cdef int J, I

    for J in range(6):
        object.KEL[J] = object.WhichAngularModel
    for J in range(object.NIN):
        object.KIN[J] = object.WhichAngularModel
    cdef int NDATA, NEL, NEPSI, NIONG, NION2, NION3, NION4, NION5, NION6, NIONK, NIONL1, NIONL2, NIONL3, NIONM1, NIONM2, NIONM3, NIONM4
    cdef int NIONM5, N1S5, N1S4, N1S3, N1S2, N2P10, N2P9, N2P8, N2P7, N2P6, N3D6, N2P5, N3D4P, N3D3, N3D4, N3D1PP, N3D1P, N2S5, N3PSUM, N2P4
    cdef int N4DSUM, N2P3, N2P2, N2P1
    NDATA = 182
    NEL = 153
    NEPSI = 182
    NIONG = 76
    NION2 = 54
    NION3 = 47
    NION4 = 42
    NION5 = 37
    NION6 = 35
    NIONK = 60
    NIONL1 = 76
    NIONL2 = 76
    NIONL3 = 76
    NIONM1 = 79
    NIONM2 = 80
    NIONM3 = 80
    NIONM4 = 82
    NIONM5 = 83
    N1S5 = 70
    N1S4 = 38
    N1S3 = 46
    N1S2 = 20
    N2P10 = 22
    N2P9 = 21
    N2P8 = 22
    N2P7 = 22
    N2P6 = 22
    N3D6 = 24
    N2P5 = 15
    N3D4P = 24
    N3D3 = 24
    N3D4 = 26
    N3D1PP = 23
    N3D1P = 22
    N2S5 = 18
    N3PSUM = 18
    N2P4 = 14
    N4DSUM = 16
    N2P3 = 14
    N2P2 = 14
    N2P1 = 15
    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27
    object.E = [0.0, 1.0, 12.129843, 0.0, 0.0, 23.7]
    object.E[1] = 2.0 * EMASS / (131.30 * AMU)
    cdef double EOBY[12]

    EOBY[0:12] = [8.7, 20.0, 38.0, 400., 410., 750.0, 800.0, 920.0, 3850., 4100., 4400., 34561.]
    object.EION[0:12] = [12.129843, 33.105, 64.155, 676.4, 689.0, 940.6, 1002.1, 1148.7, 4786., 5107., 5453., 34561.]
    # FLUORESCENCE DATA
    object.NC0[0:12] = [0, 1, 2, 4, 4, 7, 7, 8, 9, 9, 10, 17]
    object.EC0[0:12] = [0.0, 5.0, 10.0, 593.7, 604.0, 782.2, 839.7, 911.4, 4494.3, 4774.8, 5015.2, 33900]
    cdef double WKLM[12]
    WKLM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0942, 0.093, 0.0475, 0.89]
    object.WK[0:12] = WKLM
    object.EFL[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4106, 4427, 4483, 29775]
    object.NG1[0:12] = [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 8]
    object.EG1[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3900, 4181, 4233, 29406]
    object.NG2[0:12] = [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 6, 9]
    object.EG2[0:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 594, 594, 782, 4494]

    for J in range(object.NION):
        for I in range(4000):
            if object.EG[I] > object.EION[J]:
                IOFFION[J] = I - 1
                break

    object.EIN[0:51] = [8.3153, 8.4365, 9.4472, 9.5697, 9.5802, 9.6856, 9.7207, 9.7893, 9.8211, 9.8904, 9.9171, 9.9335,
                        9.9431, 9.9588, 10.0391, 10.1575, 10.2200, 10.4010, 10.5621, 10.5932, 10.9016, 10.9576, 10.9715,
                        10.9788, 11.0547, 11.0691, 11.1412, 11.1626, 11.2742, 11.4225, 11.4951, 11.5829, 11.6072,
                        11.6828, 11.7395, 11.7521, 11.8068, 11.8403, 11.8518, 11.8778, 11.8917, 11.9082, 11.9177,
                        11.9416, 11.9550, 11.9621, 11.9789, 11.9886, 11.9939, 12.0, 0.0]
    for I in range(51, 250):
        object.EIN[I] = 0.0

    cdef int NL
    for NL in range(object.NIN):
        object.PENFRA[0][NL] = 0.0
        # PENNING TRANSFER DISTANCE MICRONS
        object.PENFRA[1][NL] = 1.0
        # PENNING TRANSFER TIME PICOSECONDS
        object.PENFRA[2][NL] = 1.0

    for NL in range(object.NIN):
        for I in range(4000):
            if object.EG[I] > object.EIN[NL]:
                IOFFN[NL] = I - 1
                break
    object.NSTEP = 4000


    for I in range(object.NSTEP):
        EN = object.EG[I]
        if EN > object.EIN[0]:
            GAMMA1 = (EMASS2 + 2.0 * EN) / EMASS2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = sqrt(1.0 - 1.0 / GAMMA2)
            BETA2 = BETA * BETA

        if EN <= XEN[1]:
            QELA = 122.e-16
            QMOM = 122.e-16
        else:
            QELA = GasUtil.QLSCALE(EN, NEL, YEL, XEL)
            QMOM = GasUtil.QLSCALE(EN, NDATA, YMOM, XEN)

        TEMP = GasUtil.CALPQ3(EN, NEPSI, YEPS, XEPS)

        PQ = [0.5, 0.5 + (QELA - QMOM) / QELA, 1 - TEMP]

        object.PEQEL[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = QELA

        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM
        #  IONISATION CHARGE STATE =1


        object.QION[0][I] = 0.0
        object.PEQION[0][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[0][I] = 0
        if EN >= object.EION[0]:
            object.QION[0][I] = GasUtil.CALQIONX(EN, NIONG, YIN1, XION, BETA2, 0.8061, CONST, object.DEN[I], C, AM2)


        # USE AAnisotropicDetectedTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
        # ENERGIES ABOVE 2 * IONISATION ENERGY
        # AAnisotropicDetectedTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFF SET BY
        # IONISATION ENERGY
        if EN > (2 * object.EION[0]):
            object.PEQION[0][I] = object.PEQEL[1][I - IOFFION[0]]

        #  IONISATION CHARGE STATE =2
        object.QION[1][I] = 0.0
        object.PEQION[1][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[1][I] = 0
        if EN >= object.EION[1]:
            object.QION[1][I] = GasUtil.CALQIONX(EN, NION2, YIN2, XIN2, BETA2, 0.1133, CONST, object.DEN[I], C, AM2)

        # USE AAnisotropicDetectedTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
        # ENERGIES ABOVE 2 * IONISATION ENERGY
        # AAnisotropicDetectedTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFF SET BY
        # IONISATION ENERGY
        if EN > (2 * object.EION[1]):
            object.PEQION[1][I] = object.PEQEL[1][I - IOFFION[1]]

        #  IONISATION CHARGE STATE =3
        object.QION[2][I] = 0.0
        object.PEQION[2][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[2][I] = 0
        if EN >= object.EION[2]:
            object.QION[2][I] = GasUtil.CALQIONX(EN, NION3, YIN3, XIN3, BETA2, 0.05496, CONST, object.DEN[I], C, AM2)
        # USE AAnisotropicDetectedTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
        # ENERGIES ABOVE 2 * IONISATION ENERGY
        # AAnisotropicDetectedTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFF SET BY
        # IONISATION ENERGY
        if EN > (2 * object.EION[2]):
            object.PEQION[2][I] = object.PEQEL[2][I - IOFFION[2]]

        Q456 = 0.0
        if EN > 106.35:
            TEMP = GasUtil.CALQION(EN, NION4, YIN4, XIN4)
            Q456 = TEMP * 4.0 / 3.0
        if EN > 160.45:
            TEMP = GasUtil.CALQION(EN, NION5, YIN5, XIN5)
            Q456 += (TEMP * 5.0 / 3.0)
        if EN > 227.2:
            TEMP = GasUtil.CALQION(EN, NION6, YIN6, XIN6)
            Q456 += (TEMP * 6.0 / 3.0)

        if EN > XIN4[NION4 - 1] or EN > XIN5[NION5 - 1] or EN > XIN6[NION6 - 1]:
            X2 = 1.0 / BETA2
            X1 = X2 * log(BETA2 / (1.0 - BETA2)) - 1.0
            #  0.3629 = .01959*4/3 + .004597*5/3  + .002504*6/3
            Q456 = CONST * (AM2 * (X1 - object.DEN[I] / 2.0) + C * X2) * 0.03629

        object.QION[2][I] += Q456

        #M5-SHELL IONISATION
        object.QION[3][I] = 0.0
        object.PEQION[3][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[3][I] = 0
        if EN >= object.EION[3]:
            object.QION[3][I] = GasUtil.CALQIONREG(EN, NIONM5, YM5S, XM5S)
            object.PEQION[3][I] = object.PEQEL[1][I - IOFFION[3]]

        #M4-SHELL IONISATION
        object.QION[4][I] = 0.0
        object.PEQION[4][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[4][I] = 0
        if EN >= object.EION[4]:
            object.QION[4][I] = GasUtil.CALQIONREG(EN, NIONM4, YM4S, XM4S)
            object.PEQION[4][I] = object.PEQEL[1][I - IOFFION[4]]

        #M3-SHELL IONISATION
        object.QION[5][I] = 0.0
        object.PEQION[5][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[5][I] = 0
        if EN >= object.EION[5]:
            object.QION[5][I] = GasUtil.CALQIONREG(EN, NIONM3, YM3S, XM3S)
            object.PEQION[5][I] = object.PEQEL[1][I - IOFFION[5]]

        #M2-SHELL IONISATION
        object.QION[6][I] = 0.0
        object.PEQION[6][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[6][I] = 0
        if EN >= object.EION[6]:
            object.QION[6][I] = GasUtil.CALQIONREG(EN, NIONM2, YM2S, XM2S)
            object.PEQION[6][I] = object.PEQEL[1][I - IOFFION[6]]

        #M1-SHELL IONISATION
        object.QION[7][I] = 0.0
        object.PEQION[7][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[7][I] = 0
        if EN >= object.EION[7]:
            object.QION[7][I] = GasUtil.CALQIONREG(EN, NIONM1, YM1S, XM1S)
            object.PEQION[7][I] = object.PEQEL[1][I - IOFFION[7]]

        #L3-SHELL IONISATION
        object.QION[8][I] = 0.0
        object.PEQION[8][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[8][I] = 0
        if EN >= object.EION[8]:
            object.QION[8][I] = GasUtil.CALQIONREG(EN, NIONL3, YL3S, XL3S)
            object.PEQION[8][I] = object.PEQEL[1][I - IOFFION[8]]

        #L2-SHELL IONISATION
        object.QION[9][I] = 0.0
        object.PEQION[9][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[9][I] = 0
        if EN >= object.EION[9]:
            object.QION[9][I] = GasUtil.CALQIONREG(EN, NIONL2, YL2S, XL2S)
            object.PEQION[9][I] = object.PEQEL[1][I - IOFFION[9]]

        #L1-SHELL IONISATION
        object.QION[10][I] = 0.0
        object.PEQION[10][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[10][I] = 0
        if EN >= object.EION[10]:
            object.QION[10][I] = GasUtil.CALQIONREG(EN, NIONL1, YL1S, XL1S)
            object.PEQION[10][I] = object.PEQEL[1][I - IOFFION[10]]

        #K-SHELL IONISATION
        object.QION[11][I] = 0.0
        object.PEQION[11][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQION[11][I] = 0
        if EN >= object.EION[11]:
            object.QION[11][I] = GasUtil.CALQIONREG(EN, NIONK, YKSH, XKSH)
            object.PEQION[11][I] = object.PEQEL[1][I - IOFFION[11]]

        # ATTACHMENT
        object.Q[3][I] = 0.0
        object.QATT[3][I] = object.Q[3][I]

        # COUNTING IONISATION
        object.Q[4][I] = 0.0
        object.PEQEL[4][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEQEL[4][I] = 0.0
        if EN > object.EION[0]:
            object.Q[4][I] = GasUtil.CALQIONX(EN, NIONG, YINC, XION, BETA2, 1.0, CONST, object.DEN[I], C, AM2)

        # CORRECTION TO CHARGE STATE 1 2 3+4+5+6 X-SECTION FOR K L AND M SHELLS
        # CORRECTION GIVES TOTAL IONISATION EQUAL TO OSCILLATOR SUM
        QTEMP = 0.0
        for J in range(4, 12):
            QTEMP += object.QION[J][I]

        if object.Q[4][I] == 0.0:
            QCORR = 1.0
        else:
            QCORR = (object.Q[4][I] - QTEMP) / object.Q[4][I]
        object.QION[0][I] *= QCORR
        object.QION[1][I] *= QCORR
        object.QION[2][I] *= QCORR

        object.Q[5][I] = 0.0

        for NL in range(object.NIN + 1):
            object.QIN[NL][I] = 0.0
            object.PEQIN[NL][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEQIN[NL][I] = 0.0

        #1S5
        if EN > object.EIN[0]:
            object.QIN[0][I] = GasUtil.CALQINP(EN, N1S5, Y1S5, X1S5, 3)
        if EN > (2 * object.EIN[0]):
            object.PEQIN[0][I] = object.PEQEL[1][I - IOFFN[0]]

        #1S4 F=0.260
        if EN > object.EIN[1]:
            if EN <= X1S4[N1S4]:
                object.QIN[1][I] = GasUtil.CALQIN(EN, N1S4, Y1S4, X1S4)
            else:
                object.QIN[1][I] = 0.260 / (object.EIN[1] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[1])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[1] + object.E[2])
            if EN > (2 * object.EIN[1]):
                object.PEQIN[1][I] = object.PEQEL[1][I - IOFFN[1]]

        #1S3
        if EN > object.EIN[2]:
            object.QIN[2][I] = GasUtil.CALQINP(EN, N1S3, Y1S3, X1S3, 3)
        if EN > (2 * object.EIN[2]):
            object.PEQIN[2][I] = object.PEQEL[1][I - IOFFN[2]]

        #1S2 F=0.183
        if EN > object.EIN[3]:
            if EN <= X1S2[N1S2]:
                object.QIN[3][I] = GasUtil.CALQIN(EN, N1S2, Y1S2, X1S2)
            else:
                object.QIN[3][I] = 0.183 / (object.EIN[3] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[3])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[3] + object.E[2])
            if EN > (2 * object.EIN[1]):
                object.PEQIN[3][I] = object.PEQEL[1][I - IOFFN[3]]

        #P STATES
        #2P10
        if EN > object.EIN[4]:
            object.QIN[4][I] = GasUtil.CALQINP(EN, N2P10, Y2P10, X2P10, 3)
        if EN > (2 * object.EIN[4]):
            object.PEQIN[4][I] = object.PEQEL[1][I - IOFFN[4]]

        #2P9
        if EN > object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQINP(EN, N2P9, Y2P9, X2P9, 1)
        if EN > (2 * object.EIN[5]):
            object.PEQIN[5][I] = object.PEQEL[1][I - IOFFN[5]]

        #2P8
        if EN > object.EIN[6]:
            object.QIN[6][I] = GasUtil.CALQINP(EN, N2P8, Y2P8, X2P8, 3)
        if EN > (2 * object.EIN[6]):
            object.PEQIN[6][I] = object.PEQEL[1][I - IOFFN[6]]

        #2P7
        if EN > object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQINP(EN, N2P7, Y2P7, X2P7, 2)
        if EN > (2 * object.EIN[7]):
            object.PEQIN[7][I] = object.PEQEL[1][I - IOFFN[7]]

        #2P6
        if EN > object.EIN[8]:
            object.QIN[8][I] = GasUtil.CALQINP(EN, N2P6, Y2P6, X2P6, 1)
        if EN > (2 * object.EIN[8]):
            object.PEQIN[8][I] = object.PEQEL[1][I - IOFFN[8]]

        #3D6
        if EN > object.EIN[9]:
            object.QIN[9][I] = GasUtil.CALQINP(EN, N3D6, Y3D6, X3D6, 1.5)
        if EN > (2 * object.EIN[9]):
            object.PEQIN[9][I] = object.PEQEL[1][I - IOFFN[9]]

        #3D5 F=0.0100
        if EN > object.EIN[10]:
            object.QIN[10][I] = 0.01 / (object.EIN[10] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[10])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[10] + object.E[2])
            if EN > (2 * object.EIN[10]):
                object.PEQIN[10][I] = object.PEQEL[1][I - IOFFN[10]]
        #2P5
        if EN > object.EIN[11]:
            object.QIN[11][I] = GasUtil.CALQINP(EN, N2P5, Y2P5, X2P5, 1)
        if EN > (2 * object.EIN[11]):
            object.PEQIN[11][I] = object.PEQEL[1][I - IOFFN[11]]

        #3D4!
        if EN > object.EIN[12]:
            object.QIN[12][I] = GasUtil.CALQINP(EN, N3D4P, Y3D4P, X3D4P, 1.5)
        if EN > (2 * object.EIN[12]):
            object.PEQIN[12][I] = object.PEQEL[1][I - IOFFN[12]]

        #3D3
        if EN > object.EIN[13]:
            object.QIN[13][I] = GasUtil.CALQINP(EN, N3D3, Y3D3, X3D3, 1.5)
        if EN > (2 * object.EIN[13]):
            object.PEQIN[13][I] = object.PEQEL[1][I - IOFFN[13]]

        #3D4
        if EN > object.EIN[14]:
            object.QIN[14][I] = GasUtil.CALQINP(EN, N3D4, Y3D4, X3D4, 2)
        if EN > (2 * object.EIN[14]):
            object.PEQIN[14][I] = object.PEQEL[1][I - IOFFN[14]]

        #3D1!!
        if EN > object.EIN[15]:
            object.QIN[15][I] = GasUtil.CALQINP(EN, N3D1PP, Y3D1PP, X3D1PP, 3)
        if EN > (2 * object.EIN[15]):
            object.PEQIN[15][I] = object.PEQEL[1][I - IOFFN[15]]

        #3D1!
        if EN > object.EIN[16]:
            object.QIN[16][I] = GasUtil.CALQINP(EN, N3D1P, Y3D1P, X3D1P, 1)
        if EN > (2 * object.EIN[16]):
            object.PEQIN[16][I] = object.PEQEL[1][I - IOFFN[16]]

        #3D2 F=0.379
        if EN > object.EIN[17]:
            object.QIN[17][I] = 0.379 / (object.EIN[17] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[17])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[17] + object.E[2])
            if EN > (2 * object.EIN[17]):
                object.PEQIN[17][I] = object.PEQEL[1][I - IOFFN[17]]

        #2S5
        if EN > object.EIN[18]:
            object.QIN[18][I] = GasUtil.CALQINP(EN, N2S5, Y2S5, X2S5, 3)
        if EN > (2 * object.EIN[18]):
            object.PEQIN[18][I] = object.PEQEL[1][I - IOFFN[18]]

        #2S4 J=1 F=0.086
        if EN > object.EIN[19]:
            object.QIN[19][I] = 0.086 / (object.EIN[19] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[19])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[19] + object.E[2])
            if EN > (2 * object.EIN[19]):
                object.PEQIN[19][I] = object.PEQEL[1][I - IOFFN[19]]

        #SUM 3P10+3P9+3P8+3P7+3P6+3P5
        if EN > object.EIN[20]:
            object.QIN[20][I] = GasUtil.CALQINP(EN, N3PSUM, Y3P105, X3P105, 1)
        if EN > (2 * object.EIN[20]):
            object.PEQIN[20][I] = object.PEQEL[1][I - IOFFN[20]]

        #2P4
        if EN > object.EIN[21]:
            object.QIN[21][I] = GasUtil.CALQINP(EN, N2P4, Y2P4, X2P4, 2)
        if EN > (2 * object.EIN[21]):
            object.PEQIN[21][I] = object.PEQEL[1][I - IOFFN[21]]

        #SUM 4D6+4D3+4D4P+4D4+4D1PP+4D1P
        if EN > object.EIN[22]:
            object.QIN[22][I] = GasUtil.CALQINP(EN, N4DSUM, Y4DSUM, X4DSUM, 3)
        if EN > (2 * object.EIN[22]):
            object.PEQIN[22][I] = object.PEQEL[1][I - IOFFN[22]]

        # 4D5 J=1 F=0.0010
        if EN > object.EIN[23]:
            object.QIN[23][I] = 0.001 / (object.EIN[23] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[23])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[23] + object.E[2])
            if EN > (2 * object.EIN[23]):
                object.PEQIN[23][I] = object.PEQEL[1][I - IOFFN[23]]

        #2P3
        if EN > object.EIN[24]:
            object.QIN[24][I] = GasUtil.CALQINP(EN, N2P3, Y2P3, X2P3, 1)

        #2P2
        if EN > object.EIN[25]:
            object.QIN[25][I] = GasUtil.CALQINP(EN, N2P2, Y2P2, X2P2, 2)

        #2P1
        if EN > object.EIN[26]:
            object.QIN[26][I] = GasUtil.CALQINP(EN, N2P1, Y2P1, X2P1, 1)

        # 4D2 J=1 F=0.0835
        if EN > object.EIN[27]:
            object.QIN[27][I] = 0.0835 / (object.EIN[27] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[27])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[27] + object.E[2])

            if EN > (2 * object.EIN[27]):
                object.PEQIN[27][I] = object.PEQEL[1][I - IOFFN[27]]

        # 3S4 J=1 F=0.0225
        if EN > object.EIN[28]:
            object.QIN[28][I] = 0.0225 / (object.EIN[28] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[28])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[28] + object.E[2])
            if EN > (2 * object.EIN[28]):
                object.PEQIN[28][I] = object.PEQEL[1][I - IOFFN[28]]

        # 5D5 J=1 F=0.0227
        if EN > object.EIN[29]:
            object.QIN[29][I] = 0.0227 / (object.EIN[29] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[29])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[29] + object.E[2])
            if EN > (2 * object.EIN[29]):
                object.PEQIN[29][I] = object.PEQEL[1][I - IOFFN[29]]

        # 5D2 J=1 F=0.002
        if EN > object.EIN[30]:
            object.QIN[30][I] = 0.002 / (object.EIN[30] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[30])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[30] + object.E[2])
            if EN > (2 * object.EIN[30]):
                object.PEQIN[30][I] = object.PEQEL[1][I - IOFFN[30]]

        # 4S4 J=1 F=0.0005
        if EN > object.EIN[31]:
            object.QIN[31][I] = 0.0005 / (object.EIN[31] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[31])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[31] + object.E[2])
            if EN > (2 * object.EIN[31]):
                object.PEQIN[31][I] = object.PEQEL[1][I - IOFFN[31]]

        # 3S1! J=1 F=0.1910
        if EN > object.EIN[32]:
            object.QIN[32][I] = 0.191 / (object.EIN[32] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[32])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[32] + object.E[2])
            if EN > (2 * object.EIN[32]):
                object.PEQIN[32][I] = object.PEQEL[1][I - IOFFN[32]]

        # 6D5 J=1 F=0.0088
        if EN > object.EIN[33]:
            object.QIN[33][I] = 0.0088 / (object.EIN[33] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[33])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[33] + object.E[2])
            if EN > (2 * object.EIN[33]):
                object.PEQIN[33][I] = object.PEQEL[1][I - IOFFN[33]]

        # 6D2 J=1 F=0.0967
        if EN > object.EIN[34]:
            object.QIN[34][I] = 0.0967 / (object.EIN[34] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[34])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[34] + object.E[2])
            if EN > (2 * object.EIN[34]):
                object.PEQIN[34][I] = object.PEQEL[1][I - IOFFN[34]]

        # 5S4 J=1 F=0.0967
        if EN > object.EIN[35]:
            object.QIN[35][I] = 0.0288 / (object.EIN[35] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[35])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[35] + object.E[2])
            if EN > (2 * object.EIN[35]):
                object.PEQIN[35][I] = object.PEQEL[1][I - IOFFN[35]]

        # 7D5 J=1 F=0.0042
        if EN > object.EIN[36]:
            object.QIN[36][I] = 0.0042 / (object.EIN[36] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[36])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[36] + object.E[2])
            if EN > (2 * object.EIN[36]):
                object.PEQIN[36][I] = object.PEQEL[1][I - IOFFN[36]]

        # 7D2 J=1 F=0.0625
        if EN > object.EIN[37]:
            object.QIN[37][I] = 0.0625 / (object.EIN[37] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[37])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[37] + object.E[2])
            if EN > (2 * object.EIN[37]):
                object.PEQIN[37][I] = object.PEQEL[1][I - IOFFN[37]]

        # 6S4 J=1 F=0.0025
        if EN > object.EIN[38]:
            object.QIN[38][I] = 0.0025 / (object.EIN[38] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[37])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[37] + object.E[2])
            if EN > (2 * object.EIN[37]):
                object.PEQIN[37][I] = object.PEQEL[1][I - IOFFN[37]]

        # 2S2 J=1 F=0.029
        if EN > object.EIN[39]:
            object.QIN[39][I] = 0.029 / (object.EIN[39] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[39])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[39] + object.E[2])
            if EN > (2 * object.EIN[39]):
                object.PEQIN[39][I] = object.PEQEL[1][I - IOFFN[39]]

        # 8D5 J=1 F=0.0035
        if EN > object.EIN[40]:
            object.QIN[40][I] = 0.0035 / (object.EIN[40] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[40])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[40] + object.E[2])
            if EN > (2 * object.EIN[40]):
                object.PEQIN[40][I] = object.PEQEL[1][I - IOFFN[40]]

        # 8D2 J=1 F=0.0386
        if EN > object.EIN[41]:
            object.QIN[41][I] = 0.0386 / (object.EIN[41] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[41])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[41] + object.E[2])
            if EN > (2 * object.EIN[41]):
                object.PEQIN[41][I] = object.PEQEL[1][I - IOFFN[41]]

        # 7S4 J=1 F=0.005
        if EN > object.EIN[42]:
            object.QIN[42][I] = 0.005 / (object.EIN[42] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[42])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[42] + object.E[2])
            if EN > (2 * object.EIN[42]):
                object.PEQIN[42][I] = object.PEQEL[1][I - IOFFN[42]]

        # 9D5 J=1 F=0.0005
        if EN > object.EIN[43]:
            object.QIN[43][I] = 0.0005 / (object.EIN[43] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[43])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[43] + object.E[2])
            if EN > (2 * object.EIN[43]):
                object.PEQIN[43][I] = object.PEQEL[1][I - IOFFN[43]]

        # 9D2 J=1 F=0.0250
        if EN > object.EIN[44]:
            object.QIN[44][I] = 0.025 / (object.EIN[44] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[44])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[44] + object.E[2])
            if EN > (2 * object.EIN[44]):
                object.PEQIN[44][I] = object.PEQEL[1][I - IOFFN[44]]

        # 8S4 J=1 F=0.0023
        if EN > object.EIN[45]:
            object.QIN[45][I] = 0.0023 / (object.EIN[45] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[45])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[45] + object.E[2])
            if EN > (2 * object.EIN[45]):
                object.PEQIN[45][I] = object.PEQEL[1][I - IOFFN[45]]

        #10D5 J=1 F=0.0005
        if EN > object.EIN[46]:
            object.QIN[46][I] = 0.0005 / (object.EIN[46] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[46])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[46] + object.E[2])
            if EN > (2 * object.EIN[46]):
                object.PEQIN[46][I] = object.PEQEL[1][I - IOFFN[46]]

        #10D2 J=1 F=0.0164
        if EN > object.EIN[47]:
            object.QIN[47][I] = 0.0164 / (object.EIN[47] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[47])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[47] + object.E[2])
            if EN > (2 * object.EIN[47]):
                object.PEQIN[47][I] = object.PEQEL[1][I - IOFFN[47]]

        #9S4 J=1 F=0.0014
        if EN > object.EIN[48]:
            object.QIN[48][I] = 0.0014 / (object.EIN[48] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[48])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[48] + object.E[2])
            if EN > (2 * object.EIN[48]):
                object.PEQIN[48][I] = object.PEQEL[1][I - IOFFN[48]]

        #HIGH J=1 F=0.0831
        if EN > object.EIN[49]:
            object.QIN[49][I] = 0.0831 / (object.EIN[49] * BETA2) * (
                    log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[49])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EIN[49] + object.E[2])
            if EN > (2 * object.EIN[49]):
                object.PEQIN[49][I] = object.PEQEL[1][I - IOFFN[49]]
        for J in range(26, 50):
            if object.QIN[J][I] < 0.0:
                object.QIN[J][I] = 0.0
        QEXC=0.0
        for J in range(object.NIN):
            QEXC+=object.QIN[J][I]
        for J in range(12):
            object.Q[0][I]+=object.QION[J][I]
        object.Q[0][I]+=QEXC

    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break

