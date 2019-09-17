from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow
cimport GasUtil
cimport libc.math
import numpy as np
cimport numpy as np
import sys
from Gas cimport Gas
from cython.parallel import prange
import sys

sys.path.append('../hdf5_python')
import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.fast_getattr(True)
def MERT( epsilon, A, D, F, A1):
    a0 = 1  # 5.29e-11  # in m
    hbar = 1  # 197.32697*1e-9 # in eV m
    m = 1  # 511e3     # eV/c**2
    alpha = 27.292 * a0 ** 3
    k = np.sqrt((epsilon) / (13.605 * a0 ** 2))

    eta0 = -A * k * (1 + (4 * alpha) / (3 * a0) * k ** 2 * np.log(k * a0)) \
           - (np.pi * alpha) / (3 * a0) * k ** 2 + D * k ** 3 + F * k ** 4

    eta1 = (np.pi) / (15 * a0) * alpha * k ** 2 - A1 * k ** 3

    Qm = (4 * np.pi * a0 ** 2) / (k ** 2) * (np.sin(np.arctan(eta0) - np.arctan(eta1))) ** 2

    Qt = (4 * np.pi * a0 ** 2) / (k ** 2) * (np.sin(np.arctan(eta0))) ** 2

    return Qm * (5.29e-11) ** 2 * 1e20, Qt * (5.29e-11) ** 2 * 1e20

def WEIGHT_Q( eV, Qm, BashBoltzQm, Lamda, eV0):
    WeightQm = (1 - np.tanh(Lamda * (eV - eV0))) / 2
    WeightBB = (1 + np.tanh(Lamda * (eV - eV0))) / 2

    NewBashQm = BashBoltzQm * WeightBB
    NewMERTQm = Qm * WeightQm
    NewQm = NewBashQm + NewMERTQm
    return NewQm

def HYBRID_X_SECTIONS( MB_EMTx, MB_EMTy, MB_ETx, MB_ETy, A, D, F, A1, Lambda, eV0):
    Qm_MERT, Qt_MERT = MERT(MB_EMTx, A, D, F, A1)
    New_Qm = WEIGHT_Q(MB_EMTx, Qm_MERT, MB_EMTy, Lambda, eV0)
    Qm_MERT, Qt_MERT = MERT(MB_ETx, A, D, F, A1)
    New_Qt = WEIGHT_Q(MB_ETx, Qt_MERT, MB_ETy, Lambda, eV0)

    return MB_EMTx, New_Qm, MB_ETx, New_Qt


cdef void Gas61(Gas*object,double AA,double D, double F, double A1, double Lambda,double EV0):
    """
    This function is used to calculate the needed momentum cross sections for Xenon gas.
    """
    gd = np.load('gases.npy').item()
    cdef double EN, GAMMA1, GAMMA2, BETA, BETA2, ElasticCrossSectionA, QMOM, A, B, X1, X2, C, PQ[3], TEMP, Q456, QCORR, QTEMP, QEXC
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
    cdef double X2P4[14], Y2P4[14], YP2P4[14], X4DSum[16], Y4DSum[16], YP4DSum[16],
    cdef double X2P3[14], Y2P3[14], YP2P3[14], X2P2[14], Y2P2[14], YP2P2[14],
    cdef double X2P1[15], Y2P1[15], YP2P1[15],
    cdef int IOFFN[50], IOFFION[12]
    cdef double Z54T[25], EBRM[25],temp[183]

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
    X4DSum = gd['gas7/X4DSum']
    Y4DSum = gd['gas7/Y4DSum']
    YP4DSum = gd['gas7/YP4DSum']
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

    if AA != 0 and F != 0 and D != 0 and A1 != 0 and Lambda != 0 and EV0 != 0:
        for i in range(182):
            XEN[i], YMOM[i], temp[i], temp[i] = HYBRID_X_SECTIONS(XEN[i],
                                                                YMOM[i],
                                                                temp[i],
                                                                temp[i], AA,
                                                                D, F, A1,
                                                                Lambda, EV0)
            if YMOM[i]!=YMOM[i]:
                YMOM[i] = 0

        for i in range(153):
            XEN[i], temp[i], XEL[i], YEL[i] = HYBRID_X_SECTIONS(temp[i],
                                                                temp[i],
                                                                XEL[i],
                                                                YEL[i], AA,
                                                                D, F, A1,
                                                                Lambda, EV0)
            if YEL[i]!=YEL[i]:
                YEL[i] = 0

        YMOM[0] = 131
        YEL[0] = 131

    #   BORN BETHE VALUES FOR IONISATION
    CONST = 1.873884e-20
    ElectronMass2 = <float> (1021997.804)
    API = acos(-1)
    A0 = 0.52917720859e-8
    RY = <float> (13.60569193)
    BBCONST = 16.0 * API * A0 * A0 * RY * RY / ElectronMass2

    AM2 = <float> (8.04)
    C = <float> (75.25)

    # AVERAGE AUGER EMISSIONS FROM EACH SHELL
    AUGM5 = <float> (4.34)
    AUGM4 = <float> (4.43)
    AUGM3 = <float> (6.79)
    AUGM2 = <float> (6.85)
    AUGM1 = <float> (7.94)
    AUGL3 = <float> (8.21)
    AUGL2 = <float> (8.45)
    AUGL1 = <float> (9.39)
    AUGK = <float> (8.49)

    object.N_Ionization = 12
    object.N_Attachment = 1
    object.N_Inelastic = 50
    object.N_Null = 0
    NBREM = 25

    cdef int J, I

    for J in range(6):
        object.KEL[J] = object.WhichAngularModel
    for J in range(object.N_Inelastic):
        object.KIN[J] = object.WhichAngularModel
    cdef int NDATA, NEL, NEPSI, N_IonizationG, N_Ionization2, N_Ionization3, N_Ionization4, N_Ionization5, N_Ionization6, N_IonizationK, N_IonizationL1, N_IonizationL2, N_IonizationL3, N_IonizationM1, N_IonizationM2, N_IonizationM3, N_IonizationM4
    cdef int N_IonizationM5, N1S5, N1S4, N1S3, N1S2, N2P10, N2P9, N2P8, N2P7, N2P6, N3D6, N2P5, N3D4P, N3D3, N3D4, N3D1PP, N3D1P, N2S5, N3PSum, N2P4
    cdef int N4DSum, N2P3, N2P2, N2P1
    NDATA = 182
    NEL = 153
    NEPSI = 182
    N_IonizationG = 76
    N_Ionization2 = 54
    N_Ionization3 = 47
    N_Ionization4 = 42
    N_Ionization5 = 37
    N_Ionization6 = 35
    N_IonizationK = 60
    N_IonizationL1 = 76
    N_IonizationL2 = 76
    N_IonizationL3 = 76
    N_IonizationM1 = 79
    N_IonizationM2 = 80
    N_IonizationM3 = 80
    N_IonizationM4 = 82
    N_IonizationM5 = 83
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
    N3PSum = 18
    N2P4 = 14
    N4DSum = 16
    N2P3 = 14
    N2P2 = 14
    N2P1 = 15
    cdef double ElectronMass = 9.10938291e-31
    cdef double AMU = 1.660538921e-27
    object.E = [0.0, 1.0, <float> (12.129843), 0.0, 0.0, <float> (23.7)]
    object.E[1] = 2.0 * ElectronMass / (<float> (131.30) * AMU)
    cdef double EOBY[12]

    EOBY[0:12] = [8.7, 20.0, 38.0, 400., 410., 750.0, 800.0, 920.0, 3850., 4100., 4400., 34561.]
    object.IonizationEnergy[0:12] = [<float> (12.129843), <float> (33.105), <float> (64.155), <float> (676.4), <float> (689.0),
                         <float> (940.6), <float> (1002.1), <float> (1148.7), <float> (4786.), <float> (5107.), 5453.,
                         34561.]
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

    for J in range(object.N_Ionization):
        for I in range(4000):
            if object.EG[I] > object.IonizationEnergy[J]:
                IOFFION[J] = I
                break

    object.EnergyLevels[0:51] = [np.float32(8.3153), np.float32(8.4365), np.float32(9.4472), np.float32(9.5697),
                        np.float32(9.5802), np.float32(9.6856), np.float32(9.7207), np.float32(9.7893),
                        np.float32(9.8211), np.float32(9.8904), np.float32(9.9171), np.float32(9.9335),
                        np.float32(9.9431), np.float32(9.9588), np.float32(10.0391), np.float32(10.1575),
                        np.float32(10.2200), np.float32(10.4010), np.float32(10.5621), np.float32(10.5932),
                        np.float32(10.9016), np.float32(10.9576), np.float32(10.9715), np.float32(10.9788),
                        np.float32(11.0547), np.float32(11.0691), np.float32(11.1412), np.float32(11.1626),
                        np.float32(11.2742), np.float32(11.4225), np.float32(11.4951), np.float32(11.5829),
                        np.float32(11.6072), np.float32(11.6828), np.float32(11.7395), np.float32(11.7521),
                        np.float32(11.8068), np.float32(11.8403), np.float32(11.8518), np.float32(11.8778),
                        np.float32(11.8917), np.float32(11.9082), np.float32(11.9177), np.float32(11.9416),
                        np.float32(11.9550), np.float32(11.9621), np.float32(11.9789), np.float32(11.9886),
                        np.float32(11.9939), np.float32(12.0), np.float32(0.0)]
    for I in range(51, 250):
        object.EnergyLevels[I] = 0.0

    cdef int NL
    for NL in range(object.N_Inelastic):
        object.PenningFraction[0][NL] = 0.0
        # PENN_InelasticG TRANSFER DISTANCE MICRONS
        object.PenningFraction[1][NL] = 1.0
        # PENN_InelasticG TRANSFER TIME PICOSECONDS
        object.PenningFraction[2][NL] = 1.0

    for NL in range(object.N_Inelastic):
        for I in range(4000):
            if object.EG[I] > object.EnergyLevels[NL]:
                IOFFN[NL] = I
                break
    object.EnergySteps = 4000



    for I in range(object.EnergySteps):
        EN = object.EG[I]
        if EN > object.EnergyLevels[0]:
            GAMMA1 = (ElectronMass2 + 2.0 * EN) / ElectronMass2
            GAMMA2 = GAMMA1 * GAMMA1
            BETA = sqrt(1.0 - 1.0 / GAMMA2)
            BETA2 = BETA * BETA

        if EN <= XEN[1]:
            ElasticCrossSectionA = 122.e-16
            QMOM = 122.e-16
        else:
            ElasticCrossSectionA = GasUtil.QLSCALE(EN, NEL, YEL, XEL)
            QMOM = GasUtil.QLSCALE(EN, NDATA, YMOM, XEN)

        TEMP = GasUtil.CALPQ3(EN, NEPSI, YEPS, XEPS)

        PQ = [0.5, 0.5 + (ElasticCrossSectionA - QMOM) / ElasticCrossSectionA, 1 - TEMP]

        object.PEElasticCrossSection[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = ElasticCrossSectionA

        if object.WhichAngularModel == 0:
            object.Q[1][I] = QMOM
        #  IONISATION CHARGE STATE =1

        object.IonizationCrossSection[0][I] = 0.0
        object.PEIonizationCrossSection[0][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[0][I] = 0
        if EN >= object.IonizationEnergy[0]:
            object.IonizationCrossSection[0][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationG, YIN1, XION, BETA2, <float> (0.8061), CONST, object.DEN[I],
                                                 C, AM2)

            # USE ANISOTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
            # ENERGIES ABOVE 2 * IONISATION ENERGY
            # ANISOTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFF SET BY
            # IONISATION ENERGY
            if EN > (2 * object.IonizationEnergy[0]):
                object.PEIonizationCrossSection[0][I] = object.PEElasticCrossSection[1][I - IOFFION[0]]

        #  IONISATION CHARGE STATE =2
        object.IonizationCrossSection[1][I] = 0.0
        object.PEIonizationCrossSection[1][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[1][I] = 0
        if EN >= object.IonizationEnergy[1]:
            object.IonizationCrossSection[1][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization2, YIN2, XIN2, BETA2, <float> (0.1133), CONST, object.DEN[I],
                                                 C, AM2)

            # USE ANISOTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
            # ENERGIES ABOVE 2 * IONISATION ENERGY
            # ANISOTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFF SET BY
            # IONISATION ENERGY
            if EN > (2 * object.IonizationEnergy[1]):
                object.PEIonizationCrossSection[1][I] = object.PEElasticCrossSection[1][I - IOFFION[1]]

        #  IONISATION CHARGE STATE =3
        object.IonizationCrossSection[2][I] = 0.0
        object.PEIonizationCrossSection[2][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[2][I] = 0
        if EN >= object.IonizationEnergy[2]:
            object.IonizationCrossSection[2][I] = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization3, YIN3, XIN3, BETA2, <float> (0.05496), CONST, object.DEN[I],
                                                 C, AM2)
            # USE ANISOTROPIC SCATTERING FOR PRIMARY IONISATION ELECTRON FOR
            # ENERGIES ABOVE 2 * IONISATION ENERGY
            # ANISOTROPIC ANGULAR DISTRIBUTION SAME AS ELASTIC AT ENERGY OFF SET BY
            # IONISATION ENERGY
            if EN > (2 * object.IonizationEnergy[2]):
                object.PEIonizationCrossSection[2][I] = object.PEElasticCrossSection[2][I - IOFFION[2]]

        Q456 = 0.0
        if EN > <float>(106.35) :
            TEMP = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization4, YIN4, XIN4, BETA2, <float>(0.03629), CONST, object.DEN[I], C, AM2)
            if EN <= XIN4[N_Ionization4 - 1]:
                Q456 = TEMP * 4.0 / 3.0
            else:
                Q456 = TEMP
        if EN > <float>(160.45) and EN <= XIN4[N_Ionization4 - 1] :
            TEMP = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization5, YIN5, XIN5, BETA2, <float>(0.03629), CONST, object.DEN[I], C, AM2)
            if EN <= XIN5[N_Ionization5 - 1]:
                Q456 += (TEMP * 5.0 / 3.0)
            else:
                Q456 = TEMP
        if EN > <float>(227.2) and EN <= XIN5[N_Ionization5 - 1] and EN <= XIN4[N_Ionization4 - 1]:
            TEMP = GasUtil.CALIonizationCrossSectionX(EN, N_Ionization6, YIN6, XIN6, BETA2, <float>(0.03629), CONST, object.DEN[I], C, AM2)
            if EN <= XIN6[N_Ionization6 - 1]:
                Q456 += (TEMP * 6.0 / 3.0)
            else:
                Q456 = TEMP


        object.IonizationCrossSection[2][I] += Q456

        #M5-SHELL IONISATION
        object.IonizationCrossSection[3][I] = 0.0
        object.PEIonizationCrossSection[3][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[3][I] = 0
        if EN >= object.IonizationEnergy[3]:
            object.IonizationCrossSection[3][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationM5, YM5S, XM5S)
            object.PEIonizationCrossSection[3][I] = object.PEElasticCrossSection[1][I - IOFFION[3]]

        #M4-SHELL IONISATION
        object.IonizationCrossSection[4][I] = 0.0
        object.PEIonizationCrossSection[4][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[4][I] = 0
        if EN >= object.IonizationEnergy[4]:
            object.IonizationCrossSection[4][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationM4, YM4S, XM4S)
            object.PEIonizationCrossSection[4][I] = object.PEElasticCrossSection[1][I - IOFFION[4]]

        #M3-SHELL IONISATION
        object.IonizationCrossSection[5][I] = 0.0
        object.PEIonizationCrossSection[5][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[5][I] = 0
        if EN >= object.IonizationEnergy[5]:
            object.IonizationCrossSection[5][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationM3, YM3S, XM3S)
            object.PEIonizationCrossSection[5][I] = object.PEElasticCrossSection[1][I - IOFFION[5]]

        #M2-SHELL IONISATION
        object.IonizationCrossSection[6][I] = 0.0
        object.PEIonizationCrossSection[6][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[6][I] = 0
        if EN >= object.IonizationEnergy[6]:
            object.IonizationCrossSection[6][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationM2, YM2S, XM2S)
            object.PEIonizationCrossSection[6][I] = object.PEElasticCrossSection[1][I - IOFFION[6]]

        #M1-SHELL IONISATION
        object.IonizationCrossSection[7][I] = 0.0
        object.PEIonizationCrossSection[7][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[7][I] = 0
        if EN >= object.IonizationEnergy[7]:
            object.IonizationCrossSection[7][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationM1, YM1S, XM1S)
            object.PEIonizationCrossSection[7][I] = object.PEElasticCrossSection[1][I - IOFFION[7]]

        #L3-SHELL IONISATION
        object.IonizationCrossSection[8][I] = 0.0
        object.PEIonizationCrossSection[8][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[8][I] = 0
        if EN >= object.IonizationEnergy[8]:
            object.IonizationCrossSection[8][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationL3, YL3S, XL3S)
            object.PEIonizationCrossSection[8][I] = object.PEElasticCrossSection[1][I - IOFFION[8]]

        #L2-SHELL IONISATION
        object.IonizationCrossSection[9][I] = 0.0
        object.PEIonizationCrossSection[9][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[9][I] = 0
        if EN >= object.IonizationEnergy[9]:
            object.IonizationCrossSection[9][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationL2, YL2S, XL2S)
            object.PEIonizationCrossSection[9][I] = object.PEElasticCrossSection[1][I - IOFFION[9]]

        #L1-SHELL IONISATION
        object.IonizationCrossSection[10][I] = 0.0
        object.PEIonizationCrossSection[10][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[10][I] = 0
        if EN >= object.IonizationEnergy[10]:
            object.IonizationCrossSection[10][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationL1, YL1S, XL1S)
            object.PEIonizationCrossSection[10][I] = object.PEElasticCrossSection[1][I - IOFFION[10]]

        #K-SHELL IONISATION
        object.IonizationCrossSection[11][I] = 0.0
        object.PEIonizationCrossSection[11][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEIonizationCrossSection[11][I] = 0
        if EN >= object.IonizationEnergy[11]:
            object.IonizationCrossSection[11][I] = GasUtil.CALIonizationCrossSectionREG(EN, N_IonizationK, YKSH, XKSH)
            object.PEIonizationCrossSection[11][I] = object.PEElasticCrossSection[1][I - IOFFION[11]]

        # ATTACHMENT
        object.Q[3][I] = 0.0
        object.AttachmentCrossSection[3][I] = object.Q[3][I]

        # COUNTING IONISATION
        object.Q[4][I] = 0.0
        object.PEElasticCrossSection[4][I] = 0.5
        if object.WhichAngularModel == 2:
            object.PEElasticCrossSection[4][I] = 0.0
        if EN > object.IonizationEnergy[0]:
            object.Q[4][I] = GasUtil.CALIonizationCrossSectionX(EN, N_IonizationG, YINC, XION, BETA2, 1.0, CONST, object.DEN[I], C, AM2)

        # CORRECTION TO CHARGE STATE 1 2 3+4+5+6 X-SECTION FOR K L AND M SHELLS
        # CORRECTION GIVES TOTAL IONISATION EQUAL TO OSCILLATOR Sum
        QTEMP = 0.0
        for J in range(3, 12):
            QTEMP += object.IonizationCrossSection[J][I]

        if object.Q[4][I] == 0.0:
            QCORR = 1.0
        else:
            QCORR = (object.Q[4][I] - QTEMP) / object.Q[4][I]
        object.IonizationCrossSection[0][I] *= QCORR
        object.IonizationCrossSection[1][I] *= QCORR
        object.IonizationCrossSection[2][I] *= QCORR

        object.Q[5][I] = 0.0

        for NL in range(object.N_Inelastic + 1):
            object.InelasticCrossSectionPerGas[NL][I] = 0.0
            object.PEInelasticCrossSectionPerGas[NL][I] = 0.5
            if object.WhichAngularModel == 2:
                object.PEInelasticCrossSectionPerGas[NL][I] = 0.0

        #1S5
        if EN > object.EnergyLevels[0]:
            object.InelasticCrossSectionPerGas[0][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N1S5, Y1S5, X1S5, 3)
        if EN > (2 * object.EnergyLevels[0]):
            object.PEInelasticCrossSectionPerGas[0][I] = object.PEElasticCrossSection[1][I - IOFFN[0]]

        #1S4 F=0.260
        if EN > object.EnergyLevels[1]:
            if EN <= X1S4[N1S4-1]:
                object.InelasticCrossSectionPerGas[1][I] = GasUtil.CALInelasticCrossSectionPerGas(EN, N1S4, Y1S4, X1S4)
            else:
                object.InelasticCrossSectionPerGas[1][I] = <float> (0.260) / (object.EnergyLevels[1] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[1])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[1] + object.E[2])
            if EN > (2 * object.EnergyLevels[1]):
                object.PEInelasticCrossSectionPerGas[1][I] = object.PEElasticCrossSection[1][I - IOFFN[1]]

        #1S3
        if EN > object.EnergyLevels[2]:
            object.InelasticCrossSectionPerGas[2][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N1S3, Y1S3, X1S3, 3)
        if EN > (2 * object.EnergyLevels[2]):
            object.PEInelasticCrossSectionPerGas[2][I] = object.PEElasticCrossSection[1][I - IOFFN[2]]

        #1S2 F=0.183
        if EN > object.EnergyLevels[3]:
            if EN <= X1S2[N1S2-1]:
                object.InelasticCrossSectionPerGas[3][I] = GasUtil.CALInelasticCrossSectionPerGas(EN, N1S2, Y1S2, X1S2)
            else:
                object.InelasticCrossSectionPerGas[3][I] = <float> (0.183) / (object.EnergyLevels[3] * BETA2) * (
                        log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[3])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[3] + object.E[2])
            if EN > (2 * object.EnergyLevels[1]):
                object.PEInelasticCrossSectionPerGas[3][I] = object.PEElasticCrossSection[1][I - IOFFN[3]]

        #P STATES
        #2P10
        if EN > object.EnergyLevels[4]:
            object.InelasticCrossSectionPerGas[4][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P10, Y2P10, X2P10, 3)
            if EN > (2 * object.EnergyLevels[4]):
                object.PEInelasticCrossSectionPerGas[4][I] = object.PEElasticCrossSection[1][I - IOFFN[4]]

        #2P9
        if EN > object.EnergyLevels[5]:
            object.InelasticCrossSectionPerGas[5][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P9, Y2P9, X2P9, 1)
            if EN > (2 * object.EnergyLevels[5]):
                object.PEInelasticCrossSectionPerGas[5][I] = object.PEElasticCrossSection[1][I - IOFFN[5]]

        #2P8
        if EN > object.EnergyLevels[6]:
            object.InelasticCrossSectionPerGas[6][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P8, Y2P8, X2P8, 3)
            if EN > (2 * object.EnergyLevels[6]):
                object.PEInelasticCrossSectionPerGas[6][I] = object.PEElasticCrossSection[1][I - IOFFN[6]]

        #2P7
        if EN > object.EnergyLevels[7]:
            object.InelasticCrossSectionPerGas[7][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P7, Y2P7, X2P7, 2)
            if EN > (2 * object.EnergyLevels[7]):
                object.PEInelasticCrossSectionPerGas[7][I] = object.PEElasticCrossSection[1][I - IOFFN[7]]

        #2P6
        if EN > object.EnergyLevels[8]:
            object.InelasticCrossSectionPerGas[8][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P6, Y2P6, X2P6, 1)
            if EN > (2 * object.EnergyLevels[8]):
                object.PEInelasticCrossSectionPerGas[8][I] = object.PEElasticCrossSection[1][I - IOFFN[8]]

        #3D6
        if EN > object.EnergyLevels[9]:
            object.InelasticCrossSectionPerGas[9][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N3D6, Y3D6, X3D6, 1.5)
            if EN > (2 * object.EnergyLevels[9]):
                object.PEInelasticCrossSectionPerGas[9][I] = object.PEElasticCrossSection[1][I - IOFFN[9]]

        #3D5 F=0.0100
        if EN > object.EnergyLevels[10]:
            object.InelasticCrossSectionPerGas[10][I] = 0.01 / (object.EnergyLevels[10] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[10])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[10] + object.E[2])
            if EN > (2 * object.EnergyLevels[10]):
                object.PEInelasticCrossSectionPerGas[10][I] = object.PEElasticCrossSection[1][I - IOFFN[10]]
        #2P5
        if EN > object.EnergyLevels[11]:
            object.InelasticCrossSectionPerGas[11][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P5, Y2P5, X2P5, 1)
            if EN > (2 * object.EnergyLevels[11]):
                object.PEInelasticCrossSectionPerGas[11][I] = object.PEElasticCrossSection[1][I - IOFFN[11]]

        #3D4!
        if EN > object.EnergyLevels[12]:
            object.InelasticCrossSectionPerGas[12][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N3D4P, Y3D4P, X3D4P, 1.5)
            if EN > (2 * object.EnergyLevels[12]):
                object.PEInelasticCrossSectionPerGas[12][I] = object.PEElasticCrossSection[1][I - IOFFN[12]]

        #3D3
        if EN > object.EnergyLevels[13]:
            object.InelasticCrossSectionPerGas[13][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N3D3, Y3D3, X3D3, 1.5)
            if EN > (2 * object.EnergyLevels[13]):
                object.PEInelasticCrossSectionPerGas[13][I] = object.PEElasticCrossSection[1][I - IOFFN[13]]

        #3D4
        if EN > object.EnergyLevels[14]:
            object.InelasticCrossSectionPerGas[14][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N3D4, Y3D4, X3D4, 2)
            if EN > (2 * object.EnergyLevels[14]):
                object.PEInelasticCrossSectionPerGas[14][I] = object.PEElasticCrossSection[1][I - IOFFN[14]]

        #3D1!!
        if EN > object.EnergyLevels[15]:
            object.InelasticCrossSectionPerGas[15][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N3D1PP, Y3D1PP, X3D1PP, 3)
            if EN > (2 * object.EnergyLevels[15]):
                object.PEInelasticCrossSectionPerGas[15][I] = object.PEElasticCrossSection[1][I - IOFFN[15]]

        #3D1!
        if EN > object.EnergyLevels[16]:
            object.InelasticCrossSectionPerGas[16][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N3D1P, Y3D1P, X3D1P, 1)
            if EN > (2 * object.EnergyLevels[16]):
                object.PEInelasticCrossSectionPerGas[16][I] = object.PEElasticCrossSection[1][I - IOFFN[16]]

        #3D2 F=0.379
        if EN > object.EnergyLevels[17]:
            object.InelasticCrossSectionPerGas[17][I] = 0.379 / (object.EnergyLevels[17] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[17])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[17] + object.E[2])
            if EN > (2 * object.EnergyLevels[17]):
                object.PEInelasticCrossSectionPerGas[17][I] = object.PEElasticCrossSection[1][I - IOFFN[17]]

        #2S5
        if EN > object.EnergyLevels[18]:
            object.InelasticCrossSectionPerGas[18][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2S5, Y2S5, X2S5, 3)
            if EN > (2 * object.EnergyLevels[18]):
                object.PEInelasticCrossSectionPerGas[18][I] = object.PEElasticCrossSection[1][I - IOFFN[18]]

        #2S4 J=1 F=0.086
        if EN > object.EnergyLevels[19]:
            object.InelasticCrossSectionPerGas[19][I] = 0.086 / (object.EnergyLevels[19] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[19])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[19] + object.E[2])
            if EN > (2 * object.EnergyLevels[19]):
                object.PEInelasticCrossSectionPerGas[19][I] = object.PEElasticCrossSection[1][I - IOFFN[19]]

        #Sum 3P10+3P9+3P8+3P7+3P6+3P5
        if EN > object.EnergyLevels[20]:
            object.InelasticCrossSectionPerGas[20][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N3PSum, Y3P105, X3P105, 1)
            if EN > (2 * object.EnergyLevels[20]):
                object.PEInelasticCrossSectionPerGas[20][I] = object.PEElasticCrossSection[1][I - IOFFN[20]]

        #2P4
        if EN > object.EnergyLevels[21]:
            object.InelasticCrossSectionPerGas[21][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P4, Y2P4, X2P4, 2)
            if EN > (2 * object.EnergyLevels[21]):
                object.PEInelasticCrossSectionPerGas[21][I] = object.PEElasticCrossSection[1][I - IOFFN[21]]

        #Sum 4D6+4D3+4D4P+4D4+4D1PP+4D1P
        if EN > object.EnergyLevels[22]:
            object.InelasticCrossSectionPerGas[22][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N4DSum, Y4DSum, X4DSum, 3)
            if EN > (2 * object.EnergyLevels[22]):
                object.PEInelasticCrossSectionPerGas[22][I] = object.PEElasticCrossSection[1][I - IOFFN[22]]

        # 4D5 J=1 F=0.0010
        if EN > object.EnergyLevels[23]:
            object.InelasticCrossSectionPerGas[23][I] = 0.001 / (object.EnergyLevels[23] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[23])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[23] + object.E[2])
            if EN > (2 * object.EnergyLevels[23]):
                object.PEInelasticCrossSectionPerGas[23][I] = object.PEElasticCrossSection[1][I - IOFFN[23]]

        #2P3
        if EN > object.EnergyLevels[24]:
            object.InelasticCrossSectionPerGas[24][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P3, Y2P3, X2P3, 1)

        #2P2
        if EN > object.EnergyLevels[25]:
            object.InelasticCrossSectionPerGas[25][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P2, Y2P2, X2P2, 2)

        #2P1
        if EN > object.EnergyLevels[26]:
            object.InelasticCrossSectionPerGas[26][I] = GasUtil.CALInelasticCrossSectionPerGasP(EN, N2P1, Y2P1, X2P1, 1)

        # 4D2 J=1 F=0.0835
        if EN > object.EnergyLevels[27]:
            object.InelasticCrossSectionPerGas[27][I] = 0.0835 / (object.EnergyLevels[27] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[27])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[27] + object.E[2])

            if EN > (2 * object.EnergyLevels[27]):
                object.PEInelasticCrossSectionPerGas[27][I] = object.PEElasticCrossSection[1][I - IOFFN[27]]

        # 3S4 J=1 F=0.0225
        if EN > object.EnergyLevels[28]:
            object.InelasticCrossSectionPerGas[28][I] = 0.0225 / (object.EnergyLevels[28] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[28])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[28] + object.E[2])
            if EN > (2 * object.EnergyLevels[28]):
                object.PEInelasticCrossSectionPerGas[28][I] = object.PEElasticCrossSection[1][I - IOFFN[28]]

        # 5D5 J=1 F=0.0227
        if EN > object.EnergyLevels[29]:
            object.InelasticCrossSectionPerGas[29][I] = 0.0227 / (object.EnergyLevels[29] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[29])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[29] + object.E[2])
            if EN > (2 * object.EnergyLevels[29]):
                object.PEInelasticCrossSectionPerGas[29][I] = object.PEElasticCrossSection[1][I - IOFFN[29]]

        # 5D2 J=1 F=0.002
        if EN > object.EnergyLevels[30]:
            object.InelasticCrossSectionPerGas[30][I] = 0.002 / (object.EnergyLevels[30] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[30])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[30] + object.E[2])
            if EN > (2 * object.EnergyLevels[30]):
                object.PEInelasticCrossSectionPerGas[30][I] = object.PEElasticCrossSection[1][I - IOFFN[30]]

        # 4S4 J=1 F=0.0005
        if EN > object.EnergyLevels[31]:
            object.InelasticCrossSectionPerGas[31][I] = 0.0005 / (object.EnergyLevels[31] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[31])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[31] + object.E[2])
            if EN > (2 * object.EnergyLevels[31]):
                object.PEInelasticCrossSectionPerGas[31][I] = object.PEElasticCrossSection[1][I - IOFFN[31]]

        # 3S1! J=1 F=0.1910
        if EN > object.EnergyLevels[32]:
            object.InelasticCrossSectionPerGas[32][I] = 0.191 / (object.EnergyLevels[32] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[32])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[32] + object.E[2])
            if EN > (2 * object.EnergyLevels[32]):
                object.PEInelasticCrossSectionPerGas[32][I] = object.PEElasticCrossSection[1][I - IOFFN[32]]

        # 6D5 J=1 F=0.0088
        if EN > object.EnergyLevels[33]:
            object.InelasticCrossSectionPerGas[33][I] = 0.0088 / (object.EnergyLevels[33] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[33])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[33] + object.E[2])
            if EN > (2 * object.EnergyLevels[33]):
                object.PEInelasticCrossSectionPerGas[33][I] = object.PEElasticCrossSection[1][I - IOFFN[33]]

        # 6D2 J=1 F=0.0967
        if EN > object.EnergyLevels[34]:
            object.InelasticCrossSectionPerGas[34][I] = 0.0967 / (object.EnergyLevels[34] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[34])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[34] + object.E[2])
            if EN > (2 * object.EnergyLevels[34]):
                object.PEInelasticCrossSectionPerGas[34][I] = object.PEElasticCrossSection[1][I - IOFFN[34]]

        # 5S4 J=1 F=0.0967
        if EN > object.EnergyLevels[35]:
            object.InelasticCrossSectionPerGas[35][I] = 0.0288 / (object.EnergyLevels[35] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[35])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[35] + object.E[2])
            if EN > (2 * object.EnergyLevels[35]):
                object.PEInelasticCrossSectionPerGas[35][I] = object.PEElasticCrossSection[1][I - IOFFN[35]]

        # 7D5 J=1 F=0.0042
        if EN > object.EnergyLevels[36]:
            object.InelasticCrossSectionPerGas[36][I] = 0.0042 / (object.EnergyLevels[36] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[36])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[36] + object.E[2])
            if EN > (2 * object.EnergyLevels[36]):
                object.PEInelasticCrossSectionPerGas[36][I] = object.PEElasticCrossSection[1][I - IOFFN[36]]

        # 7D2 J=1 F=0.0625
        if EN > object.EnergyLevels[37]:
            object.InelasticCrossSectionPerGas[37][I] = 0.0625 / (object.EnergyLevels[37] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[37])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[37] + object.E[2])
            if EN > (2 * object.EnergyLevels[37]):
                object.PEInelasticCrossSectionPerGas[37][I] = object.PEElasticCrossSection[1][I - IOFFN[37]]

        # 6S4 J=1 F=0.0025
        if EN > object.EnergyLevels[38]:
            object.InelasticCrossSectionPerGas[38][I] = 0.0025 / (object.EnergyLevels[38] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[37])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[37] + object.E[2])
            if EN > (2 * object.EnergyLevels[37]):
                object.PEInelasticCrossSectionPerGas[37][I] = object.PEElasticCrossSection[1][I - IOFFN[37]]

        # 2S2 J=1 F=0.029
        if EN > object.EnergyLevels[39]:
            object.InelasticCrossSectionPerGas[39][I] = 0.029 / (object.EnergyLevels[39] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[39])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[39] + object.E[2])
            if EN > (2 * object.EnergyLevels[39]):
                object.PEInelasticCrossSectionPerGas[39][I] = object.PEElasticCrossSection[1][I - IOFFN[39]]

        # 8D5 J=1 F=0.0035
        if EN > object.EnergyLevels[40]:
            object.InelasticCrossSectionPerGas[40][I] = 0.0035 / (object.EnergyLevels[40] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[40])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[40] + object.E[2])
            if EN > (2 * object.EnergyLevels[40]):
                object.PEInelasticCrossSectionPerGas[40][I] = object.PEElasticCrossSection[1][I - IOFFN[40]]

        # 8D2 J=1 F=0.0386
        if EN > object.EnergyLevels[41]:
            object.InelasticCrossSectionPerGas[41][I] = 0.0386 / (object.EnergyLevels[41] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[41])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[41] + object.E[2])
            if EN > (2 * object.EnergyLevels[41]):
                object.PEInelasticCrossSectionPerGas[41][I] = object.PEElasticCrossSection[1][I - IOFFN[41]]

        # 7S4 J=1 F=0.005
        if EN > object.EnergyLevels[42]:
            object.InelasticCrossSectionPerGas[42][I] = 0.005 / (object.EnergyLevels[42] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[42])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[42] + object.E[2])
            if EN > (2 * object.EnergyLevels[42]):
                object.PEInelasticCrossSectionPerGas[42][I] = object.PEElasticCrossSection[1][I - IOFFN[42]]

        # 9D5 J=1 F=0.0005
        if EN > object.EnergyLevels[43]:
            object.InelasticCrossSectionPerGas[43][I] = 0.0005 / (object.EnergyLevels[43] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[43])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[43] + object.E[2])
            if EN > (2 * object.EnergyLevels[43]):
                object.PEInelasticCrossSectionPerGas[43][I] = object.PEElasticCrossSection[1][I - IOFFN[43]]

        # 9D2 J=1 F=0.0250
        if EN > object.EnergyLevels[44]:
            object.InelasticCrossSectionPerGas[44][I] = 0.025 / (object.EnergyLevels[44] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[44])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[44] + object.E[2])
            if EN > (2 * object.EnergyLevels[44]):
                object.PEInelasticCrossSectionPerGas[44][I] = object.PEElasticCrossSection[1][I - IOFFN[44]]

        # 8S4 J=1 F=0.0023
        if EN > object.EnergyLevels[45]:
            object.InelasticCrossSectionPerGas[45][I] = 0.0023 / (object.EnergyLevels[45] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[45])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[45] + object.E[2])
            if EN > (2 * object.EnergyLevels[45]):
                object.PEInelasticCrossSectionPerGas[45][I] = object.PEElasticCrossSection[1][I - IOFFN[45]]

        #10D5 J=1 F=0.0005
        if EN > object.EnergyLevels[46]:
            object.InelasticCrossSectionPerGas[46][I] = 0.0005 / (object.EnergyLevels[46] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[46])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[46] + object.E[2])
            if EN > (2 * object.EnergyLevels[46]):
                object.PEInelasticCrossSectionPerGas[46][I] = object.PEElasticCrossSection[1][I - IOFFN[46]]

        #10D2 J=1 F=0.0164
        if EN > object.EnergyLevels[47]:
            object.InelasticCrossSectionPerGas[47][I] = 0.0164 / (object.EnergyLevels[47] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[47])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[47] + object.E[2])
            if EN > (2 * object.EnergyLevels[47]):
                object.PEInelasticCrossSectionPerGas[47][I] = object.PEElasticCrossSection[1][I - IOFFN[47]]

        #9S4 J=1 F=0.0014
        if EN > object.EnergyLevels[48]:
            object.InelasticCrossSectionPerGas[48][I] = 0.0014 / (object.EnergyLevels[48] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[48])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[48] + object.E[2])
            if EN > (2 * object.EnergyLevels[48]):
                object.PEInelasticCrossSectionPerGas[48][I] = object.PEElasticCrossSection[1][I - IOFFN[48]]

        #HIGH J=1 F=0.0831
        if EN > object.EnergyLevels[49]:
            object.InelasticCrossSectionPerGas[49][I] = 0.0831 / (object.EnergyLevels[49] * BETA2) * (
                    log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * object.EnergyLevels[49])) - BETA2 - object.DEN[
                I] / 2.0) * BBCONST * EN / (EN + object.EnergyLevels[49] + object.E[2])
            if EN > (2 * object.EnergyLevels[49]):
                object.PEInelasticCrossSectionPerGas[49][I] = object.PEElasticCrossSection[1][I - IOFFN[49]]
        for J in range(26, 50):
            if object.InelasticCrossSectionPerGas[J][I] < 0.0:
                object.InelasticCrossSectionPerGas[J][I] = 0.0
        QEXC = 0.0
        for J in range(object.N_Inelastic):
            QEXC += object.InelasticCrossSectionPerGas[J][I]
        for J in range(12):
            object.Q[0][I] += object.IonizationCrossSection[J][I]
        object.Q[0][I] += QEXC

    for J in range(object.N_Inelastic):
        if object.FinalEnergy <= object.EnergyLevels[J]:
            object.N_Inelastic = J
            break
