import h5py
import numpy as np
import math


def Gas1(object):
    gd = h5py.File(r"gases.hdf5",'r')

    EIN = gd[r'gas1/EIN']
    EOBY = [1 for x in range(12)]
    # EIN=[0 for x in range(250)]#<=== input to this function
    EMASS = 9.10938291e-31
    AMU = 1.660538921e-27
    E = [0.0, 1.0, 15.9, 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * EMASS / (88.0043 * AMU)
    object.NC0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0]
    object.EC0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 253.0, 625.2]
    WKLM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0026, 0.01]
    object.EFL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 273.0, 668.0]
    object.NG1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 2]
    object.EG1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 253.0, 625.2]
    object.NG2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1]
    object.EG2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0]
    object.EION = [15.7, 21.47, 29.14, 34.5, 34.77, 36.0, 40.0, 41.0, 43.0, 63.0, 285.0, 685.4]
    IOFFION = [0 for x in range(12)]
    IOFFN = [0 for x in range(46)]
    EMASS2 = 1021997.804
    API = math.acos(-1)
    A0 = 0.52917720859e-8
    RY = 13.60569193
    BBCONST = 16.0 * API * A0 * A0 * RY * RY / EMASS2

    # BORN BETHE VALUES FOR IONISATION
    CONST = 1.873884e-20
    AM2 = 9.5
    C = 100.9

    # number of array elements
    NCF3 = 37
    NCF2 = 31
    NCF1 = 28
    NCF32 = 25
    NCF0 = 27
    NC0F = 27
    NCF22 = 25
    NCF = 22
    NCFF = 24
    NCF2F = 25
    NCF3F = 26
    object.NION = 12
    object.NATT = 1
    object.NIN = 46
    object.NNULL = 0
    NASIZE = 4000
    NBREM = 25
    object.NSTEP = 4000

    for i in range(0, 6):
        object.KEL[i] = object.NANISO
    # ASSUME CAPITELLI LONGO TYPE OF ANGULAR DISTRIBUTION FOR
    # ALL VIBRATIONAL LEVELS AND THE SUM OF HIGHER HARMONICS
    for i in range(0, 10):
        object.KIN[i] = 1
    # ANGULAR DISTRIBUTION FOR DISS.EXCITATION IS GIVEN BY OKHRIMOVSKKY
    for i in range(10, object.NIN):
        object.KIN[i] = object.NANISO
    # RATIO OF MOMENTUM TRANSFER TO TOTAL X-SEC FOR RESONANCE
    # PART OF VIBRATIONAL X-SECTIONS
    RAT = 0.75
    NDATA = 163
    NVBV4 = 11
    NVBV1 = 11
    NVBV3 = 11
    NVIB5 = 12
    NVIB6 = 12
    NATT1 = 11
    NTR1 = 12
    NTR2 = 11
    NTR3 = 11
    NKSHC = 81
    NKSHF = 79

    # OPAL BEATY IONISATION ENERGY SPLITTING
    for i in range(0, 10):
        EOBY[i] = 0.58 * object.EION[i]

    EOBY[10] = 210.0
    EOBY[11] = 510.0

    # skipped ISHELL and LEGAS, as they are not used in any calculation

    for j in range(0, NION):
        for i in range(0, NASIZE):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i - 1
                break

    # OFFSET ENERGY FOR DISSOCIATION ANGULAR DISTRIBUTION

    for NL in range(10, 46):
        for i in range(0, NASIZE):
            if object.EG[i] > abs(EIN[NL]):
                IOFFN[NL] = i - 1
                break

    # ENTER PENNING TRANSFER FRACTION FOR EACH LEVEL
    # ONLY DISSOCIATION X-SECTION (LEVEL 11) HAS ENOUGH ENERGY TO GIVE
    # POSSIBLE PENNING TRANSFER

    for L in range(0, 3):
        object.PENFRA.append(np.zeros(46))

    # PENNING TRANSFER FRACTION FOR LEVEL 11
    object.PENFRA[0][45] = 0.0
    # PENNING TRANSFER DISTANCE IN MICRONS
    object.PENFRA[1][45] = 1.0
    # PENNING TRANSFER TIME IN PICOSECONDS
    object.PENFRA[2][45] = 1.0

    # PRINT

    # VIBRATIONAL DEGENERACY
    DEGV4 = 3.0
    DEGV3 = 3.0
    DEGV2 = 2.0
    DEGV1 = 1.0

    # CALC VIB LEVEL POPULATIONS
    APOPV2 = DEGV2 * math.exp(EIN[0] / object.AKT)
    APOPV4 = DEGV4 * math.exp(EIN[2] / object.AKT)
    APOPV1 = DEGV1 * math.exp(EIN[4] / object.AKT)
    APOPV3 = DEGV3 * math.exp(EIN[6] / object.AKT)
    APOPGS = 1.0
    APOPSUM = APOPGS + APOPV2 + APOPV4 + APOPV1 + APOPV3
    APOPGS = 1.0 / APOPSUM
    APOPV2 = APOPV2 / APOPSUM
    APOPV4 = APOPV4 / APOPSUM
    APOPV1 = APOPV1 / APOPSUM
    APOPV3 = APOPV3 / APOPSUM

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

    # EN=-ESTEP/2.0  #ESTEP is function input
    for i in range(0, object.NSTEP):
        EN = object.EG[i]
        # EN=EN+ESTEP
        GAMMA1 = (EMASS2 + 2.0 * EN) / EMASS2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = np.sqrt(1.00 - 1.00 / GAMMA2)
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
        QELA = (A * EN + B) * 1e-16

        A = (YEPS[j] - YEPS[j - 1]) / (XEN[j] - XEN[j - 1])
        B = (XEN[j - 1] * YEPS[j] - XEN[j] * YEPS[j - 1]) / (XEN[j - 1] - XEN[j])
        PQ = [0.5, 0.5 + (QELA - QMOM) / QELA, 1 - (A * EN + B)]
        # ^^^^^^EPS CORRECTED FOR 1-EPS^^^^^^^^
        object.PEQEL[1][i] = PQ[object.NANISO]
        object.Q[1][i] = QELA
        # DISSOCIATIVE IONISATION
        # ION  =  CF3 +
        if object.NANISO == 0:
            object.Q[1][i] = QMOM
        object.QION[0][i] = 0
        object.PEQION[0][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[0][i] = 0

        if EN > object.EION[0]:
            if EN <= XCF3[NCF3 - 1]:  # <<<check if -1 or not
                j = 0
                for j in range(1, NCF3):
                    if EN <= XCF3[j]:
                        break
                A = (YCF3[j] - YCF3[j - 1]) / (XCF3[j] - XCF3[j - 1])
                B = (XCF3[j - 1] * YCF3[j] - XCF3[j] * YCF3[j - 1]) / (XCF3[j - 1] - XCF3[j])
                object.QION[0][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF3([NCF3] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[0][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.7344
            if EN > 2 * object.EION[0]:
                object.PEQION[0][i] = object.PEQEL[1][(i - IOFFION[0])]

        # ION = CF2 +
        object.QION[1][i] = 0.0
        object.PEQION[1][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[1][i] = 0.0

        if EN > object.EION[1]:
            if EN <= XCF2[NCF2 - 1]:
                j = 0
                for j in range(1, NCF2):
                    if EN <= XCF2[j]:
                        break
                A = (YCF2[j] - YCF2[j - 1]) / (XCF2[j] - XCF2[j - 1])
                B = (XCF2[j - 1] * YCF2[j] - XCF2[j] * YCF2[j - 1]) / (XCF2[j - 1] - XCF2[j])
                object.QION[1][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF2[NCF2] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[1][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0534
            if EN > 2 * object.EION[1]:
                object.PEQION[1][i] = object.PEQEL[1][(i - IOFFION[1])]

        #  ION = CF +
        object.QION[2][i] = 0.0
        object.PEQION[2][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[2][i] = 0.0

        if EN > object.EION[2]:
            if EN <= XCF1[NCF1 - 1]:
                j = 0
                for j in range(1, NCF1):
                    if EN <= XCF1[j]:
                        break
                A = (YCF1[j] - YCF1[j - 1]) / (XCF1[j] - XCF1[j - 1])
                B = (XCF1[j - 1] * YCF1[j] - XCF1[j] * YCF1[j - 1]) / (XCF1[j - 1] - XCF1[j])
                object.QION[2][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF1[NCF1] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[2][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0386
            if EN > 2 * object.EION[2]:
                object.PEQION[2][i] = object.PEQEL[1][(i - IOFFION[2])]

        # ION = F +
        object.QION[3][i] = 0.0
        object.PEQION[3][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[3][i] = 0.0

        if EN > object.EION[3]:
            if EN <= XC0F[NC0F - 1]:
                j = 0
                for j in range(1, NC0F):
                    if EN <= XC0F[j]:
                        break
                A = (YC0F[j] - YC0F[j - 1]) / (XC0F[j] - XC0F[j - 1])
                B = (XC0F[j - 1] * YC0F[j] - XC0F[j] * YC0F[j - 1]) / (XC0F[j - 1] - XC0F[j])
                object.QION[3][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XC0F[NC0F] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[3][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0799
            if EN > 2 * object.EION[3]:
                object.PEQION[3][i] = object.PEQEL[1][(i - IOFFION[3])]

        # ION = C +
        object.QION[4][i] = 0.0
        object.PEQION[4][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[4][i] = 0.0

        if EN > object.EION[4]:
            if EN <= XCF0[NCF0 - 1]:
                j = 0
                for j in range(1, NCF0):
                    if EN <= XCF0[j]:
                        break
                A = (YCF0[j] - YCF0[j - 1]) / (XCF0[j] - XCF0[j - 1])
                B = (XCF0[j - 1] * YCF0[j] - XCF0[j] * YCF0[j - 1]) / (XCF0[j - 1] - XCF0[j])
                object.QION[4][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF0[NCF0] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[4][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0422
            if EN > 2 * object.EION[4]:
                object.PEQION[4][i] = object.PEQEL[1][(i - IOFFION[4])]

        # DOUBLE IONS  CF3 +  AND F +
        object.QION[5][i] = 0.0
        object.PEQION[5][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[5][i] = 0.0

        if EN > object.EION[5]:
            if EN <= XCF3F[NCF3F - 1]:
                j = 0
                for j in range(1, NCF3F):
                    if EN <= XCF3F[j]:
                        break
                A = (YCF3F[j] - YCF3F[j - 1]) / (XCF3F[j] - XCF3F[j - 1])
                B = (XCF3F[j - 1] * YCF3F[j] - XCF3F[j] * YCF3F[j - 1]) / (XCF3F[j - 1] - XCF3F[j])
                object.QION[5][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF3F[NCF3F] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[5][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0058
            if EN > 2 * object.EION[5]:
                object.PEQION[5][i] = object.PEQEL[1][(i - IOFFION[5])]
        # DOUBLE IONS  CF2 +  AND F +
        object.QION[6][i] = 0.0
        object.PEQION[6][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[6][i] = 0.0

        if EN > object.EION[6]:
            if EN <= XCF2F[NCF2F - 1]:
                j = 0
                for j in range(1, NCF2F):
                    if EN <= XCF2F[j]:
                        break
                A = (YCF2F[j] - YCF2F[j - 1]) / (XCF2F[j] - XCF2F[j - 1])
                B = (XCF2F[j - 1] * YCF2F[j] - XCF2F[j] * YCF2F[j - 1]) / (XCF2F[j - 1] - XCF2F[j])
                object.QION[6][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF2F[NCF2F] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[6][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0073
            if EN > 2 * object.EION[6]:
                object.PEQION[6][i] = object.PEQEL[1][(i - IOFFION[6])]

        # DOUBLE CHARGED ION  CF3 ++
        object.QION[7][i] = 0.0
        object.PEQION[7][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[7][i] = 0.0

        if EN > object.EION[7]:
            if EN <= XCF32[NCF32 - 1]:
                j = 0
                for j in range(1, NCF32):
                    if EN <= XCF32[j]:
                        break
                A = (YCF32[j] - YCF32[j - 1]) / (XCF32[j] - XCF32[j - 1])
                B = (XCF32[j - 1] * YCF32[j] - XCF32[j] * YCF32[j - 1]) / (XCF32[j - 1] - XCF32[j])
                object.QION[7][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF32[NCF32] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[7][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0031
            if EN > 2 * object.EION[7]:
                object.PEQION[7][i] = object.PEQEL[1][(i - IOFFION[7])]

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
                object.QION[7][i] = object.QION[7][i] + (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF22[NCF22] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[7][i] = object.QION[7][i] + CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0077

        # DOUBLE IONS    CF +  AND F +
        object.QION[8][i] = 0.0
        object.PEQION[8][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[8][i] = 0.0

        if EN > object.EION[8]:
            if EN <= XCFF[NCFF - 1]:
                j = 0
                for j in range(1, NCFF):
                    if EN <= XCFF[j]:
                        break
                A = (YCFF[j] - YCFF[j - 1]) / (XCFF[j] - XCFF[j - 1])
                B = (XCFF[j - 1] * YCFF[j] - XCFF[j] * YCFF[j - 1]) / (XCFF[j - 1] - XCFF[j])
                object.QION[8][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCFF[NCFF] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[8][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0189
            if EN > 2 * object.EION[8]:
                object.PEQION[8][i] = object.PEQEL[1][(i - IOFFION[8])]

        # DOUBLE IONS    C +  AND F +
        object.QION[9][i] = 0.0
        object.PEQION[9][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[9][i] = 0.0

        if EN > object.EION[9]:
            if EN <= XCF[NCF - 1]:
                j = 0
                for j in range(1, NCF):
                    if EN <= XCF[j]:
                        break
                A = (YCF[j] - YCF[j - 1]) / (XCF[j] - XCF[j - 1])
                B = (XCF[j - 1] * YCF[j] - XCF[j] * YCF[j - 1]) / (XCF[j - 1] - XCF[j])
                object.QION[9][i] = (A * EN + B) * 1e-16
            else:
                # USE BORN BETHE X-SECTION ABOVE XCF[NCF] EV
                X2 = 1 / BETA2
                X1 = X2 * np.log(BETA2 / (1 - BETA2)) - 1
                object.QION[9][i] = CONST * (AM2 * (X1 - object.DEN[i] / 2) + C * X2) * 0.0087
            if EN > 2 * object.EION[9]:
                object.PEQION[9][i] = object.PEQEL[1][(i - IOFFION[9])]

        # CARBON K-SHELL IONISATION
        object.QION[10][i] = 0.0
        object.PEQION[10][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[10][i] = 0.0

        if EN > object.EION[10]:
            if EN <= XKSHC[NKSHC - 1]:
                j = 0
                for j in range(1, NKSHC):
                    if EN <= XKSHC[j]:
                        break
                A = (YKSHC[j] - YKSHC[j - 1]) / (XKSHC[j] - XKSHC[j - 1])
                B = (XKSHC[j - 1] * YKSHC[j] - XKSHC[j] * YKSHC[j - 1]) / (XKSHC[j - 1] - XKSHC[j])
                object.QION[10][i] = (A * EN + B) * 1e-16
            if EN > 2 * object.EION[10]:
                object.PEQION[10][i] = object.PEQEL[1][(i - IOFFION[10])]

        # Fluorine K-SHELL IONISATION
        object.QION[11][i] = 0.0
        object.PEQION[11][i] = 0.5

        if object.NANISO == 2:
            object.PEQION[11][i] = 0.0

        if EN > object.EION[11]:
            if EN <= XKSHF[NKSHF - 1]:
                j = 0
                for j in range(1, NKSHF):
                    if EN <= XKSHF[j]:
                        break
                A = (YKSHF[j] - YKSHF[j - 1]) / (XKSHF[j] - XKSHF[j - 1])
                B = (XKSHF[j - 1] * YKSHF[j] - XKSHF[j] * YKSHF[j - 1]) / (XKSHF[j - 1] - XKSHF[j])
                object.QION[11][i] = 4 * (A * EN + B) * 1e-16
            if EN > 2 * object.EION[11]:
                object.PEQION[11][i] = object.PEQEL[1][(i - IOFFION[11])]

        # ATTACHMENT
        j = 0
        object.Q[3][i] = 0.0
        if EN > XATT[0]:
            if EN <= XATT[NATT1 - 1]:
                for j in range(1, NATT1):
                    if EN <= XATT[j]:
                        break
                A = (YATT[j] - YATT[j - 1]) / (XATT[j] - XATT[j - 1])
                B = (XATT[j - 1] * YATT[j] - XATT[j] * YATT[j - 1]) / (XATT[j - 1] - XATT[j])
                object.Q[3][i] = (A * EN + B) * 1e-16
                object.QATT[0][i] = object.Q[3][i]
        object.Q[4][i] = 0.0
        object.Q[5][i] = 0.0

        # SCALE FACTOR FOR VIBRATIONAL DIPOLE V3 ABOVE 0.4EV

        VDSC = 1.0
        if EN > 0.4:
            EPR = EN
            if EN > 5.0:
                EPR = 5.0
            VDSC = (14.4 - EPR) / 14.0
        # SUPERELASTIC OF VIBRATION V2 ISOTROPIC  BELOW 100EV
        object.QIN[0][i] = 0.0
        object.PEQIN[0][i] = 0.5
        if EN > 0.0:
            EFAC = np.sqrt(1.0 - (EIN[0] / EN))
            object.QIN[0][i] = 0.007 * np.log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.QIN[0][i] = object.QIN[0][i] * APOPV2 * 1.0e-16 / DEGV2
            if EN > 100.0:
                object.PEQIN[0][i] = PQ[1]

        # VIBRATION V2 ISOTROPIC BELOW 100EV
        object.QIN[1][i] = 0.0
        object.PEQIN[1][i] = 0.5
        if EN > EIN[1]:
            EFAC = np.sqrt(1.0 - (EIN[1] / EN))
            object.QIN[1][i] = 0.007 * np.log((1.0 + EFAC) / (1.0 - EFAC)) / EN
            object.QIN[1][i] = object.QIN[1][i] * APOPGS * 1.0e-16
            if EN > 100.0:
                object.PEQIN[1][i] = PQ[1]

        # SUPERELASTIC OF VIBRATION V4 ISOTROPIC BELOW 100EV
        object.QIN[2][i] = 0.0
        object.PEQIN[2][i] = 0.5
        if EN > 0.0:
            if EN - EIN[2] <= XVBV4[NVBV4 - 1]:
                j = 0
                for j in range(1, NVBV4):
                    if EN - EIN[2] <= XVBV4[j]:
                        break
                A = (YVBV4[j] - YVBV4[j - 1]) / (XVBV4[j] - XVBV4[j - 1])
                B = (XVBV4[j - 1] * YVBV4[j] - XVBV4[j] * YVBV4[j - 1]) / (XVBV4[j - 1] - XVBV4[j])
                object.QIN[2][i] = (EN - EIN[2]) * (A * (EN - EIN[2]) + B) / EN
            else:
                object.QIN[2][i] = YVBV4[NVBV4 - 1] * (XVBV4[NVBV4 - 1] / (EN * (EN - EIN[2])) ** 2)
            EFAC = np.sqrt(1.0 - (EIN[2] / EN))
            object.QIN[2][i] = object.QIN[2][i] + 0.05 * np.log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.QIN[2][i] = object.QIN[2][i] * APOPV4 * 1.0e-16 / DEGV4
            if EN > 100.0:
                object.PEQIN[2][i] = PQ[1]

        # VIBRATION V4 ANISOTROPIC
        object.QIN[3][i] = 0.0
        object.PEQIN[3][i] = 0.5
        if EN > EIN[3]:
            if EN <= XVBV4[NVBV4 - 1]:
                j = 0
                for j in range(1, NVBV4):
                    if EN <= XVBV4[j]:
                        break
                A = (YVBV4[j] - YVBV4[j - 1]) / (XVBV4[j] - XVBV4[j - 1])
                B = (XVBV4[j - 1] * YVBV4[j] - XVBV4[j] * YVBV4[j - 1]) / (XVBV4[j - 1] - XVBV4[j])
                object.QIN[3][i] = A * EN + B
            else:
                object.QIN[3][i] = YVBV4[NVBV4 - 1] * (XVBV4[NVBV4 - 1] / EN) ** 3
            EFAC = np.sqrt(1.0 - (EIN[3] / EN))
            ADIP = 0.05 * np.log((1.0 + EFAC) / (1.0 - EFAC)) / EN
            ELF = EN - EIN[3]
            FWD = np.log((EN + ELF) / (EN + ELF - 2.0 * np.sqrt(EN * ELF)))
            BCK = np.log((EN + ELF + 2.0 * np.sqrt(EN * ELF)) / (EN + ELF))
            # RATIO OF MT TO TOTAL X-SECT FOR RESONANCE PART = RAT
            XMT = ((1.5 - FWD / (FWD + BCK)) * ADIP + RAT * object.QIN[3][i]) * APOPGS * 1.0e-16
            object.QIN[3][i] = (object.QIN[3][i] + ADIP) * APOPGS * 1.0e-16
            if EN <= 100:
                object.PEQIN[3][i] = 0.5 + (object.QIN[3][i] - XMT) / object.QIN[3][i]
            else:
                object.PEQIN[3][i] = PQ[1]

        # SUPERELASTIC OF VIBRATION V1 ISOTROPIC BELOW 100EV
        object.QIN[4][i] = 0.0
        object.PEQIN[4][i] = 0.5
        if EN > 0.0:
            if EN - EIN[4] <= XVBV1[NVBV1 - 1]:
                j = 0
                for j in range(1, NVBV1):
                    if EN - EIN[4] <= XVBV1[j]:
                        break
                A = (YVBV1[j] - YVBV1[j - 1]) / (XVBV1[j] - XVBV1[j - 1])
                B = (XVBV1[j - 1] * YVBV1[j] - XVBV1[j] * YVBV1[j - 1]) / (XVBV1[j - 1] - XVBV1[j])
                object.QIN[4][i] = (EN - EIN[4]) * (A * (EN - EIN[4]) + B) / EN
            else:
                object.QIN[4][i] = YVBV1[NVBV1 - 1] * (XVBV1[NVBV1 - 1] / (EN * (EN - EIN[4])) ** 2)
            EFAC = np.sqrt(1.0 - (EIN[4] / EN))
            object.QIN[4][i] = object.QIN[4][i] + 0.0224 * np.log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.QIN[4][i] = object.QIN[4][i] * APOPV1 * 1.0e-16 / DEGV1
            if EN > 100.0:
                object.PEQIN[4][i] = PQ[1]

        # VIBRATION V1  ISOTROPIC BELOW 100EV
        object.QIN[5][i] = 0.0
        object.PEQIN[5][i] = 0.5
        if EN > EIN[5]:
            if EN <= XVBV1[NVBV1 - 1]:
                j = 0
                for j in range(1, NVBV1):
                    if EN <= XVBV1[j]:
                        break
                A = (YVBV1[j] - YVBV1[j - 1]) / (XVBV1[j] - XVBV1[j - 1])
                B = (XVBV1[j - 1] * YVBV1[j] - XVBV1[j] * YVBV1[j - 1]) / (XVBV1[j - 1] - XVBV1[j])
                object.QIN[5][i] = A * EN + B
            else:
                object.QIN[5][i] = YVBV1[NVBV1 - 1] * (XVBV1[NVBV1 - 1] / EN) ** 3
            EFAC = np.sqrt(1.0 - (EIN[5] / EN))
            object.QIN[5][i] = object.QIN[5][i] + 0.0224 * np.log((EFAC + 1.0) / (1.0 - EFAC)) / EN
            object.QIN[5][i] = object.QIN[5][i] * APOPGS * 1.0e-16
            if EN > 100.0:
                object.PEQIN[5][i] = PQ[1]

        # SUPERELASTIC OF VIBRATION V3 ISOTROPIC BELOW 100EV
        object.QIN[6][i] = 0.0
        object.PEQIN[6][i] = 0.5
        if EN > 0.0:
            if EN - EIN[6] <= XVBV3[NVBV3 - 1]:
                j = 0
                for j in range(1, NVBV3):
                    if EN - EIN[6] <= XVBV3[j]:
                        break
                A = (YVBV3[j] - YVBV3[j - 1]) / (XVBV3[j] - XVBV3[j - 1])
                B = (XVBV3[j - 1] * YVBV3[j] - XVBV3[j] * YVBV3[j - 1]) / (XVBV3[j - 1] - XVBV3[j])
                object.QIN[6][i] = (EN - EIN[6]) * (A * (EN - EIN[6]) + B) / EN
            else:
                object.QIN[6][i] = YVBV3[NVBV3 - 1] * (XVBV3[NVBV3 - 1] / (EN * (EN - EIN[6])) ** 2)
            EFAC = np.sqrt(1.0 - (EIN[6] / EN))
            object.QIN[6][i] = object.QIN[6][i] + VDSC * 1.610 * np.log((EFAC + 1.0) / (EFAC - 1.0)) / EN
            object.QIN[6][i] = object.QIN[6][i] * APOPV3 * 1.0e-16 / DEGV3
            if EN > 100.0:
                object.PEQIN[6][i] = PQ[1]
        # VIBRATION V4 ANISOTROPIC
        object.QIN[7][i] = 0.0
        object.PEQIN[7][i] = 0.5
        if EN > EIN[7]:
            if EN <= XVBV3[NVBV3 - 1]:
                j = 0
                for j in range(1, NVBV3):
                    if EN <= XVBV3[j]:
                        break
                A = (YVBV3[j] - YVBV3[j - 1]) / (XVBV3[j] - XVBV3[j - 1])
                B = (XVBV3[j - 1] * YVBV3[j] - XVBV3[j] * YVBV3[j - 1]) / (XVBV3[j - 1] - XVBV3[j])
                object.QIN[7][i] = A * EN + B
            else:
                object.QIN[7][i] = YVBV3[NVBV3 - 1] * (XVBV3[NVBV3 - 1] / EN) ** 3
            EFAC = np.sqrt(1.0 - (EIN[7] / EN))
            ADIP = VDSC * 1.610 * np.log((EFAC + 1.0) / (1.0 - EFAC)) / EN
            ELF = EN - EIN[7]
            FWD = np.log((EN + ELF) / (EN + ELF - 2.0 * np.sqrt(EN * ELF)))
            BCK = np.log((EN + ELF + 2.0 * np.sqrt(EN * ELF)) / (EN + ELF))
            # ASSUME RATIO MOM T./ TOT X-SECT FOR RESONANCE PART = RAT
            XMT = ((1.5 - FWD / (FWD + BCK)) * ADIP + RAT * object.QIN[7][i]) * APOPGS * 1.0e-16
            object.QIN[7][i] = (object.QIN[7][i] + ADIP) * APOPGS * 1.0e-16
            if EN <= 100:
                object.PEQIN[7][i] = 0.5 + (object.QIN[7][i] - XMT) / object.QIN[7][i]
            else:
                object.PEQIN[7][i] = PQ[1]

        # VIBRATION HARMONIC 2V3
        object.QIN[8][i] = 0.0
        object.PEQION[8][i] = 0.5
        if EN > EIN[8]:
            if EN <= XVIB5[NVIB5 - 1]:
                j = 0
                for j in range(1, NVIB5):
                    if EN <= XVIB5[j]:
                        break
                A = (YVIB5[j] - YVIB5[j - 1]) / (XVIB5[j] - XVIB5[j - 1])
                B = (XVIB5[j - 1] * YVIB5[j] - XVIB5[j] * YVIB5[j - 1]) / (XVIB5[j - 1] - XVIB5[j])
                object.QIN[8][i] = A * EN + B
            else:
                object.QIN[8][i] = YVIB5[NVIB5 - 1] * (XVIB5[NVIB5 - 1] / EN)
            object.QIN[8][i] = object.QIN[8][i] * APOPGS * 1.0e-16
            if EN <= 100:
                object.PEQIN[8][i] = 0.5 + (1.0 - RAT)
            else:
                object.PEQIN[8][i] = PQ[1]
        # VIBRATION HARMONIC 3V3
        object.QIN[9][i] = 0.0
        object.PEQION[9][i] = 0.5
        if EN > EIN[9]:
            if EN <= XVIB6[NVIB6 - 1]:
                j = 0
                for j in range(1, NVIB6):
                    if EN <= XVIB6[j]:
                        break
                A = (YVIB6[j] - YVIB6[j - 1]) / (XVIB6[j] - XVIB6[j - 1])
                B = (XVIB6[j - 1] * YVIB6[j] - XVIB6[j] * YVIB6[j - 1]) / (XVIB6[j - 1] - XVIB6[j])
                object.QIN[9][i] = A * EN + B
            else:
                object.QIN[9][i] = YVIB6[NVIB6 - 1] * (XVIB6[NVIB6 - 1] / EN)
            object.QIN[9][i] = object.QIN[9][i] * APOPGS * 1.0e-16
            if EN <= 100:
                object.PEQIN[9][i] = 0.5 + (1.0 - RAT)
            else:
                object.PEQIN[9][i] = PQ[1]

        # TRIPLET NEUTRAL DISSOCIATION ELOSS=11.5 EV
        object.QIN[10][i] = 0.0
        object.PEQIN[10][i] = 0.0
        if EN > EIN[10]:
            if EN <= XTR1[NTR1 - 1]:
                j = 0
                for j in range(1, NTR1):
                    if EN <= XTR1[j]:
                        break
                A = (YTR1[j] - YTR1[j - 1]) / (XTR1[j] - XTR1[j - 1])
                B = (XTR1[j - 1] * YTR1[j] - XTR1[j] * YTR1[j - 1]) / (XTR1[j - 1] - XTR1[j])
                object.QIN[10][i] = (A * EN + B) * 1.0e-16
            else:
                object.QIN[10][i] = YTR1[NTR1 - 1] * (XTR1[NTR1 - 1] / EN) ** 2 * 1.0e-16
            if EN > 3 * EIN[10]:
                object.PEQIN[10][i] = object.PEQEL[1][(i - IOFFN[10])]
        # SINGLET NEUTRAL DISSOCIATION  ELOSS=11.63 EV     F=0.0001893
        object.QIN[11][i] = 0.0
        object.PEQIN[11][i] = 0.0
        if EN > EIN[11]:
            object.QIN[11][i] = 0.0001893 / (EIN[11] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0
                                                                                                    * EIN[
                                                                                                        11])) - BETA2 -
                                                                  object.DEN[
                                                                      i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[11] + object.E[2]) * 1.0107
        if object.QIN[11][i] < 0.0:
            object.QIN[11][i] = 0
        if EN > 3 * EIN[11]:
            object.PEQIN[11][i] = object.PEQEL[1][i - IOFFN[11]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=11.88 EV     F=0.001085
        object.QIN[12][i] = 0.0
        object.PEQIN[12][i] = 0.0
        if EN > EIN[12]:
            object.QIN[12][i] = 0.001085 / (EIN[12] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0
                                                                                                   * EIN[12])) - BETA2 -
                                                                 object.DEN[
                                                                     i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[12] + object.E[2]) * 1.0105
        if object.QIN[12][i] < 0.0:
            object.QIN[12][i] = 0
        if EN > 3 * EIN[12]:
            object.PEQIN[12][i] = object.PEQEL[1][i - IOFFN[12]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=11.88 EV     F=0.004807
        object.QIN[13][i] = 0.0
        object.PEQIN[13][i] = 0.0
        if EN > EIN[13]:
            object.QIN[13][i] = 0.004807 / (EIN[13] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                   EIN[13])) - BETA2 -
                                                                 object.DEN[
                                                                     i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[13] + object.E[2]) * 1.0103
        if object.QIN[13][i] < 0.0:
            object.QIN[13][i] = 0
        if EN > 3 * EIN[13]:
            object.PEQIN[13][i] = object.PEQEL[1][i - IOFFN[13]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=12.38 EV     F=0.008819
        object.QIN[14][i] = 0.0
        object.PEQIN[14][i] = 0.0
        if EN > EIN[14]:
            object.QIN[14][i] = 0.008819 / (EIN[14] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                   EIN[14])) - BETA2 -
                                                                 object.DEN[
                                                                     i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[14] + object.E[2]) * 1.0101
        if object.QIN[14][i] < 0.0:
            object.QIN[14][i] = 0
        if EN > 3 * EIN[14]:
            object.PEQIN[14][i] = object.PEQEL[1][i - IOFFN[14]]

        # TRIPLET NEUTRAL DISSOCIATION ELOSS=12.5 EV
        object.QIN[15][i] = 0.0
        object.PEQIN[15][i] = 0.0
        if EN > EIN[15]:
            if EN <= XTR2[NTR2 - 1]:
                j = 0
                for j in range(1, NTR2):
                    if EN <= XTR2[j]:
                        break
                A = (YTR2[j] - YTR2[j - 1]) / (XTR2[j] - XTR2[j - 1])
                B = (XTR2[j - 1] * YTR2[j] - XTR2[j] * YTR2[j - 1]) / (XTR2[j - 1] - XTR2[j])
                object.QIN[15][i] = (A * EN + B) * 1.0e-16
            else:
                object.QIN[15][i] = YTR2[NTR2 - 1] * (XTR2[NTR2 - 1] / EN) ** 2 * 1.0e-16
            if EN > 3 * EIN[15]:
                object.PEQIN[15][i] = object.PEQEL[1][i - IOFFN[15]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=12.63 EV     F=0.008918
        object.QIN[16][i] = 0.0
        object.PEQIN[16][i] = 0.0
        if EN > EIN[16]:
            object.QIN[16][i] = 0.008918 / (EIN[16] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                   EIN[16])) - BETA2 -
                                                                 object.DEN[
                                                                     i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[16] + object.E[2]) * 1.0099
        if object.QIN[16][i] < 0.0:
            object.QIN[16][i] = 0
        if EN > 3 * EIN[16]:
            object.PEQIN[16][i] = object.PEQEL[1][i - IOFFN[16]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=12.88 EV     F=0.008420
        object.QIN[17][i] = 0.0
        object.PEQIN[17][i] = 0.0
        if EN > EIN[17]:
            object.QIN[17][i] = 0.008420 / (EIN[17] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                   EIN[17])) - BETA2 -
                                                                 object.DEN[
                                                                     i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[17] + object.E[2]) * 1.0097
        if object.QIN[17][i] < 0.0:
            object.QIN[17][i] = 0
        if EN > 3 * EIN[17]:
            object.PEQIN[17][i] = object.PEQEL[1][i - IOFFN[17]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=13.13 EV     F=0.02531
        object.QIN[18][i] = 0.0
        object.PEQIN[18][i] = 0.0
        if EN > EIN[18]:
            object.QIN[18][i] = 0.02531 / (EIN[18] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[18])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[18] + object.E[2]) * 1.0095
        if object.QIN[18][i] < 0.0:
            object.QIN[18][i] = 0
        if EN > 3 * EIN[18]:
            object.PEQIN[18][i] = object.PEQEL[1][i - IOFFN[18]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=13.38 EV     F=0.09553
        object.QIN[19][i] = 0.0
        object.PEQIN[19][i] = 0.0
        if EN > EIN[19]:
            object.QIN[19][i] = 0.09553 / (EIN[19] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[19])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[19] + object.E[2]) * 1.0093
        if object.QIN[19][i] < 0.0:
            object.QIN[19][i] = 0
        if EN > 3 * EIN[19]:
            object.PEQIN[19][i] = object.PEQEL[1][i - IOFFN[19]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=13.63 EV     F=0.11193
        object.QIN[20][i] = 0.0
        object.PEQIN[20][i] = 0.0
        if EN > EIN[20]:
            object.QIN[20][i] = 0.11193 / (EIN[20] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[20])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[20] + object.E[2]) * 1.0092
        if object.QIN[20][i] < 0.0:
            object.QIN[20][i] = 0
        if EN > 3 * EIN[20]:
            object.PEQIN[20][i] = object.PEQEL[1][i - IOFFN[20]]

        # SINGLET NEUTRAL DISSOCIATION    ELOSS=13.88 EV     F=0.10103
        object.QIN[21][i] = 0.0
        object.PEQIN[21][i] = 0.0
        if EN > EIN[21]:
            object.QIN[21][i] = 0.10103 / (EIN[21] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[21])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[21] + object.E[2]) * 1.0090
        if object.QIN[21][i] < 0.0:
            object.QIN[21][i] = 0
        if EN > 3 * EIN[21]:
            object.PEQIN[21][i] = object.PEQEL[1][i - IOFFN[21]]

        # TRIPLET NEUTRAL DISSOCIATION ELOSS=14.0 EV
        object.QIN[22][i] = 0.0
        object.PEQIN[22][i] = 0.0
        if EN > EIN[22]:
            if EN <= XTR3[NTR3 - 1]:
                j = 0
                for j in range(1, NTR3):
                    if EN <= XTR3[j]:
                        break
                A = (YTR3[j] - YTR3[j - 1]) / (XTR3[j] - XTR3[j - 1])
                B = (XTR3[j - 1] * YTR3[j] - XTR3[j] * YTR3[j - 1]) / (XTR3[j - 1] - XTR3[j])
                object.QIN[22][i] = (A * EN + B) * 1.0e-16
            else:
                object.QIN[22][i] = YTR3[NTR3 - 1] * (XTR3[NTR3 - 1] / EN) ** 2 * 1.0e-16
            if EN > 3 * EIN[22]:
                object.PEQIN[22][i] = object.PEQEL[1][i - IOFFN[22]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=14.13 EV     F=0.06902
        object.QIN[23][i] = 0.0
        object.PEQIN[23][i] = 0.0
        if EN > EIN[23]:
            object.QIN[23][i] = 0.06902 / (EIN[23] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[23])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[23] + object.E[2]) * 1.0088
        if object.QIN[23][i] < 0.0:
            object.QIN[23][i] = 0
        if EN > 3 * EIN[23]:
            object.PEQIN[23][i] = object.PEQEL[1][i - IOFFN[23]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=14.38 EV     F=0.03968
        object.QIN[24][i] = 0.0
        object.PEQIN[24][i] = 0.0
        if EN > EIN[24]:
            object.QIN[24][i] = 0.03968 / (EIN[24] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[24])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[24] + object.E[2]) * 1.0087
        if object.QIN[24][i] < 0.0:
            object.QIN[24][i] = 0
        if EN > 3 * EIN[24]:
            object.PEQIN[24][i] = object.PEQEL[1][i - IOFFN[24]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=14.63 EV     F=0.02584
        object.QIN[25][i] = 0.0
        object.PEQIN[25][i] = 0.0
        if EN > EIN[25]:
            object.QIN[25][i] = 0.02584 / (EIN[25] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[25])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[25] + object.E[2]) * 1.0085
        if object.QIN[25][i] < 0.0:
            object.QIN[25][i] = 0
        if EN > 3 * EIN[25]:
            object.PEQIN[25][i] = object.PEQEL[1][i - IOFFN[25]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=14.88 EV     F=0.02071
        object.QIN[26][i] = 0.0
        object.PEQIN[26][i] = 0.0
        if EN > EIN[26]:
            object.QIN[26][i] = 0.02071 / (EIN[26] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[26])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[26] + object.E[2]) * 1.0084
        if object.QIN[26][i] < 0.0:
            object.QIN[26][i] = 0
        if EN > 3 * EIN[26]:
            object.PEQIN[26][i] = object.PEQEL[1][i - IOFFN[26]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=15.13 EV     F=0.03122
        object.QIN[27][i] = 0.0
        object.PEQIN[27][i] = 0.0
        if EN > EIN[27]:
            object.QIN[27][i] = 0.03122 / (EIN[27] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[27])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[27] + object.E[2]) * 1.0083
        if object.QIN[27][i] < 0.0:
            object.QIN[27][i] = 0
        if EN > 3 * EIN[27]:
            object.PEQIN[27][i] = object.PEQEL[1][i - IOFFN[27]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=15.38 EV     F=0.05580
        object.QIN[28][i] = 0.0
        object.PEQIN[28][i] = 0.0
        if EN > EIN[28]:
            object.QIN[28][i] = 0.05580 / (EIN[28] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[28])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[28] + object.E[2]) * 1.0081
        if object.QIN[28][i] < 0.0:
            object.QIN[28][i] = 0
        if EN > 3 * EIN[28]:
            object.PEQIN[28][i] = object.PEQEL[1][i - IOFFN[28]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=15.63 EV     F=0.10187
        object.QIN[29][i] = 0.0
        object.PEQIN[29][i] = 0.0
        if EN > EIN[29]:
            object.QIN[29][i] = 0.10187 / (EIN[29] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[29])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[29] + object.E[2]) * 1.0080
        if object.QIN[29][i] < 0.0:
            object.QIN[29][i] = 0
        if EN > 3 * EIN[29]:
            object.PEQIN[29][i] = object.PEQEL[1][i - IOFFN[29]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=15.88 EV     F=0.09427
        object.QIN[30][i] = 0.0
        object.PEQIN[30][i] = 0.0
        if EN > EIN[30]:
            object.QIN[30][i] = 0.09427 / (EIN[30] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[30])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[30] + object.E[2]) * 1.0079
        if object.QIN[30][i] < 0.0:
            object.QIN[30][i] = 0
        if EN > 3 * EIN[30]:
            object.PEQIN[30][i] = object.PEQEL[1][i - IOFFN[30]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=16.13 EV     F=0.05853
        object.QIN[31][i] = 0.0
        object.PEQIN[31][i] = 0.0
        if EN > EIN[31]:
            object.QIN[31][i] = 0.05853 / (EIN[31] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[31])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[31] + object.E[2]) * 1.0077
        if object.QIN[31][i] < 0.0:
            object.QIN[31][i] = 0
        if EN > 3 * EIN[31]:
            object.PEQIN[31][i] = object.PEQEL[1][i - IOFFN[31]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=16.38 EV     F=0.06002
        object.QIN[32][i] = 0.0
        object.PEQIN[32][i] = 0.0
        if EN > EIN[32]:
            object.QIN[32][i] = 0.06002 / (EIN[32] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[32])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[32] + object.E[2]) * 1.0076
        if object.QIN[32][i] < 0.0:
            object.QIN[32][i] = 0
        if EN > 3 * EIN[32]:
            object.PEQIN[32][i] = object.PEQEL[1][i - IOFFN[32]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=16.63 EV     F=0.05647
        object.QIN[33][i] = 0.0
        object.PEQIN[33][i] = 0.0
        if EN > EIN[33]:
            object.QIN[33][i] = 0.05647 / (EIN[33] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[33])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[33] + object.E[2]) * 1.0075
        if object.QIN[33][i] < 0.0:
            object.QIN[33][i] = 0
        if EN > 3 * EIN[33]:
            object.PEQIN[33][i] = object.PEQEL[1][i - IOFFN[33]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=16.88 EV     F=0.04885
        object.QIN[34][i] = 0.0
        object.PEQIN[34][i] = 0.0
        if EN > EIN[34]:
            object.QIN[34][i] = 0.04885 / (EIN[34] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[34])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[34] + object.E[2]) * 1.0074
        if object.QIN[34][i] < 0.0:
            object.QIN[34][i] = 0
        if EN > 3 * EIN[34]:
            object.PEQIN[34][i] = object.PEQEL[1][i - IOFFN[34]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=17.13 EV     F=0.04036
        object.QIN[35][i] = 0.0
        object.PEQIN[35][i] = 0.0
        if EN > EIN[35]:
            object.QIN[35][i] = 0.04036 / (EIN[35] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[35])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[35] + object.E[2]) * 1.0073
        if object.QIN[35][i] < 0.0:
            object.QIN[35][i] = 0
        if EN > 3 * EIN[35]:
            object.PEQIN[35][i] = object.PEQEL[1][i - IOFFN[35]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=17.38 EV     F=0.03298
        object.QIN[36][i] = 0.0
        object.PEQIN[36][i] = 0.0
        if EN > EIN[36]:
            object.QIN[36][i] = 0.03298 / (EIN[36] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[36])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[36] + object.E[2]) * 1.0072
        if object.QIN[36][i] < 0.0:
            object.QIN[36][i] = 0
        if EN > 3 * EIN[36]:
            object.PEQIN[36][i] = object.PEQEL[1][i - IOFFN[36]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=17.63 EV     F=0.02593
        object.QIN[37][i] = 0.0
        object.PEQIN[37][i] = 0.0
        if EN > EIN[37]:
            object.QIN[37][i] = 0.02593 / (EIN[37] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[37])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[37] + object.E[2]) * 1.0071
        if object.QIN[37][i] < 0.0:
            object.QIN[37][i] = 0
        if EN > 3 * EIN[37]:
            object.PEQIN[37][i] = object.PEQEL[1][i - IOFFN[37]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=17.88 EV     F=0.01802
        object.QIN[38][i] = 0.0
        object.PEQIN[38][i] = 0.0
        if EN > EIN[38]:
            object.QIN[38][i] = 0.01802 / (EIN[38] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[38])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[38] + object.E[2]) * 1.0070
        if object.QIN[38][i] < 0.0:
            object.QIN[38][i] = 0
        if EN > 3 * EIN[38]:
            object.PEQIN[38][i] = object.PEQEL[1][i - IOFFN[38]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=18.13 EV     F=0.01287
        object.QIN[39][i] = 0.0
        object.PEQIN[39][i] = 0.0
        if EN > EIN[39]:
            object.QIN[39][i] = 0.01287 / (EIN[39] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[39])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[39] + object.E[2]) * 1.0069
        if object.QIN[39][i] < 0.0:
            object.QIN[39][i] = 0
        if EN > 3 * EIN[39]:
            object.PEQIN[39][i] = object.PEQEL[1][i - IOFFN[39]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=18.38 EV     F=0.00830
        object.QIN[40][i] = 0.0
        object.PEQIN[40][i] = 0.0
        if EN > EIN[40]:
            object.QIN[40][i] = 0.00830 / (EIN[40] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[40])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[40] + object.E[2]) * 1.0068
        if object.QIN[40][i] < 0.0:
            object.QIN[40][i] = 0
        if EN > 3 * EIN[40]:
            object.PEQIN[40][i] = object.PEQEL[1][i - IOFFN[40]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=18.63 EV     F=0.00698
        object.QIN[41][i] = 0.0
        object.PEQIN[41][i] = 0.0
        if EN > EIN[41]:
            object.QIN[41][i] = 0.00698 / (EIN[41] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[41])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[41] + object.E[2]) * 1.0067
        if object.QIN[41][i] < 0.0:
            object.QIN[41][i] = 0
        if EN > 3 * EIN[41]:
            object.PEQIN[41][i] = object.PEQEL[1][i - IOFFN[41]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=18.88 EV     F=0.00581
        object.QIN[42][i] = 0.0
        object.PEQIN[42][i] = 0.0
        if EN > EIN[42]:
            object.QIN[42][i] = 0.00581 / (EIN[42] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[42])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[42] + object.E[2]) * 1.0066
        if object.QIN[42][i] < 0.0:
            object.QIN[42][i] = 0
        if EN > 3 * EIN[42]:
            object.PEQIN[42][i] = object.PEQEL[1][i - IOFFN[42]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=19.13 EV     F=0.00502
        object.QIN[43][i] = 0.0
        object.PEQIN[43][i] = 0.0
        if EN > EIN[43]:
            object.QIN[43][i] = 0.00502 / (EIN[43] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[43])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[43] + object.E[2]) * 1.0065
        if object.QIN[43][i] < 0.0:
            object.QIN[43][i] = 0
        if EN > 3 * EIN[43]:
            object.PEQIN[43][i] = object.PEQEL[1][i - IOFFN[43]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=19.38 EV     F=0.00398
        object.QIN[44][i] = 0.0
        object.PEQIN[44][i] = 0.0
        if EN > EIN[44]:
            object.QIN[44][i] = 0.00398 / (EIN[44] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[44])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[44] + object.E[2]) * 1.0064
        if object.QIN[44][i] < 0.0:
            object.QIN[44][i] = 0
        if EN > 3 * EIN[44]:
            object.PEQIN[44][i] = object.PEQEL[1][i - IOFFN[44]]

        # SINGLET NEUTRAL DISSOCIATION   ELOSS=19.63 EV     F=0.00189
        object.QIN[45][i] = 0.0
        object.PEQIN[45][i] = 0.0
        if EN > EIN[45]:
            # magboltz code is 0.00198 while the pattern should go to 0.00189
            object.QIN[45][i] = 0.00198 / (EIN[45] * BETA2) * (np.log(BETA2 * GAMMA2 * EMASS2 / (4.0 *
                                                                                                  EIN[45])) - BETA2 -
                                                                object.DEN[
                                                                    i] / 2.0) * BBCONST * EN / (
                                         EN + EIN[45] + object.E[2]) * 1.0064
        if object.QIN[45][i] < 0.0:
            object.QIN[45][i] = 0
        if EN > 3 * EIN[45]:
            object.PEQIN[45][i] = object.PEQEL[1][i - IOFFN[45]]

        QIONSUM = 0.0
        for J in range(0, 12):
            QIONSUM = QIONSUM + object.QION[J][i]
        QSNGLSUM = 0.0
        for J in range(10, 46):
            if J != 10 and J != 15 and J != 22:
                QSNGLSUM = QSNGLSUM + object.QIN[J][i]

        QTRIPSUM = object.QIN[10][i] + object.QIN[15][i] + object.QIN[22][i]

        VSUM = 0.0
        for J in range(0, 10):
            VSUM = VSUM + object.QIN[J][i]

        QIONG = QIONSUM
        for J in range(5, 12):
            QIONG = QIONG + object.QION[J][i]

        DISTOT = QSNGLSUM + QTRIPSUM + QIONSUM
        object.Q[0][i] = object.Q[1][i] + object.Q[3][i] + VSUM + DISTOT

    for J in range(10, 46):
        if object.EFINAL <= EIN[J]:
            object.NIN = J - 1
            break
    return object

    gd.close()
    print("CF4")
