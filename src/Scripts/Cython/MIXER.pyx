from Magboltz cimport Magboltz
from libc.math cimport sin, cos, acos, asin, log, sqrt
from Gasmix cimport Gasmix
from ANG cimport ANG

import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef MIXER(Magboltz object):
    cdef double QATT[6][4000], ECHARG, JHI, JLOW, EHI, F2, BP, ELOW
    cdef Gasmix MIXOBJECT
    cdef int  IE, KGAS, NP, p, sum, J, i, j, KION, JJ, IL, I
    ECHARG = 1.602176565e-19


    object.ESTEP = object.EFINAL / object.NSTEP

    EHALF = object.ESTEP / 2

    object.E[0] = EHALF
    for i in range(1, 4000):
        object.E[i] = EHALF + object.ESTEP * i
        object.EROOT[i] = sqrt(object.E[i])
    object.EROOT[0] = sqrt(EHALF)
    MIXOBJECT = Gasmix()
    MIXOBJECT.InitWithInfo(object.NGASN, object.QIN, object.NIN, object.PENFRA,
                           object.E, object.EROOT, object.QTOT, object.QREL, object.QINEL, object.QEL,
                           object.DENSY, 0, object.NGAS, object.NSTEP, object.NANISO, object.ESTEP,
                           object.EFINAL, object.AKT, object.ARY, object.TEMPC, object.TORR, object.IPEN, object.PIR2)
    MIXOBJECT.Run()

    EMASS = 9.10938291e-31
    for IE in range(4000):
        NP = 0
        for KGAS in range(object.NGAS):
            object.CF1[IE][NP] = MIXOBJECT.Gases[KGAS].Q[1][IE] * object.VANN[KGAS]
            object.PSCT1[IE][NP] = 0.5
            object.ANGCT1[IE][NP] = 1
            object.INDEX1[NP] = 0
            ANGOBJECT = ANG()

            if MIXOBJECT.Gases[KGAS].KEL[1] == 1:
                PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                ANGOBJECT.PSCT1 = PSCT1
                ANGOBJECT.ANGCUT()
                object.ANGCT1[IE][NP] = ANGOBJECT.ANGC
                object.PSCT1[IE][NP] = ANGOBJECT.PSCT2
                object.INDEX1[NP] = 1
            elif MIXOBJECT.Gases[KGAS].KEL[1] == 2:
                object.PSCT1[IE][NP] = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                object.INDEX1[NP] = 2

            if IE == 0:
                RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                object.RGAS1[NP] = RGAS
                L = 1
                object.IARRY1[NP] = L
                object.EIN1[NP] = 0.0
                object.IPN1[NP] = 0

                object.PENFRA1[0][NP] = 0.0
                object.PENFRA1[1][NP] = 0.0
                object.PENFRA1[2][NP] = 0.0
                # IONISATION

            if object.EFINAL >= MIXOBJECT.Gases[KGAS].E[2]:
                if MIXOBJECT.Gases[KGAS].NION <= 1:
                    NP += 1
                    object.CF1[IE][NP] = MIXOBJECT.Gases[KGAS].Q[2][IE] * object.VANN[KGAS]
                    object.FCION[IE] = object.FCION[IE] + object.CF1[IE][NP]
                    object.PSCT1[IE][NP] = 0.5
                    object.ANGCT1[IE][NP] = 1.0
                    object.INDEX1[NP] = 0
                    if MIXOBJECT.Gases[KGAS].KEL[2] == 1:
                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        object.ANGCT1[IE][NP] = ANGOBJECT.ANGC
                        object.PSCT1[IE][NP] = ANGOBJECT.PSCT2
                        object.INDEX1[NP] = 1
                    elif MIXOBJECT.Gases[KGAS].KEL[2] == 2:
                        object.PSCT1[IE][NP] = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        object.INDEX1[NP] = 2
                elif MIXOBJECT.Gases[KGAS].NION > 1:
                    for KION in range(MIXOBJECT.NION[KGAS]):
                        NP += 1
                        object.CF1[IE][NP] = MIXOBJECT.Gases[KGAS].QION[KION][IE] * object.VANN[KGAS]
                        object.FCION[IE] = object.FCION[IE] + object.CF1[IE][NP]
                        object.PSCT1[IE][NP] = 0.5
                        object.ANGCT1[IE][NP] = 1.0
                        object.INDEX1[NP] = 0
                        if MIXOBJECT.Gases[KGAS].KEL[2] == 1:
                            PSCT1 = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            ANGOBJECT.PSCT1 = PSCT1
                            ANGOBJECT.ANGCUT()
                            object.ANGCT1[IE][NP] = ANGOBJECT.ANGC
                            object.PSCT1[IE][NP] = ANGOBJECT.PSCT2
                            object.INDEX1[NP] = 1
                        elif MIXOBJECT.Gases[KGAS].KEL[2] == 2:
                            object.PSCT1[IE][NP] = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            object.INDEX1[NP] = 2

                if IE == 0:
                    if MIXOBJECT.Gases[KGAS].NION <= 1:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        object.RGAS1[NP] = RGAS
                        object.EIN1[NP] = MIXOBJECT.Gases[KGAS].E[2] / RGAS
                        object.WPL1[NP] = MIXOBJECT.Gases[KGAS].EB[0]
                        object.NC01[NP] = MIXOBJECT.Gases[KGAS].NC0[0]
                        object.EC01[NP] = MIXOBJECT.Gases[KGAS].EC0[0]
                        object.NG11[NP] = MIXOBJECT.Gases[KGAS].NG1[0]
                        object.EG11[NP] = MIXOBJECT.Gases[KGAS].EG1[0]
                        object.EG21[NP] = MIXOBJECT.Gases[KGAS].EG2[0]
                        object.NG21[NP] = MIXOBJECT.Gases[KGAS].NG2[0]
                        object.EFL1[NP] = MIXOBJECT.Gases[KGAS].EFL[1]
                        object.WKLM1[NP] = MIXOBJECT.Gases[KGAS].WK[1]
                        object.IPN1[NP] = 1
                        L = 2
                        object.IARRY1[NP] = L
                        object.PENFRA1[0][NP] = 0.0
                        object.PENFRA1[1][NP] = 0.0
                        object.PENFRA1[2][NP] = 0.0
                    elif MIXOBJECT.Gases[KGAS].NION > 1:
                        NP = NP - MIXOBJECT.Gases[KGAS].NION
                        for KION in range(MIXOBJECT.Gases[KGAS].NION):
                            NP = NP + 1
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            object.RGAS1[NP] = RGAS
                            object.EIN1[NP] = MIXOBJECT.Gases[KGAS].EION[KION] / RGAS
                            object.WPL1[NP] = MIXOBJECT.Gases[KGAS].EB[KION]
                            object.NC01[NP] = MIXOBJECT.Gases[KGAS].NC0[KION]
                            object.EC01[NP] = MIXOBJECT.Gases[KGAS].EC0[KION]
                            object.NG11[NP] = MIXOBJECT.Gases[KGAS].NG1[KION]
                            object.EG21[NP] = MIXOBJECT.Gases[KGAS].EG2[KION]
                            object.EFL1[NP] = MIXOBJECT.Gases[KGAS].EFL[KION]
                            object.EG11[NP] = MIXOBJECT.Gases[KGAS].EG1[KION]
                            object.NG21[NP] = MIXOBJECT.Gases[KGAS].NG2[KION]
                            object.WKLM1[NP] = MIXOBJECT.Gases[KGAS].WK[KION]
                            object.IPN1[NP] = 1
                            L = 2
                            object.IARRY1[NP] = L
                            object.PENFRA1[0][NP] = 0.0
                            object.PENFRA1[1][NP] = 0.0
                            object.PENFRA1[2][NP] = 0.0

            if object.EFINAL >= MIXOBJECT.Gases[KGAS].E[3]:
                if MIXOBJECT.Gases[KGAS].NATT <= 1:
                    NP += 1
                    object.CF1[IE][NP] = MIXOBJECT.Gases[KGAS].Q[3][IE] * object.VANN[KGAS]
                    object.FCATT[IE] = object.FCATT[IE] + object.CF1[IE][NP]
                    object.PSCT1[IE][NP] = 0.5
                    object.ANGCT1[IE][NP] = 1.0
                    if IE == 0:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        object.RGAS1[NP] = RGAS
                        object.EIN1[NP] = 0.0
                        object.INDEX1[NP] = 0
                        object.IPN1[NP] = -1
                        L = 3
                        object.IARRY1[NP] = L
                        object.PENFRA1[0][NP] = 0.0
                        object.PENFRA1[1][NP] = 0.0
                        object.PENFRA1[2][NP] = 0.0
                elif MIXOBJECT.Gases[KGAS].NATT > 1:
                    for JJ in range(int(MIXOBJECT.Gases[KGAS].NATT)):
                        NP += 1
                        object.CF1[IE][NP] = MIXOBJECT.Gases[KGAS].QATT[JJ][IE] * object.VANN[KGAS]
                        object.FCATT[IE] = object.FCATT[IE] + object.CF1[IE][NP]
                        object.PSCT1[IE][NP] = 0.5
                        object.ANGCT1[IE][NP] = 1.0
                        if IE == 0:
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            object.RGAS1[NP] = RGAS
                            object.EIN1[NP] = 0.0
                            object.INDEX1[NP] = 0
                            object.IPN1[NP] = -1
                            L = 3
                            object.IARRY1[NP] = L
                            object.PENFRA1[0][NP] = 0.0
                            object.PENFRA1[1][NP] = 0.0
                            object.PENFRA1[2][NP] = 0.0

            # INELASTIC AND SUPERELASTIC
            if MIXOBJECT.Gases[KGAS].NIN > 0:
                for J in range(int(MIXOBJECT.Gases[KGAS].NIN)):
                    NP = NP + 1
                    object.CF1[IE][NP] = MIXOBJECT.Gases[KGAS].QIN[J][IE] * object.VANN[KGAS]
                    object.PSCT1[IE][NP] = 0.5
                    object.ANGCT1[IE][NP] = 1.0
                    object.INDEX1[NP] = 0
                    if MIXOBJECT.Gases[KGAS].KIN[J] == 1:

                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        object.ANGCT1[IE][NP] = ANGOBJECT.ANGC
                        object.PSCT1[IE][NP] = ANGOBJECT.PSCT2
                        object.INDEX1[NP] = 1
                    elif MIXOBJECT.Gases[KGAS].KIN[J] == 2:

                        object.PSCT1[IE][NP] = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        object.INDEX1[NP] = 2
                    if IE == 0:

                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        object.RGAS1[NP] = RGAS
                        object.EIN1[NP] = MIXOBJECT.Gases[KGAS].EIN[J] / RGAS
                        L = 4

                        if MIXOBJECT.Gases[KGAS].EIN[J] < 0:
                            L = 5
                        object.IPN1[NP] = 0
                        object.IARRY1[NP] = L
                        object.PENFRA1[0][NP] = MIXOBJECT.Gases[KGAS].PENFRA[0][J]
                        object.PENFRA1[1][NP] = MIXOBJECT.Gases[KGAS].PENFRA[1][J] * 1.0e-16 / sqrt(3)
                        object.PENFRA1[2][NP] = MIXOBJECT.Gases[KGAS].PENFRA[2][J]

            NP += 1

        object.IPLAST1 = NP+1
        object.ISIZE1 = 1
        for I in range(1, 9):
            if object.IPLAST1 >= 2 ** I:
                object.ISIZE1 = 2 ** I
            else:
                break

        # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
        object.TCF1[IE] = 0.0
        for P in range(int(object.IPLAST1)):
            object.TCF1[IE] = object.TCF1[IE] + object.CF1[IE][P]
            if object.CF1[IE][P] < 0:
                print("WARNING NEGATIVE COLLISION FREQUENCY")

        for P in range(int(object.IPLAST1)):
            if object.TCF1[IE] != 0.0:
                object.CF1[IE][P] /= object.TCF1[IE]
            else:
                object.CF1[IE][P] = 0.0

        for P in range(1, int(object.IPLAST1)):
            object.CF1[IE][P] += object.CF1[IE][P - 1]
        object.FCATT[IE] *= object.EROOT[IE]
        object.FCION[IE] *= object.EROOT[IE]
        object.TCF1[IE] *= object.EROOT[IE]

        NP = 0
        object.NPLAST1 = 0
        NNULLSUM = 0.0
        for I in range(object.NGAS):
            NNULLSUM += MIXOBJECT.Gases[KGAS].NNULL
        if NNULLSUM != 0:
            for I in range(object.NGAS):
                if MIXOBJECT.Gases[KGAS].NNULL > 0:
                    for J in range(MIXOBJECT.Gases[KGAS].NNULL):
                        object.SCLENUL1[NP] = MIXOBJECT.Gases[KGAS].SCLN[J]
                        object.CFN1[IE][NP] = MIXOBJECT.Gases[KGAS].QNULL[J][IE] * object.VANN[KGAS] * \
                                              object.SCLENUL1[NP]
                        NP+=1
            object.NPLAST1 = NP+1
            object.TCFN1[IE] = 0.0
            for P in range(int(object.NPLAST1)):
                object.TCFN1[IE] = object.TCFN1[IE] + object.CFN1[IE][P]
                if object.CFN1[IE][P] < 0:
                    print("WARNING NEGATIVE NULL COLLISION FREQUENCY")

            for P in range(int(object.NPLAST1)):
                if object.TCFN1[IE] != 0.0:
                    object.CFN1[IE][P] /= object.TCFN1[IE]
                else:
                    object.CFN1[IE][P] = 0.0

            for P in range(1, int(object.NPLAST1)):
                object.CFN1[IE][P] += object.CFN1[IE][P - 1]
            object.TCFN1[IE] *= object.EROOT[IE]

    KELSUM = 0

    for KGAS in range(object.NGAS):
        for J in range(6):
            KELSUM += MIXOBJECT.Gases[KGAS].KEL[J]

    for KGAS in range(object.NGAS):
        for J in range(250):
            KELSUM += MIXOBJECT.Gases[KGAS].KIN[J]

    if KELSUM > 0:
        object.NISO = 1

    BP = object.EFIELD ** 2 * object.CONST1
    F2 = object.EFIELD * object.CONST3
    ELOW = object.TMAX * (object.TMAX * BP - F2 * sqrt(0.5 * object.EFINAL)) / object.ESTEP - 1
    ELOW = min(ELOW, object.SMALL)
    EHI = object.TMAX * (object.TMAX * BP + F2 * sqrt(0.5 * object.EFINAL)) / object.ESTEP + 1
    if EHI > 10000:
        EHI = 10000
    for l in range(8):
        I = l + 1
        JLOW = 4000 - 500 * (9 - I) + 1 + int(ELOW)
        JHI = 4000 - 500 * (8 - I) + int(EHI)
        JLOW = max(JLOW, 0)
        JHI = min(JHI, 4000)
        for J in range(int(JLOW-1), int(JHI)):
            if (object.TCF1[J] + object.TCFN1[J] + abs(object.FAKEI)) > object.TCFMAX1[l]:
                object.TCFMAX1[l] = object.TCF1[J] + object.TCFN1[J] + abs(object.FAKEI)
    for I in range(object.NSTEP):
        object.QTOT[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[0][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[0][I] + \
                         object.ANN[2] * MIXOBJECT.Gases[2].Q[0][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[0][I] + \
                         object.ANN[4] * MIXOBJECT.Gases[4].Q[0][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[0][I]
        object.QEL[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[1][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[1][I] + \
                        object.ANN[2] * MIXOBJECT.Gases[2].Q[1][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[1][I] + \
                        object.ANN[4] * MIXOBJECT.Gases[4].Q[1][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[1][I]

        for KGAS in range(object.NGAS):
            object.QION[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[2][I] * object.ANN[KGAS]
            QATT[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[3][I] * object.ANN[KGAS]
            if MIXOBJECT.Gases[KGAS].NION > 1:
                object.QION[KGAS][I] = 0.0
                for KION in range(MIXOBJECT.Gases[KGAS].NION):
                    object.QION[KGAS][I] += MIXOBJECT.Gases[KGAS].QION[KION][I] * object.ANN[KGAS]
        object.QREL[I] = 0.0
        object.QSATT[I] = 0.0
        object.QSUM[I] = 0.0
        for J in range(object.NGAS):
            object.QSUM[I] = object.QSUM[I] + object.QION[J][I] + QATT[J][I]
            object.QSATT[I] = object.QSATT[I] + QATT[J][I]
            object.QREL[I] = object.QREL[I] + object.QION[J][I] - QATT[J][I]
        for KGAS in range(6):
            for J in range(int(MIXOBJECT.Gases[KGAS].NIN)):
                object.QSUM[I] = object.QSUM[I] + MIXOBJECT.Gases[KGAS].QIN[J][I] * object.ANN[KGAS]
