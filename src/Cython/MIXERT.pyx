from Magboltz cimport Magboltz
from libc.math cimport sin, cos, acos, asin, log, sqrt
from Gasmix cimport Gasmix
from ANG cimport ANG

import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef MIXERT(Magboltz object):
    """
    This function sets up the given Magboltz object with the right values for the requested gas mixture. It uses the Gasmix object
    to get the momentum cross sections and all the needed values.

    The object parameter is the Magboltz object to be setup.
    """
    cdef double QATT[6][4000]
    cdef Gasmix MIXOBJECT
    cdef int  IE, KGAS, NP, p, sum, J, i, j, KION, JJ, IL, I
    ECHARG = 1.602176565e-19

    object.ESTEP = object.EFINAL / float(object.NSTEP)

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
                           object.EFINAL, object.AKT, object.ARY, object.TEMPC, object.TORR, object.IPEN,object.PIR2)
    MIXOBJECT.Run()

    EMASS = 9.10938291e-31

    for IE in range(4000):
        for KGAS in range(object.NGAS):
            object.FCION[IE] = 0.0
            object.FCATT[IE] = 0.0
            NP = 1

            object.CF[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].Q[1][IE] * object.VANN[KGAS]
            object.PSCT[KGAS][IE][NP - 1] = 0.5
            object.ANGCT[KGAS][IE][NP - 1] = 1
            object.INDEX[KGAS][NP - 1] = 0
            ANGOBJECT = ANG()

            if MIXOBJECT.Gases[KGAS].KEL[KGAS] == 1:
                PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                ANGOBJECT.PSCT1 = PSCT1
                ANGOBJECT.ANGCUT()
                object.ANGCT[KGAS][IE][NP - 1] = ANGOBJECT.ANGC
                object.PSCT[KGAS][IE][NP - 1] = ANGOBJECT.PSCT2
                object.INDEX[KGAS][NP - 1] = 1
            elif MIXOBJECT.Gases[KGAS].KEL[1] == 2:
                object.PSCT[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                object.INDEX[KGAS][NP - 1] = 2

            if IE == 0:
                RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                object.AMGAS[KGAS] = 2 * EMASS / MIXOBJECT.Gases[KGAS].E[1]
                object.RGAS[KGAS][NP - 1] = RGAS
                L = 1
                object.IARRY[KGAS][NP - 1] = L
                object.EIN[KGAS][NP - 1] = 0.0
                object.IPN[KGAS][NP - 1] = 0

                object.PENFRA[KGAS][0][NP - 1] = 0.0
                object.PENFRA[KGAS][1][NP - 1] = 0.0
                object.PENFRA[KGAS][2][NP - 1] = 0.0

            # IONISATION

            if object.EFINAL >= MIXOBJECT.Gases[KGAS].E[2]:
                if MIXOBJECT.Gases[KGAS].NION <= 1:
                    NP += 1
                    object.CF[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].Q[2][IE] * object.VANN[KGAS]
                    object.FCION[IE] = object.FCION[IE] + object.CF[KGAS][IE][NP - 1]
                    object.PSCT[KGAS][IE][NP - 1] = 0.5
                    object.ANGCT[KGAS][IE][NP - 1] = 1.0
                    object.INDEX[KGAS][NP - 1] = 0
                    if MIXOBJECT.Gases[KGAS].KEL[2] == 1:
                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        object.ANGCT[KGAS][IE][NP - 1] = ANGOBJECT.ANGC
                        object.PSCT[KGAS][IE][NP - 1] = ANGOBJECT.PSCT2
                        object.INDEX[KGAS][NP - 1] = 1
                    elif MIXOBJECT.Gases[KGAS].KEL[2] == 2:
                        object.PSCT[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        object.INDEX[KGAS][NP - 1] = 2
                elif MIXOBJECT.Gases[KGAS].NION > 1:
                    for KION in range(MIXOBJECT.Gases[KGAS].NION):
                        NP += 1
                        object.CF[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].QION[KION][IE] * object.VANN[KGAS]
                        object.FCION[IE] = object.FCION[IE] + object.CF[KGAS][IE][NP - 1]
                        object.PSCT[KGAS][IE][NP - 1] = 0.5
                        object.ANGCT[KGAS][IE][NP - 1] = 1.0
                        object.INDEX[KGAS][NP - 1] = 0
                        if MIXOBJECT.Gases[0].KEL[2] == 1:
                            PSCT1 = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            ANGOBJECT.PSCT1 = PSCT1
                            ANGOBJECT.ANGCUT()
                            object.ANGCT[KGAS][IE][NP - 1] = ANGOBJECT.ANGC
                            object.PSCT[KGAS][IE][NP - 1] = ANGOBJECT.PSCT2
                            object.INDEX[KGAS][NP - 1] = 1
                        elif MIXOBJECT.Gases[0].KEL[2] == 2:
                            object.PSCT[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            object.INDEX[KGAS][NP - 1] = 2

                if IE == 0:
                    if MIXOBJECT.Gases[KGAS].NION <= 1:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        object.RGAS[KGAS][NP - 1] = RGAS
                        object.EIN[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].E[2] / RGAS
                        object.WPL[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EB[0]
                        object.NC0[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].NC0[0]
                        object.EC0[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EC0[0]
                        object.NG1[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].NG1[0]
                        object.EG1[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EG1[0]
                        object.NG2[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].NG2[0]
                        object.EG2[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EG2[0]
                        object.WKLM[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].WK[1]
                        object.EFL[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EFL[1]
                        object.IPN[KGAS][NP - 1] = 1
                        L = 2
                        object.IARRY[KGAS][NP - 1] = L
                        object.PENFRA[KGAS][0][NP - 1] = 0.0
                        object.PENFRA[KGAS][1][NP - 1] = 0.0
                        object.PENFRA[KGAS][2][NP - 1] = 0.0
                    elif MIXOBJECT.Gases[KGAS].NION > 1:
                        NP = NP - MIXOBJECT.Gases[KGAS].NION
                        for KION in range(MIXOBJECT.Gases[KGAS].NION):
                            NP = NP + 1
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            object.RGAS[KGAS][NP - 1] = RGAS
                            object.EIN[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EION[KION] / RGAS
                            object.WPL[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EB[KION]
                            object.NC0[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].NC0[KION]
                            object.EC0[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EC0[KION]
                            object.EG2[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EG2[KION]
                            object.NG1[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].NG1[KION]
                            object.EG1[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EG1[KION]
                            object.NG2[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].NG2[KION]
                            object.WKLM[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].WK[KION]
                            object.EFL[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EFL[KION]
                            object.IPN[KGAS][NP - 1] = 1
                            L = 2
                            object.IARRY[KGAS][NP - 1] = L
                            object.PENFRA[KGAS][0][NP - 1] = 0.0
                            object.PENFRA[KGAS][1][NP - 1] = 0.0
                            object.PENFRA[KGAS][2][NP - 1] = 0.0

            if object.EFINAL >= MIXOBJECT.Gases[KGAS].E[3]:
                if MIXOBJECT.Gases[KGAS].NATT <= 1:
                    NP += 1
                    object.CF[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].Q[3][IE] * object.VANN[KGAS]
                    object.FCATT[IE] = object.FCATT[IE] + object.CF[KGAS][IE][NP - 1]
                    object.PSCT[KGAS][IE][NP - 1] = 0.5
                    object.ANGCT[KGAS][IE][NP - 1] = 1.0
                    if IE == 0:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        object.RGAS[KGAS][NP - 1] = RGAS
                        object.EIN[KGAS][NP - 1] = 0.0
                        object.INDEX[KGAS][NP - 1] = 0
                        object.IPN[KGAS][NP - 1] = -1
                        L = 3
                        object.IARRY[KGAS][NP - 1] = L
                        object.PENFRA[KGAS][0][NP - 1] = 0.0
                        object.PENFRA[KGAS][1][NP - 1] = 0.0
                        object.PENFRA[KGAS][2][NP - 1] = 0.0

                elif MIXOBJECT.Gases[KGAS].NATT > 1:
                    for JJ in range(int(MIXOBJECT.Gases[KGAS].NATT)):
                        NP += 1
                        object.CF[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].QATT[JJ][IE] * object.VANN[KGAS]
                        object.FCATT[IE] = object.FCATT[IE] + object.CF[KGAS][IE][NP - 1]
                        object.PSCT[KGAS][IE][NP - 1] = 0.5
                        object.ANGCT[KGAS][IE][NP - 1] = 1.0
                        if IE == 0:
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            object.RGAS[KGAS][NP - 1] = RGAS
                            object.EIN[KGAS][NP - 1] = 0.0
                            object.INDEX[KGAS][NP - 1] = 0
                            object.IPN[KGAS][NP - 1] = -1
                            L = 3
                            object.IARRY[KGAS][NP - 1] = L
                            object.PENFRA[KGAS][0][NP - 1] = 0.0
                            object.PENFRA[KGAS][1][NP - 1] = 0.0
                            object.PENFRA[KGAS][2][NP - 1] = 0.0

            # INELASTIC AND SUPERELASTIC
            if MIXOBJECT.Gases[KGAS].NIN > 0:
                for J in range(int(MIXOBJECT.Gases[KGAS].NIN)):
                    NP = NP + 1
                    object.CF[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].QIN[J][IE] * object.VANN[KGAS]
                    object.PSCT[KGAS][IE][NP - 1] = 0.5
                    object.ANGCT[KGAS][IE][NP - 1] = 1.0
                    object.INDEX[KGAS][NP - 1] = 0
                    if MIXOBJECT.Gases[KGAS].KIN[J] == 1:
                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        object.ANGCT[KGAS][IE][NP - 1] = ANGOBJECT.ANGC
                        object.PSCT[KGAS][IE][NP - 1] = ANGOBJECT.PSCT2
                        object.INDEX[KGAS][NP - 1] = 1
                    elif MIXOBJECT.Gases[KGAS].KIN[J] == 2:
                        object.PSCT[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        object.INDEX[KGAS][NP - 1] = 2
                    if IE == 0:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        object.RGAS[KGAS][NP - 1] = RGAS
                        object.EIN[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EIN[J] / RGAS
                        L = 4
                        if MIXOBJECT.Gases[KGAS].EIN[J] < 0:
                            L = 5
                        object.IPN[KGAS][NP - 1] = 0
                        object.IARRY[KGAS][NP - 1] = L
                        object.PENFRA[KGAS][0][NP - 1] = MIXOBJECT.Gases[KGAS].PENFRA[0][J]
                        object.PENFRA[KGAS][1][NP - 1] = MIXOBJECT.Gases[KGAS].PENFRA[1][J] * 1.0e-16 / sqrt(3)
                        object.PENFRA[KGAS][2][NP - 1] = MIXOBJECT.Gases[KGAS].PENFRA[2][J]

            object.IPLAST[KGAS] = NP
            object.ISIZE[KGAS] = 1
            for I in range(1, 9):
                if object.IPLAST[KGAS] >= 2 ** I:
                    object.ISIZE[KGAS] = 2 ** I
                else:
                    break
            # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
            object.TCF[KGAS][IE] = 0.0
            for p in range(int(object.IPLAST[KGAS])):
                object.TCF[KGAS][IE] = object.TCF[KGAS][IE] + object.CF[KGAS][IE][p]
                if object.CF[KGAS][IE][p] < 0:
                    print "WARNING NEGATIVE COLLISION FREQUENCY"

            for p in range(int(object.IPLAST[KGAS])):
                if object.TCF[KGAS][IE] == 0:
                    object.CF[KGAS][IE][p] = 0
                else:
                    object.CF[KGAS][IE][p] = object.CF[KGAS][IE][p] / object.TCF[KGAS][IE]

            for p in range(1, int(object.IPLAST[KGAS])):
                object.CF[KGAS][IE][p] = object.CF[KGAS][IE][p] + object.CF[KGAS][IE][p - 1]
            object.FCATT[IE] = object.FCATT[IE] * object.EROOT[IE]
            object.FCION[IE] = object.FCION[IE] * object.EROOT[IE]
            object.TCF[KGAS][IE] = object.TCF[KGAS][IE] * object.EROOT[IE]
    # CALCULATION OF NULL COLLISION FREQUENCIES
    for IE in range(4000):
        sum = 0
        for i in range(6):
            object.NPLAST[i] = MIXOBJECT.Gases[i].NNULL
            sum += int(object.NPLAST[i])

            if sum == 0:
                break
        for i in range(6):
            if object.NPLAST[i] > 0:
                for J in range(int(object.NPLAST[i])):
                    object.SCLENUL[i][J] = MIXOBJECT.Gases[i].SCLN[J]
                    object.CFN[i][IE][J] = MIXOBJECT.Gases[i].QNULL[J][IE] * object.VANN[i] * object.SCLENUL[i][J]
            # CALCULATE NULL COLLISION FREQUENCY FOR EACH GAS COMPONENT

        for KGAS in range(object.NGAS):
            object.TCFN[KGAS][IE] = 0.0
            for IL in range(int(object.NPLAST[KGAS])):
                object.TCFN[KGAS][IE] = object.TCFN[KGAS][IE] + object.CFN[KGAS][IE][IL]
                if object.CFN[KGAS][IE][IL] < 0:
                    print "WARNING NEGATIVE NULL COLLISION FREQUENCY"
            for IL in range(int(object.NPLAST[KGAS])):
                if object.TCFN[KGAS][IE] == 0:
                    object.CFN[KGAS][IE][IL] = 0.0
                else:
                    object.CFN[KGAS][IE][IE] = object.CFN[KGAS][IE][IL] / object.TCFN[KGAS][IE]

            for IL in range(1, int(object.NPLAST[KGAS])):
                object.CFN[KGAS][IE][IL] = object.CFN[KGAS][IE][IL] + object.CFN[KGAS][IE][IL - 1]
            object.TCFN[KGAS][IE] = object.TCFN[KGAS][IE] * object.EROOT[IE]
    KELSUM = 0

    for KGAS in range(object.NGAS):
        for J in range(6):
            KELSUM += MIXOBJECT.Gases[KGAS].KEL[J]

    for KGAS in range(object.NGAS):
        for J in range(250):
            KELSUM += MIXOBJECT.Gases[KGAS].KIN[J]

    if KELSUM > 0:
        object.NISO = 1

    # CALCULATE NULL COLLISION FREQUENCIES FOR EACH GAS COMPONENT
    FAKEIN = abs(object.FAKEI) / object.NGAS
    for KGAS in range(object.NGAS):
        object.TCFMAX[KGAS] = 0.0
        for IE in range(4000):
            if object.TCF[KGAS][IE] + object.TCFN[KGAS][IE] + FAKEIN >= object.TCFMAX[KGAS]:
                object.TCFMAX[KGAS] = object.TCF[KGAS][IE] + object.TCFN[KGAS][IE] + FAKEIN
    # CALCULATE EACH GAS CUMLATIVE FRACTION NULL COLLISION FREQUENCIES
    object.TCFMX = 0.0
    for KGAS in range(object.NGAS):
        object.TCFMX = object.TCFMX + object.TCFMAX[KGAS]
    for KGAS in range(object.NGAS):
        object.TCFMXG[KGAS] = object.TCFMAX[KGAS] / object.TCFMX
    for KGAS in range(1, object.NGAS):
        object.TCFMXG[KGAS] = object.TCFMXG[KGAS] + object.TCFMXG[KGAS - 1]

    # CALCULATE MAXWELL BOLTZMAN VELOCITY FACTOR FOR EACH GAS COMPONENT

    for KGAS in range(object.NGAS):
        object.VTMB[KGAS] = sqrt(2.0 * ECHARG * object.AKT / object.AMGAS[KGAS]) * 1e-12

    for I in range(object.NSTEP):
        object.QTOT[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[0][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[0][I] + \
                         object.ANN[2] * MIXOBJECT.Gases[2].Q[0][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[0][I] + \
                         object.ANN[4] * MIXOBJECT.Gases[4].Q[0][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[0][I]
        object.QEL[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[1][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[1][I] + \
                        object.ANN[2] * MIXOBJECT.Gases[2].Q[1][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[1][I] + \
                        object.ANN[4] * MIXOBJECT.Gases[4].Q[1][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[1][I]

        for KGAS in range(6):
            object.QION[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[2][I] * object.ANN[KGAS]
            QATT[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[3][I] * object.ANN[KGAS]
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
    return
