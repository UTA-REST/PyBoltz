from Magboltz cimport Magboltz
from libc.math cimport sin, cos, acos, asin, log, sqrt
from Gasmix cimport Gasmix
from MIXERT_obj cimport MIXERT_obj
from libc.string cimport memset
from ANG cimport ANG

import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef MIXERT(Magboltz object):
    cdef double QATT[6][4000]
    cdef int  IE, KGAS, NP, p, sum, J, i, j, KION, JJ, IL, I
    cdef MIXERT_obj MIXERTOBJ = MIXERT_obj()
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
                           object.EFINAL, object.AKT, object.ARY, object.TEMPC, object.TORR, object.IPEN)
    MIXOBJECT.Run()

    for i in range(6):
        for j in range(6):
            MIXERTOBJ.Q[i][j][:] = MIXOBJECT.Gases[i].Q[j][:]
            MIXERTOBJ.PEQEL[i][j][:] = MIXOBJECT.Gases[i].PEQEL[j][:]

        for j in range(250):
            MIXERTOBJ.QIN[i][j][:] = MIXOBJECT.Gases[i].QIN[j][:]
            MIXERTOBJ.PEQIN[i][j][:] = MIXOBJECT.Gases[i].PEQIN[j][:]

        MIXERTOBJ.E[i] = MIXOBJECT.Gases[i].E
        MIXERTOBJ.EIN[i] = MIXOBJECT.Gases[i].EIN
        MIXERTOBJ.KIN[i] = MIXOBJECT.Gases[i].KIN
        for j in range(30):
            MIXERTOBJ.QION[i][j][:] = MIXOBJECT.Gases[i].QION[j][:]
            MIXERTOBJ.PEQION[i][j][:] = MIXOBJECT.Gases[i].PEQION[j][:]
        MIXERTOBJ.EION[i] = MIXOBJECT.Gases[i].EION
        MIXERTOBJ.EB[i] = MIXOBJECT.Gases[i].EB
        MIXERTOBJ.KEL[i] = MIXOBJECT.Gases[i].KEL
        for j in range(3):
            MIXERTOBJ.PENFRA[i][j][:] = MIXOBJECT.Gases[i].PENFRA[j][:]
        MIXERTOBJ.NC0[i] = MIXOBJECT.Gases[i].NC0
        MIXERTOBJ.EC0[i] = MIXOBJECT.Gases[i].EC0
        MIXERTOBJ.WK[i] = MIXOBJECT.Gases[i].WK
        MIXERTOBJ.EFL[i] = MIXOBJECT.Gases[i].EFL
        MIXERTOBJ.NG1[i] = MIXOBJECT.Gases[i].NG1
        MIXERTOBJ.NG2[i] = MIXOBJECT.Gases[i].NG2
        MIXERTOBJ.EG1[i] = MIXOBJECT.Gases[i].EG1
        MIXERTOBJ.EG2[i] = MIXOBJECT.Gases[i].EG2
        for j in range(8):
            MIXERTOBJ.QATTT[i][j][:] = MIXOBJECT.Gases[i].QATT[j][:]
        MIXERTOBJ.NIN[i] = MIXOBJECT.Gases[i].NIN
        MIXERTOBJ.NATT[i] = MIXOBJECT.Gases[i].NATT
        MIXERTOBJ.NNULL[i] = MIXOBJECT.Gases[i].NNULL
        MIXERTOBJ.NION[i] = MIXOBJECT.Gases[i].NION
        MIXERTOBJ.SCLN[i] = MIXOBJECT.Gases[i].SCLN
        for j in range(10):
            MIXERTOBJ.QNULL[i][j][:] = MIXOBJECT.Gases[i].QNULL[j][:]
        print(i)
    EMASS = 9.10938291e-31


    for IE in range(4000):
        for KGAS in range(object.NGAS):
            object.FCION[IE] = 0.0
            object.FCATT[IE] = 0.0
            NP = 1
            if KGAS == 0:
                object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.Q[0][1][IE] * object.VANN[0]
            elif KGAS == 1:
                object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.Q[1][1][IE] * object.VANN[1]
            elif KGAS == 2:
                object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.Q[2][1][IE] * object.VANN[2]
            elif KGAS == 3:
                object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.Q[3][1][IE] * object.VANN[3]
            elif KGAS == 4:
                object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.Q[4][1][IE] * object.VANN[4]
            elif KGAS == 5:
                object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.Q[5][1][IE] * object.VANN[5]
            object.PSCT[KGAS][IE][NP - 1] = 0.5
            object.ANGCT[KGAS][IE][NP - 1] = 1
            object.INDEX[KGAS][NP - 1] = 0
            ANGOBJECT = ANG()

            if MIXERTOBJ.KEL[KGAS][1] == 1:
                PSCT1 = MIXERTOBJ.PEQEL[KGAS][1][IE]
                ANGOBJECT.PSCT1 = PSCT1
                ANGOBJECT.ANGCUT()
                object.ANGCT[KGAS][IE][NP - 1] = ANGOBJECT.ANGC
                object.PSCT[KGAS][IE][NP - 1] = ANGOBJECT.PSCT2
                object.INDEX[KGAS][NP - 1] = 1
            elif MIXERTOBJ.KEL[KGAS][1] == 2:
                object.PSCT[KGAS][IE][NP - 1] = MIXERTOBJ.PEQEL[KGAS][1][IE]
                object.INDEX[KGAS][NP - 1] = 2

            if IE == 0:
                RGAS = 1 + MIXERTOBJ.E[KGAS][1] / 2
                object.AMGAS[KGAS] = 2 * EMASS / MIXERTOBJ.E[KGAS][1]
                object.RGAS[KGAS][NP - 1] = RGAS
                L = 1
                object.IARRY[KGAS][NP - 1] = L
                object.EIN[KGAS][NP - 1] = 0.0
                object.IPN[KGAS][NP - 1] = 0

                object.PENFRA[KGAS][0][NP - 1] = 0.0
                object.PENFRA[KGAS][1][NP - 1] = 0.0
                object.PENFRA[KGAS][2][NP - 1] = 0.0

            # IONISATION

            if object.EFINAL >= MIXERTOBJ.E[KGAS][2]:
                if MIXERTOBJ.NION[KGAS] <= 1:
                    NP += 1
                    object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.Q[KGAS][2][IE] * object.VANN[KGAS]
                    object.FCION[IE] = object.FCION[IE] + object.CF[KGAS][IE][NP - 1]
                    object.PSCT[KGAS][IE][NP - 1] = 0.5
                    object.ANGCT[KGAS][IE][NP - 1] = 1.0
                    object.INDEX[KGAS][NP - 1] = 0
                    if MIXERTOBJ.KEL[KGAS][2] == 1:
                        PSCT1 = MIXERTOBJ.PEQEL[KGAS][2][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        object.ANGCT[KGAS][IE][NP - 1] = ANGOBJECT.ANGC
                        object.PSCT[KGAS][IE][NP - 1] = ANGOBJECT.PSCT2
                        object.INDEX[KGAS][NP - 1] = 1
                    elif MIXERTOBJ.KEL[KGAS][2] == 2:
                        object.PSCT[KGAS][IE][NP - 1] = MIXERTOBJ.PEQEL[KGAS][2][IE]
                        object.INDEX[KGAS][NP - 1] = 2
                elif MIXERTOBJ.NION[KGAS] > 1:
                    for KION in range(MIXERTOBJ.NION[KGAS]):
                        NP += 1
                        object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.QION[KGAS][KION][IE] * object.VANN[KGAS]
                        object.FCION[IE] = object.FCION[IE] + object.CF[KGAS][IE][NP - 1]
                        object.PSCT[KGAS][IE][NP - 1] = 0.5
                        object.ANGCT[KGAS][IE][NP - 1] = 1.0
                        object.INDEX[KGAS][NP - 1] = 0
                        if MIXERTOBJ.KEL[0][2] == 1:
                            PSCT1 = MIXERTOBJ.PEQION[KGAS][KION][IE]
                            ANGOBJECT.PSCT1 = PSCT1
                            ANGOBJECT.ANGCUT()
                            object.ANGCT[KGAS][IE][NP - 1] = ANGOBJECT.ANGC
                            object.PSCT[KGAS][IE][NP - 1] = ANGOBJECT.PSCT2
                            object.INDEX[KGAS][NP - 1] = 1
                        elif MIXERTOBJ.KEL[0][2] == 2:
                            object.PSCT[KGAS][IE][NP - 1] = MIXERTOBJ.PEQION[KGAS][KION][IE]
                            object.INDEX[KGAS][NP - 1] = 2

                if IE == 0:
                    if MIXERTOBJ.NION[KGAS] <= 1:
                        RGAS = 1 + MIXERTOBJ.E[KGAS][1] / 2
                        object.RGAS[KGAS][NP - 1] = RGAS
                        object.EIN[KGAS][NP - 1] = MIXERTOBJ.E[KGAS][2] / RGAS
                        object.WPL[KGAS][NP - 1] = MIXERTOBJ.EB[KGAS][0]
                        object.NC0[KGAS][NP - 1] = MIXERTOBJ.NC0[KGAS][0]
                        object.EC0[KGAS][NP - 1] = MIXERTOBJ.EC0[KGAS][0]
                        object.NG1[KGAS][NP - 1] = MIXERTOBJ.NG1[KGAS][0]
                        object.EG1[KGAS][NP - 1] = MIXERTOBJ.EG1[KGAS][0]
                        object.NG2[KGAS][NP - 1] = MIXERTOBJ.NG2[KGAS][0]
                        object.EG2[KGAS][NP - 1] = MIXERTOBJ.EG2[KGAS][0]
                        object.WKLM[KGAS][NP - 1] = MIXERTOBJ.WK[KGAS][1]
                        object.EFL[KGAS][NP - 1] = MIXERTOBJ.EFL[KGAS][1]
                        object.IPN[KGAS][NP - 1] = 1
                        L = 2
                        object.IARRY[KGAS][NP - 1] = L
                        object.PENFRA[KGAS][0][NP - 1] = 0.0
                        object.PENFRA[KGAS][1][NP - 1] = 0.0
                        object.PENFRA[KGAS][2][NP - 1] = 0.0
                    elif MIXERTOBJ.NION[KGAS] > 1:
                        NP = NP - MIXERTOBJ.NION[KGAS]
                        for KION in range(MIXERTOBJ.NION[KGAS]):
                            NP = NP + 1
                            RGAS = 1 + MIXERTOBJ.E[KGAS][1] / 2
                            object.RGAS[KGAS][NP - 1] = RGAS
                            object.EIN[KGAS][NP - 1] = MIXERTOBJ.EION[KGAS][KION] / RGAS
                            object.WPL[KGAS][NP - 1] = MIXERTOBJ.EB[KGAS][KION]
                            object.NC0[KGAS][NP - 1] = MIXERTOBJ.NC0[KGAS][KION]
                            object.EC0[KGAS][NP - 1] = MIXERTOBJ.EC0[KGAS][KION]
                            object.EG2[KGAS][NP - 1] = MIXERTOBJ.EG2[KGAS][KION]
                            object.NG1[KGAS][NP - 1] = MIXERTOBJ.NG1[KGAS][KION]
                            object.EG1[KGAS][NP - 1] = MIXERTOBJ.EG1[KGAS][KION]
                            object.NG2[KGAS][NP - 1] = MIXERTOBJ.NG2[KGAS][KION]
                            object.WKLM[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].WK[KION]
                            object.EFL[KGAS][NP - 1] = MIXERTOBJ.EFL[KGAS][KION]
                            object.IPN[KGAS][NP - 1] = 1
                            L = 2
                            object.IARRY[KGAS][NP - 1] = L
                            object.PENFRA[KGAS][0][NP - 1] = 0.0
                            object.PENFRA[KGAS][1][NP - 1] = 0.0
                            object.PENFRA[KGAS][2][NP - 1] = 0.0

            if object.EFINAL >= MIXERTOBJ.E[KGAS][3]:
                if MIXERTOBJ.NATT[KGAS] <= 1:
                    NP += 1
                    object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.Q[KGAS][3][IE] * object.VANN[KGAS]
                    object.FCATT[IE] = object.FCATT[IE] + object.CF[KGAS][IE][NP - 1]
                    object.PSCT[KGAS][IE][NP - 1] = 0.5
                    object.ANGCT[KGAS][IE][NP - 1] = 1.0
                    if IE == 0:
                        RGAS = 1 + MIXERTOBJ.E[KGAS][1] / 2
                        object.RGAS[KGAS][NP - 1] = RGAS
                        object.EIN[KGAS][NP - 1] = 0.0
                        object.INDEX[KGAS][NP - 1] = 0
                        object.IPN[KGAS][NP - 1] = -1
                        L = 3
                        object.IARRY[KGAS][NP - 1] = L
                        object.PENFRA[KGAS][0][NP - 1] = 0.0
                        object.PENFRA[KGAS][1][NP - 1] = 0.0
                        object.PENFRA[KGAS][2][NP - 1] = 0.0

                elif MIXERTOBJ.NATT[KGAS] > 1:
                    for JJ in range(int(MIXERTOBJ.NATT[KGAS])):
                        NP += 1
                        object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.QATTT[KGAS][JJ][IE] * object.VANN[KGAS]
                        object.FCATT[IE] = object.FCATT[IE] + object.CF[KGAS][IE][NP - 1]
                        object.PSCT[KGAS][IE][NP - 1] = 0.5
                        object.ANGCT[KGAS][IE][NP - 1] = 1.0
                        if IE == 0:
                            RGAS = 1 + MIXERTOBJ.E[KGAS][1] / 2
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
            if MIXERTOBJ.NIN[KGAS] > 0:
                for J in range(int(MIXERTOBJ.NIN[KGAS])):
                    NP = NP + 1
                    object.CF[KGAS][IE][NP - 1] = MIXERTOBJ.QIN[KGAS][J][IE] * object.VANN[KGAS]
                    object.PSCT[KGAS][IE][NP - 1] = 0.5
                    object.ANGCT[KGAS][IE][NP - 1] = 1.0
                    object.INDEX[KGAS][NP - 1] = 0
                    if MIXERTOBJ.KIN[KGAS][J] == 1:
                        PSCT1 = MIXERTOBJ.PEQIN[KGAS][J][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        object.ANGCT[KGAS][IE][NP - 1] = ANGOBJECT.ANGC
                        object.PSCT[KGAS][IE][NP - 1] = ANGOBJECT.PSCT2
                        object.INDEX[KGAS][NP - 1] = 1
                    elif MIXERTOBJ.KIN[KGAS][J] == 2:
                        object.PSCT[KGAS][IE][NP - 1] = MIXERTOBJ.PEQIN[KGAS][J][IE]
                        object.INDEX[KGAS][NP - 1] = 2
                    if IE == 0:
                        RGAS = 1 + MIXERTOBJ.E[KGAS][1] / 2
                        object.RGAS[KGAS][NP - 1] = RGAS
                        object.EIN[KGAS][NP - 1] = MIXERTOBJ.EIN[KGAS][J] / RGAS
                        L = 4
                        if MIXERTOBJ.EIN[KGAS][J] < 0:
                            L = 5
                        object.IPN[KGAS][NP - 1] = 0
                        object.IARRY[KGAS][NP - 1] = L
                        object.PENFRA[KGAS][0][NP - 1] = MIXERTOBJ.PENFRA[KGAS][0][J]
                        object.PENFRA[KGAS][1][NP - 1] = MIXERTOBJ.PENFRA[KGAS][1][J] * 1.0e-16 / sqrt(3)
                        object.PENFRA[KGAS][2][NP - 1] = MIXERTOBJ.PENFRA[KGAS][2][J]

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
                    print("WARNING NEGATIVE COLLISION FREQUENCY")

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
            object.NPLAST[i] = MIXERTOBJ.NNULL[i]
            sum += int(object.NPLAST[i])

            if sum == 0:
                break
        for i in range(6):
            if object.NPLAST[i] > 0:
                for J in range(int(object.NPLAST[i])):
                    object.SCLENUL[i][J] = MIXERTOBJ.SCLN[i][J]
                    object.CFN[i][IE][J] = MIXERTOBJ.QNULL[i][J][IE] * object.VANN[i] * object.SCLENUL[i][J]
            # CALCULATE NULL COLLISION FREQUENCY FOR EACH GAS COMPONENT

        for KGAS in range(object.NGAS):
            object.TCFN[KGAS][IE] = 0.0
            for IL in range(int(object.NPLAST[KGAS])):
                object.TCFN[KGAS][IE] = object.TCFN[KGAS][IE] + object.CFN[KGAS][IE][IL]
                if object.CFN[KGAS][IE][IL] < 0:
                    print("WARNING NEGATIVE NULL COLLISION FREQUENCY")
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
            KELSUM += MIXERTOBJ.KEL[KGAS][J]

    for KGAS in range(object.NGAS):
        for J in range(250):
            KELSUM += MIXERTOBJ.KIN[KGAS][J]

    if KELSUM > 0:
        object.NISO = 1

    # CALCULATE NULL COLLISION FREQUENCIES FOR EACH GAS COMPONENT
    FAKEIN = abs(object.FAKEI) / object.NGAS
    for KGAS in range(object.NGAS):
        object.TCFMAX[KGAS] = 0.0
        for IE in range(4000):
            if object.TCF[KGAS][IE] + object.TCFN[KGAS][IE] + FAKEIN >= object.TCFMAX[KGAS]:
                object.TCFMAX[KGAS] = object.TCF[KGAS][IE] + object.TCFN[KGAS][IE] + FAKEIN
    print(object.TCFMAX)
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
        object.QTOT[I] = object.ANN[0] * MIXERTOBJ.Q[0][0][I] + object.ANN[1] * MIXERTOBJ.Q[1][0][I] + \
                         object.ANN[2] * MIXERTOBJ.Q[2][0][I] + object.ANN[3] * MIXERTOBJ.Q[3][0][I] + \
                         object.ANN[4] * MIXERTOBJ.Q[4][0][I] + object.ANN[5] * MIXERTOBJ.Q[5][0][I]
        object.QEL[I] = object.ANN[0] * MIXERTOBJ.Q[0][1][I] + object.ANN[1] * MIXERTOBJ.Q[1][1][I] + \
                        object.ANN[2] * MIXERTOBJ.Q[2][1][I] + object.ANN[3] * MIXERTOBJ.Q[3][1][I] + \
                        object.ANN[4] * MIXERTOBJ.Q[4][1][I] + object.ANN[5] * MIXERTOBJ.Q[5][1][I]

        for KGAS in range(6):
            object.QION[KGAS][I] = MIXERTOBJ.Q[KGAS][2][I] * object.ANN[KGAS]
            QATT[KGAS][I] = MIXERTOBJ.Q[KGAS][3][I] * object.ANN[KGAS]
        object.QREL[I] = 0.0
        object.QSATT[I] = 0.0
        object.QSUM[I] = 0.0
        for J in range(object.NGAS):
            object.QSUM[I] = object.QSUM[I] + object.QION[J][I] + QATT[J][I]
            object.QSATT[I] = object.QSATT[I] + QATT[J][I]
            object.QREL[I] = object.QREL[I] + object.QION[J][I] - QATT[J][I]
        for KGAS in range(6):
            for J in range(int(MIXERTOBJ.NIN[KGAS])):
                object.QSUM[I] = object.QSUM[I] + MIXERTOBJ.QIN[KGAS][J][I] * object.ANN[KGAS]
    return
