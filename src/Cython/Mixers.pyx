from Magboltz cimport Magboltz
from libc.math cimport sin, cos, acos, asin, log, sqrt
from Gasmix cimport Gasmix
from ANG cimport ANG

import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef Mixer(Magboltz object):
    """
    This function sets up the given Magboltz object with the right values for the requested gas mixture. It uses the Gasmix object
    to get the momentum cross sections and all the needed values.

    The object parameter is the Magboltz object to be setup.
    """
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
    MIXOBJECT.InitWithInfo(object.NumberOfGasesN, object.QIN, object.NIN, object.PENFRA,
                           object.E, object.EROOT, object.QTOT, object.QREL, object.QINEL, object.QEL,
                           object.DENSY, 0, object.NumberOfGases, object.NSTEP, object.NANISO, object.ESTEP,
                           object.EFINAL, object.AKT, object.ARY, object.TEMPC, object.TORR, object.IPEN, object.PIR2)
    MIXOBJECT.Run()

    EMASS = 9.10938291e-31
    for IE in range(4000):
        NP = 0
        for KGAS in range(object.NumberOfGases):
            object.CFNT[IE][NP] = MIXOBJECT.Gases[KGAS].Q[1][IE] * object.VANN[KGAS]
            object.PSCTNT[IE][NP] = 0.5
            object.ANGCTNT[IE][NP] = 1
            object.INDEXNT[NP] = 0
            ANGOBJECT = ANG()

            if MIXOBJECT.Gases[KGAS].KEL[1] == 1:
                PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                ANGOBJECT.PSCT1 = PSCT1
                ANGOBJECT.ANGCUT()
                object.ANGCTNT[IE][NP] = ANGOBJECT.ANGC
                object.PSCTNT[IE][NP] = ANGOBJECT.PSCT2
                object.INDEXNT[NP] = 1
            elif MIXOBJECT.Gases[KGAS].KEL[1] == 2:
                object.PSCTNT[IE][NP] = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                object.INDEXNT[NP] = 2

            if IE == 0:
                RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                object.RGASNT[NP] = RGAS
                L = 1
                object.IARRYNT[NP] = L
                object.EINNT[NP] = 0.0
                object.IPNNT[NP] = 0

                object.PENFRANT[0][NP] = 0.0
                object.PENFRANT[1][NP] = 0.0
                object.PENFRANT[2][NP] = 0.0
                # IONISATION

            if object.EFINAL >= MIXOBJECT.Gases[KGAS].E[2]:
                if MIXOBJECT.Gases[KGAS].NION <= 1:
                    NP += 1
                    object.CFNT[IE][NP] = MIXOBJECT.Gases[KGAS].Q[2][IE] * object.VANN[KGAS]
                    object.FCION[IE] = object.FCION[IE] + object.CFNT[IE][NP]
                    object.PSCTNT[IE][NP] = 0.5
                    object.ANGCTNT[IE][NP] = 1.0
                    object.INDEXNT[NP] = 0
                    if MIXOBJECT.Gases[KGAS].KEL[2] == 1:
                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        object.ANGCTNT[IE][NP] = ANGOBJECT.ANGC
                        object.PSCTNT[IE][NP] = ANGOBJECT.PSCT2
                        object.INDEXNT[NP] = 1
                    elif MIXOBJECT.Gases[KGAS].KEL[2] == 2:
                        object.PSCTNT[IE][NP] = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        object.INDEXNT[NP] = 2
                elif MIXOBJECT.Gases[KGAS].NION > 1:
                    for KION in range(MIXOBJECT.NION[KGAS]):
                        NP += 1
                        object.CFNT[IE][NP] = MIXOBJECT.Gases[KGAS].QION[KION][IE] * object.VANN[KGAS]
                        object.FCION[IE] = object.FCION[IE] + object.CFNT[IE][NP]
                        object.PSCTNT[IE][NP] = 0.5
                        object.ANGCTNT[IE][NP] = 1.0
                        object.INDEXNT[NP] = 0
                        if MIXOBJECT.Gases[KGAS].KEL[2] == 1:
                            PSCT1 = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            ANGOBJECT.PSCT1 = PSCT1
                            ANGOBJECT.ANGCUT()
                            object.ANGCTNT[IE][NP] = ANGOBJECT.ANGC
                            object.PSCTNT[IE][NP] = ANGOBJECT.PSCT2
                            object.INDEXNT[NP] = 1
                        elif MIXOBJECT.Gases[KGAS].KEL[2] == 2:
                            object.PSCTNT[IE][NP] = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            object.INDEXNT[NP] = 2

                if IE == 0:
                    if MIXOBJECT.Gases[KGAS].NION <= 1:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EINNT[NP] = MIXOBJECT.Gases[KGAS].E[2] / RGAS
                        object.WPLNT[NP] = MIXOBJECT.Gases[KGAS].EB[0]
                        object.NC0NT[NP] = MIXOBJECT.Gases[KGAS].NC0[0]
                        object.EC0NT[NP] = MIXOBJECT.Gases[KGAS].EC0[0]
                        object.NG1NT[NP] = MIXOBJECT.Gases[KGAS].NG1[0]
                        object.EG1NT[NP] = MIXOBJECT.Gases[KGAS].EG1[0]
                        object.EG2NT[NP] = MIXOBJECT.Gases[KGAS].EG2[0]
                        object.NG2NT[NP] = MIXOBJECT.Gases[KGAS].NG2[0]
                        object.EFLNT[NP] = MIXOBJECT.Gases[KGAS].EFL[0]
                        object.WKLMNT[NP] = MIXOBJECT.Gases[KGAS].WK[0]
                        object.IPNNT[NP] = 1
                        L = 2
                        object.IARRYNT[NP] = L
                        object.PENFRANT[0][NP] = 0.0
                        object.PENFRANT[1][NP] = 0.0
                        object.PENFRANT[2][NP] = 0.0
                    elif MIXOBJECT.Gases[KGAS].NION > 1:
                        NP = NP - MIXOBJECT.Gases[KGAS].NION
                        for KION in range(MIXOBJECT.Gases[KGAS].NION):
                            NP = NP + 1
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            object.RGASNT[NP] = RGAS
                            object.EINNT[NP] = MIXOBJECT.Gases[KGAS].EION[KION] / RGAS
                            object.WPLNT[NP] = MIXOBJECT.Gases[KGAS].EB[KION]
                            object.NC0NT[NP] = MIXOBJECT.Gases[KGAS].NC0[KION]
                            object.EC0NT[NP] = MIXOBJECT.Gases[KGAS].EC0[KION]
                            object.NG1NT[NP] = MIXOBJECT.Gases[KGAS].NG1[KION]
                            object.EG2NT[NP] = MIXOBJECT.Gases[KGAS].EG2[KION]
                            object.EFLNT[NP] = MIXOBJECT.Gases[KGAS].EFL[KION]
                            object.EG1NT[NP] = MIXOBJECT.Gases[KGAS].EG1[KION]
                            object.NG2NT[NP] = MIXOBJECT.Gases[KGAS].NG2[KION]
                            object.WKLMNT[NP] = MIXOBJECT.Gases[KGAS].WK[KION]
                            object.IPNNT[NP] = 1
                            L = 2
                            object.IARRYNT[NP] = L
                            object.PENFRANT[0][NP] = 0.0
                            object.PENFRANT[1][NP] = 0.0
                            object.PENFRANT[2][NP] = 0.0

            if object.EFINAL >= MIXOBJECT.Gases[KGAS].E[3]:
                if MIXOBJECT.Gases[KGAS].NATT <= 1:
                    NP += 1
                    object.CFNT[IE][NP] = MIXOBJECT.Gases[KGAS].Q[3][IE] * object.VANN[KGAS]
                    object.FCATT[IE] = object.FCATT[IE] + object.CFNT[IE][NP]
                    object.PSCTNT[IE][NP] = 0.5
                    object.ANGCTNT[IE][NP] = 1.0
                    if IE == 0:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EINNT[NP] = 0.0
                        object.INDEXNT[NP] = 0
                        object.IPNNT[NP] = -1
                        L = 3
                        object.IARRYNT[NP] = L
                        object.PENFRANT[0][NP] = 0.0
                        object.PENFRANT[1][NP] = 0.0
                        object.PENFRANT[2][NP] = 0.0
                elif MIXOBJECT.Gases[KGAS].NATT > 1:
                    for JJ in range(int(MIXOBJECT.Gases[KGAS].NATT)):
                        NP += 1
                        object.CFNT[IE][NP] = MIXOBJECT.Gases[KGAS].QATT[JJ][IE] * object.VANN[KGAS]
                        object.FCATT[IE] = object.FCATT[IE] + object.CFNT[IE][NP]
                        object.PSCTNT[IE][NP] = 0.5
                        object.ANGCTNT[IE][NP] = 1.0
                        if IE == 0:
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            object.RGASNT[NP] = RGAS
                            object.EINNT[NP] = 0.0
                            object.INDEXNT[NP] = 0
                            object.IPNNT[NP] = -1
                            L = 3
                            object.IARRYNT[NP] = L
                            object.PENFRANT[0][NP] = 0.0
                            object.PENFRANT[1][NP] = 0.0
                            object.PENFRANT[2][NP] = 0.0

            # INELASTIC AND SUPERELASTIC
            if MIXOBJECT.Gases[KGAS].NIN > 0:
                for J in range(int(MIXOBJECT.Gases[KGAS].NIN)):
                    NP = NP + 1
                    object.CFNT[IE][NP] = MIXOBJECT.Gases[KGAS].QIN[J][IE] * object.VANN[KGAS]
                    object.PSCTNT[IE][NP] = 0.5
                    object.ANGCTNT[IE][NP] = 1.0
                    object.INDEXNT[NP] = 0
                    if MIXOBJECT.Gases[KGAS].KIN[J] == 1:

                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        object.ANGCTNT[IE][NP] = ANGOBJECT.ANGC
                        object.PSCTNT[IE][NP] = ANGOBJECT.PSCT2
                        object.INDEXNT[NP] = 1
                    elif MIXOBJECT.Gases[KGAS].KIN[J] == 2:

                        object.PSCTNT[IE][NP] = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        object.INDEXNT[NP] = 2
                    if IE == 0:

                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EINNT[NP] = MIXOBJECT.Gases[KGAS].EIN[J] / RGAS
                        L = 4

                        if MIXOBJECT.Gases[KGAS].EIN[J] < 0:
                            L = 5
                        object.IPNNT[NP] = 0
                        object.IARRYNT[NP] = L
                        object.PENFRANT[0][NP] = MIXOBJECT.Gases[KGAS].PENFRA[0][J]
                        object.PENFRANT[1][NP] = MIXOBJECT.Gases[KGAS].PENFRA[1][J] * 1.0e-16 / sqrt(3)
                        object.PENFRANT[2][NP] = MIXOBJECT.Gases[KGAS].PENFRA[2][J]

            NP += 1

        object.IPLASTNT = NP
        object.ISIZENT = 1
        for I in range(1, 9):
            if object.IPLASTNT >= 2 ** I:
                object.ISIZENT = 2 ** I
            else:
                break

        # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
        object.TCFNT[IE] = 0.0
        for P in range(int(object.IPLASTNT)):
            object.TCFNT[IE] = object.TCFNT[IE] + object.CFNT[IE][P]
            if object.CFNT[IE][P] < 0:
                print("WARNING NEGATIVE COLLISION FREQUENCY")

        for P in range(int(object.IPLASTNT)):
            if object.TCFNT[IE] != 0.0:
                object.CFNT[IE][P] /= object.TCFNT[IE]
            else:
                object.CFNT[IE][P] = 0.0

        for P in range(1, int(object.IPLASTNT)):
            object.CFNT[IE][P] += object.CFNT[IE][P - 1]
        object.FCATT[IE] *= object.EROOT[IE]
        object.FCION[IE] *= object.EROOT[IE]
        object.TCFNT[IE] *= object.EROOT[IE]

        NP = 0
        object.NPLASTNT = 0
        NNULLSUM = 0.0
        for I in range(object.NumberOfGases):
            NNULLSUM += MIXOBJECT.Gases[KGAS].NNULL
        if NNULLSUM != 0:
            for I in range(object.NumberOfGases):
                if MIXOBJECT.Gases[KGAS].NNULL > 0:
                    for J in range(MIXOBJECT.Gases[KGAS].NNULL):
                        object.SCLENULNT[NP] = MIXOBJECT.Gases[KGAS].SCLN[J]
                        object.CFNNT[IE][NP] = MIXOBJECT.Gases[KGAS].QNULL[J][IE] * object.VANN[KGAS] * \
                                               object.SCLENULNT[NP]
                        NP+=1
            object.NPLASTNT = NP + 1
            object.TCFNNT[IE] = 0.0
            for P in range(int(object.NPLASTNT)):
                object.TCFNNT[IE] = object.TCFNNT[IE] + object.CFNNT[IE][P]
                if object.CFNNT[IE][P] < 0:
                    print("WARNING NEGATIVE NULL COLLISION FREQUENCY")

            for P in range(int(object.NPLASTNT)):
                if object.TCFNNT[IE] != 0.0:
                    object.CFNNT[IE][P] /= object.TCFNNT[IE]
                else:
                    object.CFNNT[IE][P] = 0.0

            for P in range(1, int(object.NPLASTNT)):
                object.CFNNT[IE][P] += object.CFNNT[IE][P - 1]
            object.TCFNNT[IE] *= object.EROOT[IE]

    KELSUM = 0

    for KGAS in range(object.NumberOfGases):
        for J in range(6):
            KELSUM += MIXOBJECT.Gases[KGAS].KEL[J]

    for KGAS in range(object.NumberOfGases):
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
            if (object.TCFNT[J] + object.TCFNNT[J] + abs(object.FAKEI)) > object.TCFMAXNT[l]:
                object.TCFMAXNT[l] = object.TCFNT[J] + object.TCFNNT[J] + abs(object.FAKEI)
    for I in range(object.NSTEP):
        object.QTOT[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[0][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[0][I] + \
                         object.ANN[2] * MIXOBJECT.Gases[2].Q[0][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[0][I] + \
                         object.ANN[4] * MIXOBJECT.Gases[4].Q[0][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[0][I]
        object.QEL[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[1][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[1][I] + \
                        object.ANN[2] * MIXOBJECT.Gases[2].Q[1][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[1][I] + \
                        object.ANN[4] * MIXOBJECT.Gases[4].Q[1][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[1][I]

        for KGAS in range(object.NumberOfGases):
            object.QION[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[2][I] * object.ANN[KGAS]
            QATT[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[3][I] * object.ANN[KGAS]
            if MIXOBJECT.Gases[KGAS].NION > 1:
                object.QION[KGAS][I] = 0.0
                for KION in range(MIXOBJECT.Gases[KGAS].NION):
                    object.QION[KGAS][I] += MIXOBJECT.Gases[KGAS].QION[KION][I] * object.ANN[KGAS]
        object.QREL[I] = 0.0
        object.QSATT[I] = 0.0
        object.QSUM[I] = 0.0
        for J in range(object.NumberOfGases):
            object.QSUM[I] = object.QSUM[I] + object.QION[J][I] + QATT[J][I]
            object.QSATT[I] = object.QSATT[I] + QATT[J][I]
            object.QREL[I] = object.QREL[I] + object.QION[J][I] - QATT[J][I]
        for KGAS in range(6):
            for J in range(int(MIXOBJECT.Gases[KGAS].NIN)):
                object.QSUM[I] = object.QSUM[I] + MIXOBJECT.Gases[KGAS].QIN[J][I] * object.ANN[KGAS]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef MixerT(Magboltz object):
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
    MIXOBJECT.InitWithInfo(object.NumberOfGasesN, object.QIN, object.NIN, object.PENFRA,
                           object.E, object.EROOT, object.QTOT, object.QREL, object.QINEL, object.QEL,
                           object.DENSY, 0, object.NumberOfGases, object.NSTEP, object.NANISO, object.ESTEP,
                           object.EFINAL, object.AKT, object.ARY, object.TEMPC, object.TORR, object.IPEN,object.PIR2)
    MIXOBJECT.Run()

    #-----------------------------------------------------------------
    #     CALCULATION OF COLLISION FREQUENCIES FOR AN ARRAY OF
    #     ELECTRON ENERGIES IN THE RANGE ZERO TO EFINAL
    #
    #     L=1      ELASTIC NTH GAS
    #     L=2      IONISATION NTH GAS
    #     L=3      ATTACHMENT NTH GAS
    #     L=4      INELASTIC NTH GAS
    #     L=5      SUPERELASTIC NTH GAS
    #---------------------------------------------------------------
    EMASS = 9.10938291e-31

    for IE in range(4000):
        for KGAS in range(object.NumberOfGases):
            object.FCION[IE] = 0.0
            object.FCATT[IE] = 0.0
            NP = 1

            object.CF[KGAS][IE][NP - 1] = MIXOBJECT.Gases[KGAS].Q[1][IE] * object.VANN[KGAS]
            object.PSCT[KGAS][IE][NP - 1] = 0.5
            object.ANGCT[KGAS][IE][NP - 1] = 1
            object.INDEX[KGAS][NP - 1] = 0
            ANGOBJECT = ANG()

            if MIXOBJECT.Gases[KGAS].KEL[1] == 1:
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
                        object.WKLM[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].WK[0]
                        object.EFL[KGAS][NP - 1] = MIXOBJECT.Gases[KGAS].EFL[0]
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
            # ATTACHMENT
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
                    print ("WARNING NEGATIVE COLLISION FREQUENCY at gas " +str(object.CF[KGAS][IE][p])+"  "+ str(IE))

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

        for KGAS in range(object.NumberOfGases):
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

    for KGAS in range(object.NumberOfGases):
        for J in range(6):
            KELSUM += MIXOBJECT.Gases[KGAS].KEL[J]

    for KGAS in range(object.NumberOfGases):
        for J in range(250):
            KELSUM += MIXOBJECT.Gases[KGAS].KIN[J]

    if KELSUM > 0:
        object.NISO = 1

    # CALCULATE NULL COLLISION FREQUENCIES FOR EACH GAS COMPONENT
    FAKEIN = abs(object.FAKEI) / object.NumberOfGases
    for KGAS in range(object.NumberOfGases):
        object.TCFMAX[KGAS] = 0.0
        for IE in range(4000):
            if object.TCF[KGAS][IE] + object.TCFN[KGAS][IE] + FAKEIN >= object.TCFMAX[KGAS]:
                object.TCFMAX[KGAS] = object.TCF[KGAS][IE] + object.TCFN[KGAS][IE] + FAKEIN
    # CALCULATE EACH GAS CUMLATIVE FRACTION NULL COLLISION FREQUENCIES
    object.TCFMX = 0.0
    for KGAS in range(object.NumberOfGases):
        object.TCFMX = object.TCFMX + object.TCFMAX[KGAS]
    for KGAS in range(object.NumberOfGases):
        object.TCFMXG[KGAS] = object.TCFMAX[KGAS] / object.TCFMX
    for KGAS in range(1, object.NumberOfGases):
        object.TCFMXG[KGAS] = object.TCFMXG[KGAS] + object.TCFMXG[KGAS - 1]

    # CALCULATE MAXWELL BOLTZMAN VELOCITY FACTOR FOR EACH GAS COMPONENT

    for KGAS in range(object.NumberOfGases):
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
        for J in range(object.NumberOfGases):
            object.QSUM[I] = object.QSUM[I] + object.QION[J][I] + QATT[J][I]
            object.QSATT[I] = object.QSATT[I] + QATT[J][I]
            object.QREL[I] = object.QREL[I] + object.QION[J][I] - QATT[J][I]
        for KGAS in range(6):
            for J in range(int(MIXOBJECT.Gases[KGAS].NIN)):
                object.QSUM[I] = object.QSUM[I] + MIXOBJECT.Gases[KGAS].QIN[J][I] * object.ANN[KGAS]

    return
