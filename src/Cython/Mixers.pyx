from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt
from Gasmix cimport Gasmix
from Ang cimport Ang

import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef Mixer(PyBoltz object):
    """
    This function sets up the given PyBoltz object with the right values for the requested gas mixture. It uses the Gasmix object
    to get the momentum cross sections and all the needed values.

    The object parameter is the PyBoltz object to be setup.
    """
    cdef double QATT[6][4000], ECHARG, JHI, JLOW, EHI, F2, BP, ELOW
    cdef Gasmix MIXOBJECT
    cdef int  IE, GasIndex, NP, p, sum, J, i, j, KION, JJ, IL, I
    ECHARG = 1.602176565e-19


    object.ElectronEnergyStep = object.FinalElectronEnergy / object.EnergySteps

    EHALF = object.ElectronEnergyStep / 2

    object.E[0] = EHALF
    for i in range(1, 4000):
        object.E[i] = EHALF + object.ElectronEnergyStep * i
        object.EROOT[i] = sqrt(object.E[i])
    object.EROOT[0] = sqrt(EHALF)
    MIXOBJECT = Gasmix()
    MIXOBJECT.InitWithInfo(object.GasIDs, object.QIN, object.NIN, object.PENFRA,
                           object.E, object.EROOT, object.QTOT, object.QREL, object.QINEL, object.QEL,
                           object.DENSY, 0, object.NumberOfGases, object.EnergySteps, object.WhichAngularModel, object.ElectronEnergyStep,
                           object.FinalElectronEnergy, object.ThermalEnergy, object.RhydbergConst, object.TemperatureCentigrade, object.PressureTorr, object.EnablePenning, object.PIR2)
    MIXOBJECT.Run()

    EMASS = 9.10938291e-31
    for IE in range(4000):
        NP = 0
        for GasIndex in range(object.NumberOfGases):
            object.CFNT[IE][NP] = MIXOBJECT.Gases[GasIndex].Q[1][IE] * object.VANN[GasIndex]
            object.PSCTNT[IE][NP] = 0.5
            object.ANGCTNT[IE][NP] = 1
            object.INDEXNT[NP] = 0
            AngObject = Ang()

            if MIXOBJECT.Gases[GasIndex].KEL[1] == 1:
                PSCT1 = MIXOBJECT.Gases[GasIndex].PEQEL[1][IE]
                AngObject.PSCT1 = PSCT1
                AngObject.AngCut()
                object.ANGCTNT[IE][NP] = AngObject.ANGC
                object.PSCTNT[IE][NP] = AngObject.PSCT2
                object.INDEXNT[NP] = 1
            elif MIXOBJECT.Gases[GasIndex].KEL[1] == 2:
                object.PSCTNT[IE][NP] = MIXOBJECT.Gases[GasIndex].PEQEL[1][IE]
                object.INDEXNT[NP] = 2

            if IE == 0:
                RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                object.RGASNT[NP] = RGAS
                L = 1
                object.IARRYNT[NP] = L
                object.EINNT[NP] = 0.0
                object.IPNNT[NP] = 0

                object.PENFRANT[0][NP] = 0.0
                object.PENFRANT[1][NP] = 0.0
                object.PENFRANT[2][NP] = 0.0
                # IONISATION

            if object.FinalElectronEnergy >= MIXOBJECT.Gases[GasIndex].E[2]:
                if MIXOBJECT.Gases[GasIndex].NION <= 1:
                    NP += 1
                    object.CFNT[IE][NP] = MIXOBJECT.Gases[GasIndex].Q[2][IE] * object.VANN[GasIndex]
                    object.FCION[IE] = object.FCION[IE] + object.CFNT[IE][NP]
                    object.PSCTNT[IE][NP] = 0.5
                    object.ANGCTNT[IE][NP] = 1.0
                    object.INDEXNT[NP] = 0
                    if MIXOBJECT.Gases[GasIndex].KEL[2] == 1:
                        PSCT1 = MIXOBJECT.Gases[GasIndex].PEQEL[2][IE]
                        AngObject.PSCT1 = PSCT1
                        AngObject.AngCut()
                        object.ANGCTNT[IE][NP] = AngObject.ANGC
                        object.PSCTNT[IE][NP] = AngObject.PSCT2
                        object.INDEXNT[NP] = 1
                    elif MIXOBJECT.Gases[GasIndex].KEL[2] == 2:
                        object.PSCTNT[IE][NP] = MIXOBJECT.Gases[GasIndex].PEQEL[2][IE]
                        object.INDEXNT[NP] = 2
                elif MIXOBJECT.Gases[GasIndex].NION > 1:
                    for KION in range(MIXOBJECT.NION[GasIndex]):
                        NP += 1
                        object.CFNT[IE][NP] = MIXOBJECT.Gases[GasIndex].QION[KION][IE] * object.VANN[GasIndex]
                        object.FCION[IE] = object.FCION[IE] + object.CFNT[IE][NP]
                        object.PSCTNT[IE][NP] = 0.5
                        object.ANGCTNT[IE][NP] = 1.0
                        object.INDEXNT[NP] = 0
                        if MIXOBJECT.Gases[GasIndex].KEL[2] == 1:
                            PSCT1 = MIXOBJECT.Gases[GasIndex].PEQION[KION][IE]
                            AngObject.PSCT1 = PSCT1
                            AngObject.AngCut()
                            object.ANGCTNT[IE][NP] = AngObject.ANGC
                            object.PSCTNT[IE][NP] = AngObject.PSCT2
                            object.INDEXNT[NP] = 1
                        elif MIXOBJECT.Gases[GasIndex].KEL[2] == 2:
                            object.PSCTNT[IE][NP] = MIXOBJECT.Gases[GasIndex].PEQION[KION][IE]
                            object.INDEXNT[NP] = 2

                if IE == 0:
                    if MIXOBJECT.Gases[GasIndex].NION <= 1:
                        RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EINNT[NP] = MIXOBJECT.Gases[GasIndex].E[2] / RGAS
                        object.WPLNT[NP] = MIXOBJECT.Gases[GasIndex].EB[0]
                        object.NC0NT[NP] = MIXOBJECT.Gases[GasIndex].NC0[0]
                        object.EC0NT[NP] = MIXOBJECT.Gases[GasIndex].EC0[0]
                        object.NG1NT[NP] = MIXOBJECT.Gases[GasIndex].NG1[0]
                        object.EG1NT[NP] = MIXOBJECT.Gases[GasIndex].EG1[0]
                        object.EG2NT[NP] = MIXOBJECT.Gases[GasIndex].EG2[0]
                        object.NG2NT[NP] = MIXOBJECT.Gases[GasIndex].NG2[0]
                        object.EFLNT[NP] = MIXOBJECT.Gases[GasIndex].EFL[0]
                        object.WKLMNT[NP] = MIXOBJECT.Gases[GasIndex].WK[0]
                        object.IPNNT[NP] = 1
                        L = 2
                        object.IARRYNT[NP] = L
                        object.PENFRANT[0][NP] = 0.0
                        object.PENFRANT[1][NP] = 0.0
                        object.PENFRANT[2][NP] = 0.0
                    elif MIXOBJECT.Gases[GasIndex].NION > 1:
                        NP = NP - MIXOBJECT.Gases[GasIndex].NION
                        for KION in range(MIXOBJECT.Gases[GasIndex].NION):
                            NP = NP + 1
                            RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                            object.RGASNT[NP] = RGAS
                            object.EINNT[NP] = MIXOBJECT.Gases[GasIndex].EION[KION] / RGAS
                            object.WPLNT[NP] = MIXOBJECT.Gases[GasIndex].EB[KION]
                            object.NC0NT[NP] = MIXOBJECT.Gases[GasIndex].NC0[KION]
                            object.EC0NT[NP] = MIXOBJECT.Gases[GasIndex].EC0[KION]
                            object.NG1NT[NP] = MIXOBJECT.Gases[GasIndex].NG1[KION]
                            object.EG2NT[NP] = MIXOBJECT.Gases[GasIndex].EG2[KION]
                            object.EFLNT[NP] = MIXOBJECT.Gases[GasIndex].EFL[KION]
                            object.EG1NT[NP] = MIXOBJECT.Gases[GasIndex].EG1[KION]
                            object.NG2NT[NP] = MIXOBJECT.Gases[GasIndex].NG2[KION]
                            object.WKLMNT[NP] = MIXOBJECT.Gases[GasIndex].WK[KION]
                            object.IPNNT[NP] = 1
                            L = 2
                            object.IARRYNT[NP] = L
                            object.PENFRANT[0][NP] = 0.0
                            object.PENFRANT[1][NP] = 0.0
                            object.PENFRANT[2][NP] = 0.0

            if object.FinalElectronEnergy >= MIXOBJECT.Gases[GasIndex].E[3]:
                if MIXOBJECT.Gases[GasIndex].NATT <= 1:
                    NP += 1
                    object.CFNT[IE][NP] = MIXOBJECT.Gases[GasIndex].Q[3][IE] * object.VANN[GasIndex]
                    object.FCATT[IE] = object.FCATT[IE] + object.CFNT[IE][NP]
                    object.PSCTNT[IE][NP] = 0.5
                    object.ANGCTNT[IE][NP] = 1.0
                    if IE == 0:
                        RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EINNT[NP] = 0.0
                        object.INDEXNT[NP] = 0
                        object.IPNNT[NP] = -1
                        L = 3
                        object.IARRYNT[NP] = L
                        object.PENFRANT[0][NP] = 0.0
                        object.PENFRANT[1][NP] = 0.0
                        object.PENFRANT[2][NP] = 0.0
                elif MIXOBJECT.Gases[GasIndex].NATT > 1:
                    for JJ in range(int(MIXOBJECT.Gases[GasIndex].NATT)):
                        NP += 1
                        object.CFNT[IE][NP] = MIXOBJECT.Gases[GasIndex].QATT[JJ][IE] * object.VANN[GasIndex]
                        object.FCATT[IE] = object.FCATT[IE] + object.CFNT[IE][NP]
                        object.PSCTNT[IE][NP] = 0.5
                        object.ANGCTNT[IE][NP] = 1.0
                        if IE == 0:
                            RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
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
            if MIXOBJECT.Gases[GasIndex].NIN > 0:
                for J in range(int(MIXOBJECT.Gases[GasIndex].NIN)):
                    NP = NP + 1
                    object.CFNT[IE][NP] = MIXOBJECT.Gases[GasIndex].QIN[J][IE] * object.VANN[GasIndex]
                    object.PSCTNT[IE][NP] = 0.5
                    object.ANGCTNT[IE][NP] = 1.0
                    object.INDEXNT[NP] = 0
                    if MIXOBJECT.Gases[GasIndex].KIN[J] == 1:

                        PSCT1 = MIXOBJECT.Gases[GasIndex].PEQIN[J][IE]
                        AngObject.PSCT1 = PSCT1
                        AngObject.AngCut()
                        object.ANGCTNT[IE][NP] = AngObject.ANGC
                        object.PSCTNT[IE][NP] = AngObject.PSCT2
                        object.INDEXNT[NP] = 1
                    elif MIXOBJECT.Gases[GasIndex].KIN[J] == 2:

                        object.PSCTNT[IE][NP] = MIXOBJECT.Gases[GasIndex].PEQIN[J][IE]
                        object.INDEXNT[NP] = 2
                    if IE == 0:

                        RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EINNT[NP] = MIXOBJECT.Gases[GasIndex].EIN[J] / RGAS
                        L = 4

                        if MIXOBJECT.Gases[GasIndex].EIN[J] < 0:
                            L = 5
                        object.IPNNT[NP] = 0
                        object.IARRYNT[NP] = L
                        object.PENFRANT[0][NP] = MIXOBJECT.Gases[GasIndex].PENFRA[0][J]
                        object.PENFRANT[1][NP] = MIXOBJECT.Gases[GasIndex].PENFRA[1][J] * 1.0e-16 / sqrt(3)
                        object.PENFRANT[2][NP] = MIXOBJECT.Gases[GasIndex].PENFRA[2][J]

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
            NNULLSUM += MIXOBJECT.Gases[GasIndex].NNULL
        if NNULLSUM != 0:
            for I in range(object.NumberOfGases):
                if MIXOBJECT.Gases[GasIndex].NNULL > 0:
                    for J in range(MIXOBJECT.Gases[GasIndex].NNULL):
                        object.SCLENULNT[NP] = MIXOBJECT.Gases[GasIndex].SCLN[J]
                        object.CFNNT[IE][NP] = MIXOBJECT.Gases[GasIndex].QNULL[J][IE] * object.VANN[GasIndex] * \
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

    for GasIndex in range(object.NumberOfGases):
        for J in range(6):
            KELSUM += MIXOBJECT.Gases[GasIndex].KEL[J]

    for GasIndex in range(object.NumberOfGases):
        for J in range(250):
            KELSUM += MIXOBJECT.Gases[GasIndex].KIN[J]

    if KELSUM > 0:
        object.AnisotropicDetected = 1

    BP = object.EField ** 2 * object.CONST1
    F2 = object.EField * object.CONST3
    ELOW = object.MaxCollisionTime * (object.MaxCollisionTime * BP - F2 * sqrt(0.5 * object.FinalElectronEnergy)) / object.ElectronEnergyStep - 1
    ELOW = min(ELOW, object.SmallNumber)
    EHI = object.MaxCollisionTime * (object.MaxCollisionTime * BP + F2 * sqrt(0.5 * object.FinalElectronEnergy)) / object.ElectronEnergyStep + 1
    if EHI > 10000:
        EHI = 10000
    for l in range(8):
        I = l + 1
        JLOW = 4000 - 500 * (9 - I) + 1 + int(ELOW)
        JHI = 4000 - 500 * (8 - I) + int(EHI)
        JLOW = max(JLOW, 0)
        JHI = min(JHI, 4000)
        for J in range(int(JLOW-1), int(JHI)):
            if (object.TCFNT[J] + object.TCFNNT[J] + abs(object.FAKEI)) > object.MaxCollisionFreqNT[l]:
                object.MaxCollisionFreqNT[l] = object.TCFNT[J] + object.TCFNNT[J] + abs(object.FAKEI)
    for I in range(object.EnergySteps):
        object.QTOT[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[0][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[0][I] + \
                         object.ANN[2] * MIXOBJECT.Gases[2].Q[0][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[0][I] + \
                         object.ANN[4] * MIXOBJECT.Gases[4].Q[0][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[0][I]
        object.QEL[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[1][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[1][I] + \
                        object.ANN[2] * MIXOBJECT.Gases[2].Q[1][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[1][I] + \
                        object.ANN[4] * MIXOBJECT.Gases[4].Q[1][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[1][I]

        for GasIndex in range(object.NumberOfGases):
            object.QION[GasIndex][I] = MIXOBJECT.Gases[GasIndex].Q[2][I] * object.ANN[GasIndex]
            QATT[GasIndex][I] = MIXOBJECT.Gases[GasIndex].Q[3][I] * object.ANN[GasIndex]
            if MIXOBJECT.Gases[GasIndex].NION > 1:
                object.QION[GasIndex][I] = 0.0
                for KION in range(MIXOBJECT.Gases[GasIndex].NION):
                    object.QION[GasIndex][I] += MIXOBJECT.Gases[GasIndex].QION[KION][I] * object.ANN[GasIndex]
        object.QREL[I] = 0.0
        object.QSATT[I] = 0.0
        object.QSUM[I] = 0.0
        for J in range(object.NumberOfGases):
            object.QSUM[I] = object.QSUM[I] + object.QION[J][I] + QATT[J][I]
            object.QSATT[I] = object.QSATT[I] + QATT[J][I]
            object.QREL[I] = object.QREL[I] + object.QION[J][I] - QATT[J][I]
        for GasIndex in range(6):
            for J in range(int(MIXOBJECT.Gases[GasIndex].NIN)):
                object.QSUM[I] = object.QSUM[I] + MIXOBJECT.Gases[GasIndex].QIN[J][I] * object.ANN[GasIndex]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef MixerT(PyBoltz object):
    """
    This function sets up the given PyBoltz object with the right values for the requested gas mixture. It uses the Gasmix object
    to get the momentum cross sections and all the needed values.

    The object parameter is the PyBoltz object to be setup.
    """
    cdef double QATT[6][4000]
    cdef Gasmix MIXOBJECT
    cdef int  IE, GasIndex, NP, p, sum, J, i, j, KION, JJ, IL, I
    ECHARG = 1.602176565e-19

    object.ElectronEnergyStep = object.FinalElectronEnergy / float(object.EnergySteps)

    EHALF = object.ElectronEnergyStep / 2

    object.E[0] = EHALF
    for i in range(1, 4000):
        object.E[i] = EHALF + object.ElectronEnergyStep * i
        object.EROOT[i] = sqrt(object.E[i])
    object.EROOT[0] = sqrt(EHALF)

    MIXOBJECT = Gasmix()
    MIXOBJECT.InitWithInfo(object.GasIDs, object.QIN, object.NIN, object.PENFRA,
                           object.E, object.EROOT, object.QTOT, object.QREL, object.QINEL, object.QEL,
                           object.DENSY, 0, object.NumberOfGases, object.EnergySteps, object.WhichAngularModel, object.ElectronEnergyStep,
                           object.FinalElectronEnergy, object.ThermalEnergy, object.RhydbergConst, object.TemperatureCentigrade, object.PressureTorr, object.EnablePenning, object.PIR2)
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
        for GasIndex in range(object.NumberOfGases):
            object.FCION[IE] = 0.0
            object.FCATT[IE] = 0.0
            NP = 1

            object.CF[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].Q[1][IE] * object.VANN[GasIndex]
            object.PSCT[GasIndex][IE][NP - 1] = 0.5
            object.ANGCT[GasIndex][IE][NP - 1] = 1
            object.INDEX[GasIndex][NP - 1] = 0
            AngObject = Ang()

            if MIXOBJECT.Gases[GasIndex].KEL[1] == 1:
                PSCT1 = MIXOBJECT.Gases[GasIndex].PEQEL[1][IE]
                AngObject.PSCT1 = PSCT1
                AngObject.AngCut()
                object.ANGCT[GasIndex][IE][NP - 1] = AngObject.ANGC
                object.PSCT[GasIndex][IE][NP - 1] = AngObject.PSCT2
                object.INDEX[GasIndex][NP - 1] = 1
            elif MIXOBJECT.Gases[GasIndex].KEL[1] == 2:
                object.PSCT[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].PEQEL[1][IE]
                object.INDEX[GasIndex][NP - 1] = 2

            if IE == 0:
                RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                object.AMGAS[GasIndex] = 2 * EMASS / MIXOBJECT.Gases[GasIndex].E[1]
                object.RGAS[GasIndex][NP - 1] = RGAS
                L = 1
                object.IARRY[GasIndex][NP - 1] = L
                object.EIN[GasIndex][NP - 1] = 0.0
                object.IPN[GasIndex][NP - 1] = 0

                object.PENFRA[GasIndex][0][NP - 1] = 0.0
                object.PENFRA[GasIndex][1][NP - 1] = 0.0
                object.PENFRA[GasIndex][2][NP - 1] = 0.0

            # IONISATION
            if object.FinalElectronEnergy >= MIXOBJECT.Gases[GasIndex].E[2]:
                if MIXOBJECT.Gases[GasIndex].NION <= 1:
                    NP += 1
                    object.CF[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].Q[2][IE] * object.VANN[GasIndex]
                    object.FCION[IE] = object.FCION[IE] + object.CF[GasIndex][IE][NP - 1]
                    object.PSCT[GasIndex][IE][NP - 1] = 0.5
                    object.ANGCT[GasIndex][IE][NP - 1] = 1.0
                    object.INDEX[GasIndex][NP - 1] = 0
                    if MIXOBJECT.Gases[GasIndex].KEL[2] == 1:
                        PSCT1 = MIXOBJECT.Gases[GasIndex].PEQEL[2][IE]
                        AngObject.PSCT1 = PSCT1
                        AngObject.AngCut()
                        object.ANGCT[GasIndex][IE][NP - 1] = AngObject.ANGC
                        object.PSCT[GasIndex][IE][NP - 1] = AngObject.PSCT2
                        object.INDEX[GasIndex][NP - 1] = 1
                    elif MIXOBJECT.Gases[GasIndex].KEL[2] == 2:
                        object.PSCT[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].PEQEL[2][IE]
                        object.INDEX[GasIndex][NP - 1] = 2
                elif MIXOBJECT.Gases[GasIndex].NION > 1:
                    for KION in range(MIXOBJECT.Gases[GasIndex].NION):
                        NP += 1
                        object.CF[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].QION[KION][IE] * object.VANN[GasIndex]
                        object.FCION[IE] = object.FCION[IE] + object.CF[GasIndex][IE][NP - 1]
                        object.PSCT[GasIndex][IE][NP - 1] = 0.5
                        object.ANGCT[GasIndex][IE][NP - 1] = 1.0
                        object.INDEX[GasIndex][NP - 1] = 0
                        if MIXOBJECT.Gases[0].KEL[2] == 1:
                            PSCT1 = MIXOBJECT.Gases[GasIndex].PEQION[KION][IE]
                            AngObject.PSCT1 = PSCT1
                            AngObject.AngCut()
                            object.ANGCT[GasIndex][IE][NP - 1] = AngObject.ANGC
                            object.PSCT[GasIndex][IE][NP - 1] = AngObject.PSCT2
                            object.INDEX[GasIndex][NP - 1] = 1
                        elif MIXOBJECT.Gases[0].KEL[2] == 2:
                            object.PSCT[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].PEQION[KION][IE]
                            object.INDEX[GasIndex][NP - 1] = 2
                if IE == 0:
                    if MIXOBJECT.Gases[GasIndex].NION <= 1:
                        RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                        object.RGAS[GasIndex][NP - 1] = RGAS
                        object.EIN[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].E[2] / RGAS
                        object.WPL[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EB[0]
                        object.NC0[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].NC0[0]
                        object.EC0[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EC0[0]
                        object.NG1[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].NG1[0]
                        object.EG1[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EG1[0]
                        object.NG2[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].NG2[0]
                        object.EG2[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EG2[0]
                        object.WKLM[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].WK[0]
                        object.EFL[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EFL[0]
                        object.IPN[GasIndex][NP - 1] = 1
                        L = 2
                        object.IARRY[GasIndex][NP - 1] = L
                        object.PENFRA[GasIndex][0][NP - 1] = 0.0
                        object.PENFRA[GasIndex][1][NP - 1] = 0.0
                        object.PENFRA[GasIndex][2][NP - 1] = 0.0
                    elif MIXOBJECT.Gases[GasIndex].NION > 1:
                        NP = NP - MIXOBJECT.Gases[GasIndex].NION
                        for KION in range(MIXOBJECT.Gases[GasIndex].NION):
                            NP = NP + 1
                            RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                            object.RGAS[GasIndex][NP - 1] = RGAS
                            object.EIN[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EION[KION] / RGAS
                            object.WPL[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EB[KION]
                            object.NC0[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].NC0[KION]
                            object.EC0[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EC0[KION]
                            object.EG2[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EG2[KION]
                            object.NG1[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].NG1[KION]
                            object.EG1[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EG1[KION]
                            object.NG2[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].NG2[KION]
                            object.WKLM[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].WK[KION]
                            object.EFL[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EFL[KION]
                            object.IPN[GasIndex][NP - 1] = 1
                            L = 2
                            object.IARRY[GasIndex][NP - 1] = L
                            object.PENFRA[GasIndex][0][NP - 1] = 0.0
                            object.PENFRA[GasIndex][1][NP - 1] = 0.0
                            object.PENFRA[GasIndex][2][NP - 1] = 0.0
            # ATTACHMENT
            if object.FinalElectronEnergy >= MIXOBJECT.Gases[GasIndex].E[3]:
                if MIXOBJECT.Gases[GasIndex].NATT <= 1:
                    NP += 1
                    object.CF[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].Q[3][IE] * object.VANN[GasIndex]
                    object.FCATT[IE] = object.FCATT[IE] + object.CF[GasIndex][IE][NP - 1]
                    object.PSCT[GasIndex][IE][NP - 1] = 0.5
                    object.ANGCT[GasIndex][IE][NP - 1] = 1.0
                    if IE == 0:
                        RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                        object.RGAS[GasIndex][NP - 1] = RGAS
                        object.EIN[GasIndex][NP - 1] = 0.0
                        object.INDEX[GasIndex][NP - 1] = 0
                        object.IPN[GasIndex][NP - 1] = -1
                        L = 3
                        object.IARRY[GasIndex][NP - 1] = L
                        object.PENFRA[GasIndex][0][NP - 1] = 0.0
                        object.PENFRA[GasIndex][1][NP - 1] = 0.0
                        object.PENFRA[GasIndex][2][NP - 1] = 0.0

                elif MIXOBJECT.Gases[GasIndex].NATT > 1:
                    for JJ in range(int(MIXOBJECT.Gases[GasIndex].NATT)):
                        NP += 1
                        object.CF[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].QATT[JJ][IE] * object.VANN[GasIndex]
                        object.FCATT[IE] = object.FCATT[IE] + object.CF[GasIndex][IE][NP - 1]
                        object.PSCT[GasIndex][IE][NP - 1] = 0.5
                        object.ANGCT[GasIndex][IE][NP - 1] = 1.0
                        if IE == 0:
                            RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                            object.RGAS[GasIndex][NP - 1] = RGAS
                            object.EIN[GasIndex][NP - 1] = 0.0
                            object.INDEX[GasIndex][NP - 1] = 0
                            object.IPN[GasIndex][NP - 1] = -1
                            L = 3
                            object.IARRY[GasIndex][NP - 1] = L
                            object.PENFRA[GasIndex][0][NP - 1] = 0.0
                            object.PENFRA[GasIndex][1][NP - 1] = 0.0
                            object.PENFRA[GasIndex][2][NP - 1] = 0.0
            # INELASTIC AND SUPERELASTIC
            if MIXOBJECT.Gases[GasIndex].NIN > 0:
                for J in range(int(MIXOBJECT.Gases[GasIndex].NIN)):
                    NP = NP + 1
                    object.CF[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].QIN[J][IE] * object.VANN[GasIndex]
                    object.PSCT[GasIndex][IE][NP - 1] = 0.5
                    object.ANGCT[GasIndex][IE][NP - 1] = 1.0
                    object.INDEX[GasIndex][NP - 1] = 0
                    if MIXOBJECT.Gases[GasIndex].KIN[J] == 1:
                        PSCT1 = MIXOBJECT.Gases[GasIndex].PEQIN[J][IE]
                        AngObject.PSCT1 = PSCT1
                        AngObject.AngCut()
                        object.ANGCT[GasIndex][IE][NP - 1] = AngObject.ANGC
                        object.PSCT[GasIndex][IE][NP - 1] = AngObject.PSCT2
                        object.INDEX[GasIndex][NP - 1] = 1
                    elif MIXOBJECT.Gases[GasIndex].KIN[J] == 2:
                        object.PSCT[GasIndex][IE][NP - 1] = MIXOBJECT.Gases[GasIndex].PEQIN[J][IE]
                        object.INDEX[GasIndex][NP - 1] = 2
                    if IE == 0:
                        RGAS = 1 + MIXOBJECT.Gases[GasIndex].E[1] / 2
                        object.RGAS[GasIndex][NP - 1] = RGAS
                        object.EIN[GasIndex][NP - 1] = MIXOBJECT.Gases[GasIndex].EIN[J] / RGAS
                        L = 4
                        if MIXOBJECT.Gases[GasIndex].EIN[J] < 0:
                            L = 5
                        object.IPN[GasIndex][NP - 1] = 0
                        object.IARRY[GasIndex][NP - 1] = L
                        object.PENFRA[GasIndex][0][NP - 1] = MIXOBJECT.Gases[GasIndex].PENFRA[0][J]
                        object.PENFRA[GasIndex][1][NP - 1] = MIXOBJECT.Gases[GasIndex].PENFRA[1][J] * 1.0e-16 / sqrt(3)
                        object.PENFRA[GasIndex][2][NP - 1] = MIXOBJECT.Gases[GasIndex].PENFRA[2][J]

            object.IPLAST[GasIndex] = NP
            object.ISIZE[GasIndex] = 1
            for I in range(1, 9):
                if object.IPLAST[GasIndex] >= 2 ** I:
                    object.ISIZE[GasIndex] = 2 ** I
                else:
                    break
            # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
            object.TCF[GasIndex][IE] = 0.0
            for p in range(int(object.IPLAST[GasIndex])):
                object.TCF[GasIndex][IE] = object.TCF[GasIndex][IE] + object.CF[GasIndex][IE][p]
                if object.CF[GasIndex][IE][p] < 0:
                    print ("WARNING NEGATIVE COLLISION FREQUENCY at gas " +str(p)+"  "+ str(IE))

            for p in range(int(object.IPLAST[GasIndex])):
                if object.TCF[GasIndex][IE] == 0:
                    object.CF[GasIndex][IE][p] = 0
                else:
                    object.CF[GasIndex][IE][p] = object.CF[GasIndex][IE][p] / object.TCF[GasIndex][IE]

            for p in range(1, int(object.IPLAST[GasIndex])):
                object.CF[GasIndex][IE][p] = object.CF[GasIndex][IE][p] + object.CF[GasIndex][IE][p - 1]
            object.FCATT[IE] = object.FCATT[IE] * object.EROOT[IE]
            object.FCION[IE] = object.FCION[IE] * object.EROOT[IE]
            object.TCF[GasIndex][IE] = object.TCF[GasIndex][IE] * object.EROOT[IE]
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

        for GasIndex in range(object.NumberOfGases):
            object.TCFN[GasIndex][IE] = 0.0
            for IL in range(int(object.NPLAST[GasIndex])):
                object.TCFN[GasIndex][IE] = object.TCFN[GasIndex][IE] + object.CFN[GasIndex][IE][IL]
                if object.CFN[GasIndex][IE][IL] < 0:
                    print "WARNING NEGATIVE NULL COLLISION FREQUENCY"
            for IL in range(int(object.NPLAST[GasIndex])):
                if object.TCFN[GasIndex][IE] == 0:
                    object.CFN[GasIndex][IE][IL] = 0.0
                else:
                    object.CFN[GasIndex][IE][IE] = object.CFN[GasIndex][IE][IL] / object.TCFN[GasIndex][IE]

            for IL in range(1, int(object.NPLAST[GasIndex])):
                object.CFN[GasIndex][IE][IL] = object.CFN[GasIndex][IE][IL] + object.CFN[GasIndex][IE][IL - 1]
            object.TCFN[GasIndex][IE] = object.TCFN[GasIndex][IE] * object.EROOT[IE]
    KELSUM = 0

    for GasIndex in range(object.NumberOfGases):
        for J in range(6):
            KELSUM += MIXOBJECT.Gases[GasIndex].KEL[J]

    for GasIndex in range(object.NumberOfGases):
        for J in range(250):
            KELSUM += MIXOBJECT.Gases[GasIndex].KIN[J]

    if KELSUM > 0:
        object.AnisotropicDetected = 1

    # CALCULATE NULL COLLISION FREQUENCIES FOR EACH GAS COMPONENT
    FAKEIN = abs(object.FAKEI) / object.NumberOfGases
    for GasIndex in range(object.NumberOfGases):
        object.MaxCollisionFreq[GasIndex] = 0.0
        for IE in range(4000):
            if object.TCF[GasIndex][IE] + object.TCFN[GasIndex][IE] + FAKEIN >= object.MaxCollisionFreq[GasIndex]:
                object.MaxCollisionFreq[GasIndex] = object.TCF[GasIndex][IE] + object.TCFN[GasIndex][IE] + FAKEIN
    # CALCULATE EACH GAS CUMLATIVE FRACTION NULL COLLISION FREQUENCIES
    object.MaxCollisionFreqTotal = 0.0
    for GasIndex in range(object.NumberOfGases):
        object.MaxCollisionFreqTotal = object.MaxCollisionFreqTotal + object.MaxCollisionFreq[GasIndex]
    for GasIndex in range(object.NumberOfGases):
        object.MaxCollisionFreqTotalG[GasIndex] = object.MaxCollisionFreq[GasIndex] / object.MaxCollisionFreqTotal
    for GasIndex in range(1, object.NumberOfGases):
        object.MaxCollisionFreqTotalG[GasIndex] = object.MaxCollisionFreqTotalG[GasIndex] + object.MaxCollisionFreqTotalG[GasIndex - 1]

    # CALCULATE MAXWELL BOLTZMAN VELOCITY FACTOR FOR EACH GAS COMPONENT

    for GasIndex in range(object.NumberOfGases):
        object.VTMB[GasIndex] = sqrt(2.0 * ECHARG * object.ThermalEnergy / object.AMGAS[GasIndex]) * 1e-12

    for I in range(object.EnergySteps):
        object.QTOT[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[0][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[0][I] + \
                         object.ANN[2] * MIXOBJECT.Gases[2].Q[0][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[0][I] + \
                         object.ANN[4] * MIXOBJECT.Gases[4].Q[0][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[0][I]
        object.QEL[I] = object.ANN[0] * MIXOBJECT.Gases[0].Q[1][I] + object.ANN[1] * MIXOBJECT.Gases[1].Q[1][I] + \
                        object.ANN[2] * MIXOBJECT.Gases[2].Q[1][I] + object.ANN[3] * MIXOBJECT.Gases[3].Q[1][I] + \
                        object.ANN[4] * MIXOBJECT.Gases[4].Q[1][I] + object.ANN[5] * MIXOBJECT.Gases[5].Q[1][I]

        for GasIndex in range(6):
            object.QION[GasIndex][I] = MIXOBJECT.Gases[GasIndex].Q[2][I] * object.ANN[GasIndex]
            QATT[GasIndex][I] = MIXOBJECT.Gases[GasIndex].Q[3][I] * object.ANN[GasIndex]
        object.QREL[I] = 0.0
        object.QSATT[I] = 0.0
        object.QSUM[I] = 0.0
        for J in range(object.NumberOfGases):
            object.QSUM[I] = object.QSUM[I] + object.QION[J][I] + QATT[J][I]
            object.QSATT[I] = object.QSATT[I] + QATT[J][I]
            object.QREL[I] = object.QREL[I] + object.QION[J][I] - QATT[J][I]
        for GasIndex in range(6):
            for J in range(int(MIXOBJECT.Gases[GasIndex].NIN)):
                object.QSUM[I] = object.QSUM[I] + MIXOBJECT.Gases[GasIndex].QIN[J][I] * object.ANN[GasIndex]

    return
