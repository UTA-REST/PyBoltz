import Magboltz
import math
from Gasmix import Gasmix
import numpy as np
from ANG import ANG


def MIXER():
    global Magboltz
    ECHARG = 1.602176565e-19
    KEL = np.zeros(shape=(6, 6))

    E = np.zeros(shape=(6, 6))
    Q = np.zeros(shape=(6, 6, 4000))

    PEQEL = np.zeros(shape=(6, 6, 4000))

    EION = np.zeros(shape=(6, 30))

    EB = np.zeros(shape=(6, 30))

    EC0 = np.zeros(shape=(6, 30))

    EG1 = np.zeros(shape=(6, 30))

    EG2 = np.zeros(shape=(6, 30))

    WK = np.zeros(shape=(6, 30))

    EFL = np.zeros(shape=(6, 30))

    NC0 = np.zeros(shape=(6, 30))

    NG1 = np.zeros(shape=(6, 30))

    NG2 = np.zeros(shape=(6, 30))

    EI = np.zeros(shape=(6, 250))

    KIN = np.zeros(shape=(6, 250))

    NION = np.zeros(6)

    QION = np.zeros(shape=(6, 30, 4000))

    PEQION = np.zeros(shape=(6, 30, 4000))

    PEQIN = np.zeros(shape=(6, 250, 4000))

    QATT = np.zeros(shape=(6, 4000))

    QNULL = np.zeros(shape=(6, 10, 4000))

    SCLN = np.zeros(shape=(6, 10))

    NATT = np.zeros(6)

    NNULL = np.zeros(6)

    Magboltz.ESTEP = Magboltz.EFINAL / Magboltz.NSTEP

    EHALF = Magboltz.ESTEP / 2

    Magboltz.E[0] = EHALF

    MIXOBJECT = Gasmix()
    MIXOBJECT.InitWithInfo(Magboltz.NGASN, Q, Magboltz.QIN, Magboltz.NIN, E, EI, KIN, QION, PEQION, EION, EB, PEQEL,
                           PEQIN, KEL, Magboltz.PENFRA, NC0, EC0, WK, EFL, NG1, EG1, NG2, EG2, QATT, QNULL, SCLN,
                           Magboltz.E, Magboltz.EROOT, Magboltz.QTOT, Magboltz.QREL, Magboltz.QINEL, Magboltz.QEL,
                           Magboltz.DENSY, 0, Magboltz.NGAS, Magboltz.NSTEP, Magboltz.NANISO, Magboltz.ESTEP,
                           Magboltz.EFINAL, Magboltz.AKT, Magboltz.ARY, Magboltz.TEMPC, Magboltz.TORR, Magboltz.IPEN,
                           NION, NATT, NNULL)
    MIXOBJECT.Run()
    EMASS = 9.10938291e-31

    for IE in range(4000):
        NP = 0
        for KGAS in range(Magboltz.NGAS):
            if KGAS == 1:
                Magboltz.CF[IE][NP] = MIXOBJECT.Gases[0].Q[1][IE] * Magboltz.VANN[0]
            elif KGAS == 2:
                Magboltz.CF[IE][NP] = MIXOBJECT.Gases[1].Q[1][IE] * Magboltz.VANN[1]
            elif KGAS == 3:
                Magboltz.CF[IE][NP] = MIXOBJECT.Gases[2].Q[1][IE] * Magboltz.VANN[2]
            elif KGAS == 4:
                Magboltz.CF[IE][NP] = MIXOBJECT.Gases[3].Q[1][IE] * Magboltz.VANN[3]
            elif KGAS == 5:
                Magboltz.CF[IE][NP] = MIXOBJECT.Gases[4].Q[1][IE] * Magboltz.VANN[4]
            elif KGAS == 6:
                Magboltz.CF[IE][NP] = MIXOBJECT.Gases[5].Q[1][IE] * Magboltz.VANN[5]
            Magboltz.PSCT[IE][NP] = 0.5
            Magboltz.ANGCT[IE][NP] = 1
            Magboltz.INDEX[NP] = 0
            ANGOBJECT = ANG()

            if MIXOBJECT.Gases[KGAS].KEL[1] == 1:
                PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                ANGOBJECT.PSCT1 = PSCT1
                ANGOBJECT.ANGCUT()
                Magboltz.ANGCT[IE][NP] = ANGOBJECT.ANGC
                Magboltz.PSCT[IE][NP] = ANGOBJECT.PSCT2
                Magboltz.INDEX[NP] = 1
            elif MIXOBJECT.Gases[KGAS].KEL[1] == 2:
                Magboltz.PSCT[IE][NP] = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                Magboltz.INDEX[NP] = 2

            if IE == 0:
                RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                Magboltz.RGAS[NP] = RGAS
                L = 1
                Magboltz.IARRY[NP] = L
                Magboltz.EIN[NP] = 0.0
                Magboltz.IPN[NP] = 0

                Magboltz.PENFRA[0][NP] = 0.0
                Magboltz.PENFRA[1][NP] = 0.0
                Magboltz.PENFRA[2][NP] = 0.0
                # IONISATION

            if Magboltz.EFINAL >= MIXOBJECT.Gases[KGAS].E[2]:
                if MIXOBJECT.Gases[KGAS].NION <= 1:
                    NP += 1
                    Magboltz.CF[IE][NP] = MIXOBJECT.Gases[KGAS].Q[2][IE] * Magboltz.VANN[KGAS]
                    Magboltz.FCION[IE] = Magboltz.FCION[IE] + Magboltz.CF[KGAS][IE][NP]
                    Magboltz.PSCT[IE][NP] = 0.5
                    Magboltz.ANGCT[IE][NP] = 1.0
                    Magboltz.INDEX[NP] = 0
                    if MIXOBJECT.Gases[KGAS].KEL[2] == 1:
                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        Magboltz.ANGCT[IE][NP] = ANGOBJECT.ANGC
                        Magboltz.PSCT[IE][NP] = ANGOBJECT.PSCT2
                        Magboltz.INDEX[NP] = 1
                    elif MIXOBJECT.Gases[KGAS].KEL[2] == 2:
                        Magboltz.PSCT[IE][NP] = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        Magboltz.INDEX[NP] = 2
                elif MIXOBJECT.Gases[KGAS].NION > 1:
                    for KION in range(MIXOBJECT.NION[KGAS]):
                        NP += 1
                        Magboltz.CF[IE][NP] = MIXOBJECT.Gases[KGAS].QION[KION][IE] * Magboltz.VANN[KGAS]
                        Magboltz.FCION[IE] = Magboltz.FCION[IE] + Magboltz.CF[KGAS][IE][NP]
                        Magboltz.PSCT[IE][NP] = 0.5
                        Magboltz.ANGCT[IE][NP] = 1.0
                        Magboltz.INDEX[NP] = 0
                        if MIXOBJECT.Gases[KGAS].KEL[2] == 1:
                            PSCT1 = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            ANGOBJECT.PSCT1 = PSCT1
                            ANGOBJECT.ANGCUT()
                            Magboltz.ANGCT[IE][NP] = ANGOBJECT.ANGC
                            Magboltz.PSCT[IE][NP] = ANGOBJECT.PSCT2
                            Magboltz.INDEX[NP] = 1
                        elif MIXOBJECT.Gases[KGAS].KEL[2] == 2:
                            Magboltz.PSCT[IE][NP] = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            Magboltz.INDEX[NP] = 2

                if IE == 0:
                    if MIXOBJECT.Gases[KGAS].NION <= 1:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        Magboltz.RGAS[NP] = RGAS
                        Magboltz.EIN[NP] = MIXOBJECT.Gases[KGAS].E[2] / RGAS
                        Magboltz.WPL[NP] = MIXOBJECT.Gases[KGAS].EB[0]
                        Magboltz.NC0[NP] = MIXOBJECT.Gases[KGAS].NC0[0]
                        Magboltz.EC0[NP] = MIXOBJECT.Gases[KGAS].EC0[0]
                        Magboltz.NG1[NP] = MIXOBJECT.Gases[KGAS].NG1[0]
                        Magboltz.EG1[NP] = MIXOBJECT.Gases[KGAS].EG1[0]
                        Magboltz.EG2[NP] = MIXOBJECT.Gases[KGAS].EG2[0]
                        Magboltz.NG2[NP] = MIXOBJECT.Gases[KGAS].NG2[0]
                        Magboltz.EFL[NP] = MIXOBJECT.Gases[KGAS].EFL[1]
                        Magboltz.WKLM[NP] = MIXOBJECT.Gases[KGAS].WK[1]
                        Magboltz.IPN[NP] = 1
                        L = 2
                        Magboltz.IARRY[NP] = L
                        Magboltz.PENFRA[0][NP] = 0.0
                        Magboltz.PENFRA[1][NP] = 0.0
                        Magboltz.PENFRA[2][NP] = 0.0
                    elif MIXOBJECT.Gases[KGAS].NION > 1:
                        NP = NP - MIXOBJECT.Gases[KGAS].NION
                        for KION in range(MIXOBJECT.Gases[KGAS].NION):
                            NP = NP + 1
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            Magboltz.RGAS[NP] = RGAS
                            Magboltz.EIN[NP] = MIXOBJECT.Gases[KGAS].EION[KION] / RGAS
                            Magboltz.WPL[NP] = MIXOBJECT.Gases[KGAS].EB[KION]
                            Magboltz.NC0[NP] = MIXOBJECT.Gases[KGAS].NC0[KION]
                            Magboltz.EC0[NP] = MIXOBJECT.Gases[KGAS].EC0[KION]
                            Magboltz.NG1[NP] = MIXOBJECT.Gases[KGAS].NG1[KION]
                            Magboltz.EG2[NP] = MIXOBJECT.Gases[KGAS].EG2[KION]
                            Magboltz.EFL[NP] = MIXOBJECT.Gases[KGAS].EFL[KION]
                            Magboltz.EG1[NP] = MIXOBJECT.Gases[KGAS].EG1[KION]
                            Magboltz.NG2[NP] = MIXOBJECT.Gases[KGAS].NG2[KION]
                            Magboltz.WKLM[NP] = MIXOBJECT.Gases[KGAS].WK[KION]
                            Magboltz.IPN[NP] = 1
                            L = 2
                            Magboltz.IARRY[NP] = L
                            Magboltz.PENFRA[0][NP] = 0.0
                            Magboltz.PENFRA[1][NP] = 0.0
                            Magboltz.PENFRA[2][NP] = 0.0

            if Magboltz.EFINAL >= MIXOBJECT.Gases[KGAS].E[3]:
                if MIXOBJECT.Gases[KGAS].NATT <= 1:
                    NP += 1
                    Magboltz.CF[IE][NP] = MIXOBJECT.Gases[KGAS].Q[3][IE] * Magboltz.VANN[KGAS]
                    Magboltz.FCATT[IE] = Magboltz.FCATT[IE] + Magboltz.CF[KGAS][IE][NP]
                    Magboltz.PSCT[IE][NP] = 0.5
                    Magboltz.ANGCT[IE][NP] = 1.0
                elif MIXOBJECT.Gases[KGAS].NATT > 1:
                    for JJ in range(int(MIXOBJECT.Gases[KGAS].NATT)):
                        NP += 1
                        Magboltz.CF[IE][NP] = MIXOBJECT.Gases[KGAS].QATT[JJ][IE] * Magboltz.VANN[KGAS]
                        Magboltz.FCATT[IE] = Magboltz.FCATT[IE] + Magboltz.CF[KGAS][IE][NP]
                        Magboltz.PSCT[IE][NP] = 0.5
                        Magboltz.ANGCT[IE][NP] = 1.0
                        if IE == 0:
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            Magboltz.RGAS[NP] = RGAS
                            Magboltz.EIN[NP] = 0.0
                            Magboltz.INDEX[NP] = 0
                            Magboltz.IPN[NP] = -1
                            L = 3
                            Magboltz.IARRY[NP] = L
                            Magboltz.PENFRA[0][NP] = 0.0
                            Magboltz.PENFRA[1][NP] = 0.0
                            Magboltz.PENFRA[2][NP] = 0.0
                if IE == 0:
                    RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                    Magboltz.RGAS[NP] = RGAS
                    Magboltz.EIN[NP] = 0.0
                    Magboltz.INDEX[NP] = 0
                    Magboltz.IPN[NP] = -1
                    L = 3
                    Magboltz.IARRY[NP] = L
                    Magboltz.PENFRA[0][NP] = 0.0
                    Magboltz.PENFRA[1][NP] = 0.0
                    Magboltz.PENFRA[2][NP] = 0.0

            # INELASTIC AND SUPERELASTIC
            if MIXOBJECT.Gases[KGAS].NIN > 0:
                for J in range(int(MIXOBJECT.Gases[KGAS].NIN)):
                    NP = NP + 1
                    Magboltz.CF[IE][NP] = MIXOBJECT.Gases[KGAS].QIN[J][IE] * Magboltz.VANN[KGAS]
                    Magboltz.PSCT[IE][NP] = 0.5
                    Magboltz.ANGCT[IE][NP] = 1.0
                    Magboltz.INDEX[NP] = 0
                    if MIXOBJECT.Gases[KGAS].KIN[J] == 1:
                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        Magboltz.ANGCT[IE][NP] = ANGOBJECT.ANGC
                        Magboltz.PSCT[IE][NP] = ANGOBJECT.PSCT2
                        Magboltz.INDEX[NP] = 1
                    elif MIXOBJECT.Gases[KGAS].KIN[J] == 2:
                        Magboltz.PSCT[IE][NP] = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        Magboltz.INDEX[NP] = 2
                    if IE == 0:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        Magboltz.RGAS[NP] = RGAS
                        Magboltz.EIN[NP] = MIXOBJECT.Gases[KGAS].EI[J] / RGAS
                        L = 4
                        if MIXOBJECT.Gases[KGAS].EI[J] < 0:
                            L = 5
                        Magboltz.IPN[NP] = 0
                        Magboltz.IARRY[NP] = L
                        Magboltz.PENFRA[0][NP] = MIXOBJECT.Gases[KGAS].PENFRA[0][J]
                        Magboltz.PENFRA[1][NP] = MIXOBJECT.Gases[KGAS].PENFRA[1][J] * 1.0e-16 / math.sqrt(3)
                        Magboltz.PENFRA[2][NP] = MIXOBJECT.Gases[KGAS].PENFRA[2][J]
            Magboltz.IPLAST = NP
            Magboltz.ISIZE = 1
            for I in range(1, 9):
                if Magboltz.IPLAST >= 2 ** I:
                    Magboltz.ISIZE = 2 ** I
                else:
                    break
        # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
        Magboltz.TCF[IE] = 0.0
        for IF in range(int(Magboltz.IPLAST)):
            Magboltz.TCF[IE] = Magboltz.TCF[IE] + Magboltz.CF[IE][IF]
            if Magboltz.CF[IE][IF] < 0:
                print("WARNING NEGATIVE COLLISION FREQUENCY")

        for IF in range(int(Magboltz.IPLAST)):
            if Magboltz.TCF[IE] != 0.0:
                Magboltz.CF[IE][IF] /= Magboltz.TCF[IE]
            else:
                Magboltz.CF[IE][IF] = 0.0

        for IF in range(1, int(Magboltz.IPLAST)):
            Magboltz.CF[IE][IF] += Magboltz.CF[IE][IF - 1]
        Magboltz.FCATT[IE] *= Magboltz.EROOT[IE]
        Magboltz.FCION[IE] *= Magboltz.EROOT[IE]
        Magboltz.TCF[IE] *= Magboltz.EROOT[IE]

        NP = 0
        Magboltz.NPLAST = 0
        NNULLSUM = 0.0
        for I in range(Magboltz.NGAS):
            NNULLSUM += MIXOBJECT.Gases[KGAS].NNULL
        if NNULLSUM != 0:
            for I in range(Magboltz.NGAS):
                if MIXOBJECT.Gases[KGAS].NNULL > 0:
                    for J in range(MIXOBJECT.Gases[KGAS].NNULL):
                        Magboltz.SCLENUL[NP] = MIXOBJECT.Gases[KGAS].SCLN[J]
                        Magboltz.CFN[IE][NP] = MIXOBJECT.Gases[KGAS].QNULL[J][IE] * Magboltz.VANN[KGAS] * \
                                               Magboltz.SCLENUL[NP]
            Magboltz.NPLAST = NP
            Magboltz.TCFN[IE] = 0.0
            for IF in range(int(Magboltz.NPLAST)):
                Magboltz.TCFN[IE] = Magboltz.TCFN[IE] + Magboltz.CFN[IE][IF]
                if Magboltz.CFN[IE][IF] < 0:
                    print("WARNING NEGATIVE NULL COLLISION FREQUENCY")

            for IF in range(int(Magboltz.NPLAST)):
                if Magboltz.TCFN[IE] != 0.0:
                    Magboltz.CFN[IE][IF] /= Magboltz.TCFN[IE]
                else:
                    Magboltz.CFN[IE][IF] = 0.0

            for IF in range(1, int(Magboltz.NPLAST)):
                Magboltz.CFN[IE][IF] += Magboltz.CFN[IE][IF - 1]
            Magboltz.TCFN[IE] *= Magboltz.EROOT[IE]

    KELSUM = 0

    for KGAS in range(Magboltz.NGAS):
        for J in range(6):
            KELSUM += MIXOBJECT.Gases[KGAS].KEL[J]

    for KGAS in range(Magboltz.NGAS):
        for J in range(250):
            KELSUM += MIXOBJECT.Gases[KGAS].KIN[J]

    if KELSUM > 0:
        Magboltz.NISO = 1

    BP = Magboltz.EFIELD ** 2 * Magboltz.CONST1
    F2 = Magboltz.EFIELD * Magboltz.CONST3
    Magboltz.ELOW = Magboltz.TMAX * (Magboltz.TMAX * BP - F2 * math.sqrt(0.5 * Magboltz.EFINAL)) / Magboltz.ESTEP - 1
    Magboltz.ELOW = min(Magboltz.ELOW, Magboltz.SMALL)
    EHI = Magboltz.TMAX * (Magboltz.TMAX * BP + F2 * math.sqrt(0.5 * Magboltz.EFINAL)) / Magboltz.ESTEP + 1
    if EHI > 10000:
        EHI = 10000
    for l in range(8):
        l = I + 1
        JLOW = 4000 - 500 * (9 - I) + 1 + int(Magboltz.ELOW)
        JHI = 4000 - 500 * (8 - I) + input(EHI)
        JLOW = max(JLOW, 0)
        JHI = min(JHI, 4000)
        for J in range(int(JLOW), int(JHI)):
            if (Magboltz.TCF[J] + Magboltz.TCFN[J] + abs(Magboltz.FAKEI)) > Magboltz.TCFMAX[l]:
                Magboltz.TCFMAX[l] = Magboltz.TCF[J] + Magboltz.TCFN[J] + abs(Magboltz.FAKEI)
    for I in range(Magboltz.NSTEP):
        Magboltz.QTOT[I] = Magboltz.ANN[0] * MIXOBJECT.Gases[0].Q[0][I] + Magboltz.ANN[1] * MIXOBJECT.Gases[1].Q[0][I] + \
                           Magboltz.ANN[2] * MIXOBJECT.Gases[2].Q[0][I] + Magboltz.ANN[3] * MIXOBJECT.Gases[3].Q[0][I] + \
                           Magboltz.ANN[4] * MIXOBJECT.Gases[4].Q[0][I] + Magboltz.ANN[5] * MIXOBJECT.Gases[5].Q[0][I]
        Magboltz.QEL[I] = Magboltz.ANN[0] * MIXOBJECT.Gases[0].Q[1][I] + Magboltz.ANN[1] * MIXOBJECT.Gases[1].Q[1][I] + \
                          Magboltz.ANN[2] * MIXOBJECT.Gases[2].Q[1][I] + Magboltz.ANN[3] * MIXOBJECT.Gases[3].Q[1][I] + \
                          Magboltz.ANN[4] * MIXOBJECT.Gases[4].Q[1][I] + Magboltz.ANN[5] * MIXOBJECT.Gases[5].Q[1][I]

        for KGAS in range(Magboltz.NGAS):
            Magboltz.QION[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[2][I] * Magboltz.ANN[KGAS]
            QATT[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[3][I] * Magboltz.ANN[KGAS]
            if MIXOBJECT.Gases[KGAS].NION > 1:
                Magboltz.QION[KGAS][I] = 0.0
                for KION in range(MIXOBJECT.Gases[KGAS].NION):
                    Magboltz.QION[KGAS][I] += MIXOBJECT.Gases[KGAS].QION[KION][I] * Magboltz.ANN[KGAS]
        Magboltz.QREL[I] = 0.0
        Magboltz.QSATT[I] = 0.0
        Magboltz.QSUM[I] = 0.0
        for J in range(Magboltz.NGAS):
            Magboltz.QSUM[I] = Magboltz.QSUM[I] + Magboltz.QION[J][I] + QATT[J][I]
            Magboltz.QSATT[I] = Magboltz.QSATT[I] + QATT[J][I]
            Magboltz.QREL[I] = Magboltz.QREL[I] + Magboltz.QION[J][I] + QATT[J][I]
        for KGAS in range(6):
            for J in range(int(MIXOBJECT.Gases[KGAS].NIN)):
                Magboltz.QSUM[I] = Magboltz.QSUM[I] + MIXOBJECT.Gases[KGAS].QIN[J][I] * Magboltz.ANN[KGAS]
        Magboltz.Mixobject = MIXOBJECT


