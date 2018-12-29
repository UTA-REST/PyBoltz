import Magboltz
import math
from Gasmix import Gasmix
from ANG import ANG


def MIXERT(Magboltz):
    NONE = 1
    ECHARG = 1.602176565e-19
    KEL = [[0 for i in range(6)] for j in range(6)]

    E = [[0 for i in range(6)] for j in range(6)]
    Q = [[[0 for i in range(4000)] for j in range(6)] for g in range(6)]

    PEQEL = [[[0 for i in range(4000)] for j in range(6)] for g in range(6)]

    EION = [[0 for i in range(30)] for j in range(6)]

    EB = [[0 for i in range(30)] for j in range(6)]

    EC0 = [[0 for i in range(30)] for j in range(6)]

    EG1 = [[0 for i in range(30)] for g in range(6)]

    EG2 = [[0 for i in range(30)] for g in range(6)]

    WK = [[0 for i in range(30)] for g in range(6)]

    EFL = [[0 for i in range(30)] for g in range(6)]

    NC0 = [[0 for i in range(30)] for g in range(6)]

    NG1 = [[0 for i in range(30)] for g in range(6)]

    NG2 = [[0 for i in range(30)] for g in range(6)]

    EI = [[0 for i in range(250)] for g in range(6)]

    KIN = [[0 for i in range(250)] for g in range(6)]

    NION = [0 for i in range(6)]

    QION = [[[0 for i in range(4000)] for j in range(30)] for g in range(6)]

    PEQION = [[[0 for i in range(4000)] for j in range(30)] for g in range(6)]

    PEQIN = [[[0 for i in range(4000)] for j in range(250)] for g in range(6)]

    QATT = [[[0 for i in range(4000)] for j in range(8)] for g in range(6)]

    QNULL = [[[0 for i in range(4000)] for j in range(10)] for g in range(6)]

    SCLN = [[0 for i in range(10)] for j in range(6)]

    NATT = [0 for i in range(6)]

    NNULL = [0 for i in range(6)]

    Magboltz.ESTEP = Magboltz.EFINAL / Magboltz.NSTEP

    EHALF = Magboltz.ESTEP / 2

    Magboltz.E[0] = EHALF
    for i in range(1, 4000):
        Magboltz.E[i] = EHALF + Magboltz.ESTEP * i
        Magboltz.EROOT[i] = math.sqrt(Magboltz.E[i])
    Magboltz.EROOT[0] = math.sqrt(EHALF)
    KIN1 = [0 for i in range(250)]
    KIN2 = [0 for i in range(250)]
    KIN3 = [0 for i in range(250)]
    KIN4 = [0 for i in range(250)]
    KIN5 = [0 for i in range(250)]
    KIN6 = [0 for i in range(250)]

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
        for KGAS in range(Magboltz.NGAS):
            NP = 0
            if KGAS == 1:
                Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[0].Q[1][IE] * Magboltz.VANN[0]
            elif KGAS == 2:
                Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[1].Q[1][IE] * Magboltz.VANN[1]
            elif KGAS == 3:
                Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[2].Q[1][IE] * Magboltz.VANN[2]
            elif KGAS == 4:
                Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[3].Q[1][IE] * Magboltz.VANN[3]
            elif KGAS == 5:
                Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[4].Q[1][IE] * Magboltz.VANN[4]
            elif KGAS == 6:
                Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[5].Q[1][IE] * Magboltz.VANN[5]
            Magboltz.PSCT[KGAS][IE][NP] = 0.5
            Magboltz.ANGCT[KGAS][IE][NP] = 1
            Magboltz.INDEX[KGAS][NP] = 0
            ANGOBJECT = ANG()

            if MIXOBJECT.Gases[KGAS].KEL[1] == 1:
                PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                ANGOBJECT.PSCT1 = PSCT1
                ANGOBJECT.ANGCUT()
                Magboltz.ANGCT[KGAS][IE][NP] = ANGOBJECT.ANGC
                Magboltz.PSCT[KGAS][IE][NP] = ANGOBJECT.PSCT2
                Magboltz.INDEX[KGAS][NP] = 1
            elif MIXOBJECT.Gases[KGAS].KEL[1] == 2:
                Magboltz.PSCT[KGAS][IE][NP] = MIXOBJECT.Gases[KGAS].PEQEL[1][IE]
                Magboltz.INDEX[KGAS][NP] = 2

            if IE == 0:
                RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                Magboltz.AMGAS[KGAS] = 2 * EMASS / MIXOBJECT.Gases[KGAS].E[1]
                Magboltz.RGAS[KGAS][NP] = RGAS
                print("1st")
                print(NP)
            L = 1
            Magboltz.IARRY[KGAS][NP] = L
            Magboltz.EIN[KGAS][NP] = 0.0
            Magboltz.IPN[KGAS][NP] = 0

            Magboltz.PENFRA[KGAS][0][NP] = 0.0
            Magboltz.PENFRA[KGAS][1][NP] = 0.0
            Magboltz.PENFRA[KGAS][2][NP] = 0.0

            #IONISATION
            print(Magboltz.EFINAL)
            print(MIXOBJECT.Gases[KGAS].E[2])
            if Magboltz.EFINAL >= MIXOBJECT.Gases[KGAS].E[2]:
                if MIXOBJECT.Gases[KGAS].NION <= 1:
                    NP+=1
                    Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[KGAS].Q[2][IE]
                    Magboltz.FCION[IE] = Magboltz.FCION[IE] + Magboltz.CF[KGAS][IE][NP]
                    Magboltz.PSCT[KGAS][IE][NP] = 0.5
                    Magboltz.ANGCT[KGAS][IE][NP] = 1.0
                    Magboltz.INDEX[KGAS][NP] = 0
                    if MIXOBJECT.Gases[KGAS].KEL[2] == 1:
                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        Magboltz.ANGCT[KGAS][IE][NP] = ANGOBJECT.ANGC
                        Magboltz.PSCT[KGAS][IE][NP] = ANGOBJECT.PSCT2
                        Magboltz.INDEX[KGAS][NP] = 1
                    elif MIXOBJECT.Gases[KGAS].KEL[2] == 2:
                        Magboltz.PSCT[KGAS][IE][NP] = MIXOBJECT.Gases[KGAS].PEQEL[2][IE]
                        Magboltz.INDEX[KGAS][NP] = 2
                elif MIXOBJECT.Gases[KGAS].NION > 1:
                    for KION in range(MIXOBJECT.NION[KGAS]):
                        NP+= 1
                        print(NP)
                        Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[KGAS].QION[KION][IE]
                        Magboltz.FCION[IE] = Magboltz.FCION[IE] + Magboltz.CF[KGAS][IE][NP]
                        Magboltz.PSCT[KGAS][IE][NP] = 0.5
                        Magboltz.ANGCT[KGAS][IE][NP] = 1.0
                        Magboltz.INDEX[KGAS][NP] = 0
                        if MIXOBJECT.Gases[0].KEL[2] == 1:
                            PSCT1 = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            ANGOBJECT.PSCT1 = PSCT1
                            ANGOBJECT.ANGCUT()
                            Magboltz.ANGCT[KGAS][IE][NP] = ANGOBJECT.ANGC
                            Magboltz.PSCT[KGAS][IE][NP] = ANGOBJECT.PSCT2
                            Magboltz.INDEX[KGAS][NP] = 1
                        elif MIXOBJECT.Gases[0].KEL[2] == 2:
                            Magboltz.PSCT[KGAS][IE][NP] = MIXOBJECT.Gases[KGAS].PEQION[KION][IE]
                            Magboltz.INDEX[KGAS][NP] = 2

                if IE == 0:
                    if MIXOBJECT.Gases[KGAS].NION <= 1:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        Magboltz.RGAS[KGAS][NP] = RGAS
                        print(NP)
                        Magboltz.EIN[KGAS][NP] = MIXOBJECT.Gases[KGAS].E[2] / RGAS
                        Magboltz.WPL[KGAS][NP] = MIXOBJECT.Gases[KGAS].EB[0]
                        Magboltz.NC0[KGAS][NP] = MIXOBJECT.Gases[KGAS].NC0[0]
                        Magboltz.EC0[KGAS][NP] = MIXOBJECT.Gases[KGAS].EC0[0]
                        Magboltz.NG1[KGAS][NP] = MIXOBJECT.Gases[KGAS].NG1[0]
                        Magboltz.EG1[KGAS][NP] = MIXOBJECT.Gases[KGAS].EG1[0]
                        Magboltz.NG2[KGAS][NP] = MIXOBJECT.Gases[KGAS].NG2[0]
                        Magboltz.WKLM[KGAS][NP] = MIXOBJECT.Gases[KGAS].WK[1]
                        Magboltz.IPN[KGAS][NP] = 1
                        L = 2
                        Magboltz.IARRY[KGAS][NP] = L
                        Magboltz.PENFRA[KGAS][0][NP] = 0.0
                        Magboltz.PENFRA[KGAS][1][NP] = 0.0
                        Magboltz.PENFRA[KGAS][2][NP] = 0.0
                    elif MIXOBJECT.Gases[KGAS].NION > 1:
                        NP = NP - MIXOBJECT.Gases[KGAS].NION
                        for KION in range(MIXOBJECT.Gases[KGAS].NION):
                            NP = NP + 1
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            Magboltz.RGAS[KGAS][NP] = RGAS
                            print(NP)
                            Magboltz.EIN[KGAS][NP] = MIXOBJECT.Gases[KGAS].EION[KION] / RGAS
                            Magboltz.WPL[KGAS][NP] = MIXOBJECT.Gases[KGAS].EB[KION]
                            Magboltz.NC0[KGAS][NP] = MIXOBJECT.Gases[KGAS].NC0[KION]
                            Magboltz.EC0[KGAS][NP] = MIXOBJECT.Gases[KGAS].EC0[KION]
                            Magboltz.NG1[KGAS][NP] = MIXOBJECT.Gases[KGAS].NG1[KION]
                            Magboltz.EG1[KGAS][NP] = MIXOBJECT.Gases[KGAS].EG1[KION]
                            Magboltz.NG2[KGAS][NP] = MIXOBJECT.Gases[KGAS].NG2[KION]
                            Magboltz.WKLM[KGAS][NP] = MIXOBJECT.Gases[KGAS].WK[KION]
                            Magboltz.IPN[KGAS][NP] = 1
                            L = 2
                            Magboltz.IARRY[KGAS][NP] = L
                            Magboltz.PENFRA[KGAS][0][NP] = 0.0
                            Magboltz.PENFRA[KGAS][1][NP] = 0.0
                            Magboltz.PENFRA[KGAS][2][NP] = 0.0


            if Magboltz.EFINAL >= MIXOBJECT.Gases[KGAS].E[3]:
                if MIXOBJECT.Gases[KGAS].NATT <= 1:
                    NP+=1
                    Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[KGAS].Q[3][IE] * Magboltz.VANN[KGAS]
                    Magboltz.FCATT[IE] = Magboltz.FCATT[IE] + Magboltz.CF[KGAS][IE][NP]
                    Magboltz.PSCT[KGAS][IE][NP] = 0.5
                    Magboltz.ANGCT[KGAS][IE][NP] = 1.0
                elif MIXOBJECT.Gases[KGAS].NATT > 1:
                    for JJ in range(MIXOBJECT.Gases[KGAS].NATT):
                        NP+=1
                        Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[KGAS].QATT[JJ][IE] * Magboltz.VANN[KGAS]
                        Magboltz.FCATT[IE] = Magboltz.FCATT[IE] + Magboltz.CF[KGAS][IE][NP]
                        Magboltz.PSCT[KGAS][IE][NP] = 0.5
                        Magboltz.ANGCT[KGAS][IE][NP] = 1.0
                        if IE == 0:
                            RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                            Magboltz.RGAS[KGAS][NP] = RGAS
                            Magboltz.EIN[KGAS][NP] = 0.0
                            Magboltz.INDEX[KGAS][NP] = 0
                            Magboltz.IPN[KGAS][NP] = -1
                            L = 3
                            Magboltz.IARRY[KGAS][NP] = L
                            Magboltz.PENFRA[KGAS][0][NP] = 0.0
                            Magboltz.PENFRA[KGAS][1][NP] = 0.0
                            Magboltz.PENFRA[KGAS][2][NP] = 0.0
                if IE == 0:
                    RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                    Magboltz.RGAS[KGAS][NP] = RGAS
                    print(NP)
                    Magboltz.EIN[KGAS][NP] = 0.0
                    Magboltz.INDEX[KGAS][NP] = 0
                    Magboltz.IPN[KGAS][NP] = -1
                    L = 3
                    Magboltz.IARRY[KGAS][NP] = L
                    Magboltz.PENFRA[KGAS][0][NP] = 0.0
                    Magboltz.PENFRA[KGAS][1][NP] = 0.0
                    Magboltz.PENFRA[KGAS][2][NP] = 0.0

            # INELASTIC AND SUPERELASTIC
            if MIXOBJECT.Gases[KGAS].NIN > 0:
                print("NIN ====")
                print (MIXOBJECT.Gases[KGAS].NIN)
                for J in range(MIXOBJECT.Gases[KGAS].NIN):
                    NP = NP + 1
                    Magboltz.CF[KGAS][IE][NP] = MIXOBJECT.Gases[KGAS].QIN[J][IE] * Magboltz.VANN[KGAS]
                    Magboltz.PSCT[KGAS][IE][NP] = 0.5
                    Magboltz.ANGCT[KGAS][IE][NP] = 1.0
                    Magboltz.INDEX[KGAS][NP] = 0
                    if MIXOBJECT.Gases[KGAS].KIN[J] == 1:
                        PSCT1 = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        ANGOBJECT.PSCT1 = PSCT1
                        ANGOBJECT.ANGCUT()
                        Magboltz.ANGCT[KGAS][IE][NP] = ANGOBJECT.ANGC
                        Magboltz.PSCT[KGAS][IE][NP] = ANGOBJECT.PSCT2
                        Magboltz.INDEX[KGAS][NP] = 1
                    elif MIXOBJECT.Gases[KGAS].KIN[J] == 2:
                        Magboltz.PSCT[KGAS][IE][NP] = MIXOBJECT.Gases[KGAS].PEQIN[J][IE]
                        Magboltz.INDEX[KGAS][NP] = 2
                    if IE == 0:
                        RGAS = 1 + MIXOBJECT.Gases[KGAS].E[1] / 2
                        Magboltz.EIN[KGAS][NP] = MIXOBJECT.Gases[KGAS].EI[J] / RGAS
                        L = 4
                        if MIXOBJECT.Gases[KGAS].EI[J] < 0:
                            L = 5
                        Magboltz.IPN[KGAS][NP] = 0
                        Magboltz.IARRY[KGAS][NP] = L
                        Magboltz.PENFRA[KGAS][0][NP] = MIXOBJECT.Gases[KGAS].PENFRA[0][J]
                        Magboltz.PENFRA[KGAS][1][NP] = MIXOBJECT.Gases[KGAS].PENFRA[1][J] * 1.0e-16 / math.sqrt(3)
                        Magboltz.PENFRA[KGAS][2][NP] = MIXOBJECT.Gases[KGAS].PENFRA[2][J]
            Magboltz.IPLAST[KGAS] = NP
            Magboltz.ISIZE[KGAS] = 1
            for I in range(1, 9):
                if Magboltz.IPLAST[KGAS] >= 2 ** I:
                    Magboltz.ISIZE[KGAS] = 2 ** I
                else:
                    break
            # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
            Magboltz.TCF[KGAS][IE] = 0.0
            for IF in range(Magboltz.IPLAST[KGAS]):
                Magboltz.TCF[KGAS][IE] = Magboltz.TCF[KGAS][IE] + Magboltz.CF[KGAS][IE][IF]
                if Magboltz.CF[KGAS][IE][IF] < 0:
                    print("WARNING NEGATIVE COLLISION FREQUENCY")

            for IF in range(Magboltz.IPLAST[KGAS]):
                if Magboltz.TCF[KGAS][IE] == 0:
                    Magboltz.CF[KGAS][IE][IF] = 0
                else:
                    Magboltz.CF[KGAS][IE][IF] = Magboltz.CF[KGAS][IE][IF] / Magboltz.TCF[KGAS][IE]

            for IF in range(1, Magboltz.IPLAST[KGAS]):
                Magboltz.CF[KGAS][IE][IF] = Magboltz.CF[KGAS][IE][IF] + Magboltz.CF[KGAS][IE][IF - 1]
            Magboltz.FCATT[IE] = Magboltz.FCATT[IE] + Magboltz.EROOT[IE]
            Magboltz.FCION[IE] = Magboltz.FCION[IE] + Magboltz.EROOT[IE]
            Magboltz.TCF[KGAS][IE] = Magboltz.TCF[KGAS][IE] + Magboltz.EROOT[IE]
    # CALCULATION OF NULL COLLISION FREQUENCIES
    for IE in range(4000):
        sum = 0
        for i in range(6):
            Magboltz.NPLAST[i] = MIXOBJECT.Gases[i].NNULL
            sum += Magboltz.NPLAST[i]

            if sum == 0:
                break

            if Magboltz.NPLAST[i] > 0:
                for J in range(Magboltz.NPLAST[i]):
                    Magboltz.SCLENUL[i][J] = MIXOBJECT.Gases[i].SCLN[J]
                    Magboltz.CFN[i][J] = MIXOBJECT.Gases[i].QNULL[J] * Magboltz.VANN[i] * Magboltz.SCLENUL[i][J]
            # CALCULATE NULL COLLISrION FREQUENCY FOR EACH GAS COMPONENT

            for KGAS in range(Magboltz.NGAS):
                Magboltz.TCFN[KGAS][IE] = 0.0
                for IL in range(Magboltz.NPLAST[KGAS]):
                    Magboltz.TCFN[KGAS][IE] = Magboltz.TCFN[KGAS][IE] + Magboltz.CFN[KGAS][IE][IL]
                    if Magboltz.CFN[KGAS][IE][IL] < 0:
                        print("WARNING NEGATIVE NULL COLLISION FREQUENCY")
                for IL in range(Magboltz.NPLAST[KGAS]):
                    if Magboltz.TCFN[KGAS][IE] == 0:
                        Magboltz.CFN[KGAS][IE][IL] = 0.0
                    else:
                        Magboltz.CFN[KGAS][IE][IE] = Magboltz.CFN[KGAS][IE][IL] / Magboltz.TCFN[KGAS][IE]

                for IL in range(1, Magboltz.NPLAST[KGAS]):
                    Magboltz.CFN[KGAS][IE][IL] = Magboltz.CFN[KGAS][IE][IL] + Magboltz[KGAS][IE][IL - 1]
                Magboltz.TCFN[KGAS][IE] = Magboltz.TCFN[KGAS][IE] * Magboltz.EROOT[IE]
    KELSUM = 0

    for KGAS in range(Magboltz.NGAS):
        for J in range(6):
            KELSUM += MIXOBJECT.Gases[KGAS].KEL[J]

    for KGAS in range(Magboltz.NGAS):
        for J in range(250):
            KELSUM += MIXOBJECT.Gases[KGAS].KIN[J]

    if KELSUM > 0:
        Magboltz.NISO = 1

    # CALCULATE NULL COLLISION FREQUENCIES FOR EACH GAS COMPONENT
    FAKEIN = abs(Magboltz.FAKEI) / Magboltz.NGAS
    for KGAS in range(Magboltz.NGAS):
        Magboltz.TCFMAX[KGAS] = 0.0
        for IE in range(4000):
            if Magboltz.TCF[KGAS][IE] + Magboltz.TCFN[KGAS][IE] + FAKEIN > Magboltz.TCFMAX[KGAS]:
                Magboltz.TCFMAX[KGAS] = Magboltz.TCF[KGAS][IE] + Magboltz.TCFN[KGAS][IE] + FAKEIN
    # CALCULATE EACH GAS CUMLATIVE FRACTION NULL COLLISION FREQUENCIES
    Magboltz.TCFMX = 0.0
    for KGAS in range(Magboltz.NGAS):
        Magboltz.TCFMX = Magboltz.TCFMX + Magboltz.TCFMAX[KGAS]
    for KGAS in range(Magboltz.NGAS):
        Magboltz.TCFMXG[KGAS] = Magboltz.TCFMAX[KGAS] / Magboltz.TCFMX
    for KGAS in range(1, Magboltz.NGAS):
        Magboltz.TCFMXG[KGAS] = Magboltz.TCFMXG[KGAS] + Magboltz.TCFMXG[KGAS - 1]

    # CALCULATE MAXWELL BOLTZMAN VELOCITY FACTOR FOR EACH GAS COMPONENT

    for KGAS in range(Magboltz.NGAS):
        Magboltz.VTMB[KGAS] = math.sqrt(2.0 * ECHARG * Magboltz.AKT / Magboltz.AMGAS[KGAS]) * 1e-12

    for I in range(Magboltz.NSTEP):
        Magboltz.QTOT[I] = Magboltz.ANN[0] * MIXOBJECT.Gases[0].Q[0][I] + Magboltz.ANN[1] * MIXOBJECT.Gases[1].Q[0][I] + \
                           Magboltz.ANN[2] * MIXOBJECT.Gases[2].Q[0][I] + Magboltz.ANN[3] * MIXOBJECT.Gases[3].Q[0][I] + \
                           Magboltz.ANN[4] * MIXOBJECT.Gases[4].Q[0][I] + Magboltz.ANN[5] * MIXOBJECT.Gases[5].Q[0][I]
        Magboltz.QEL[I] = Magboltz.ANN[0] * MIXOBJECT.Gases[0].Q[1][I] + Magboltz.ANN[1] * MIXOBJECT.Gases[1].Q[1][I] + \
                          Magboltz.ANN[2] * MIXOBJECT.Gases[2].Q[1][I] + Magboltz.ANN[3] * MIXOBJECT.Gases[3].Q[1][I] + \
                          Magboltz.ANN[4] * MIXOBJECT.Gases[4].Q[1][I] + Magboltz.ANN[5] * MIXOBJECT.Gases[5].Q[1][I]

        for KGAS in range(6):
            Magboltz.QION[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[2][I] * Magboltz.ANN[KGAS]
            QATT[KGAS][I] = MIXOBJECT.Gases[KGAS].Q[3][I] * Magboltz.ANN[KGAS]
        Magboltz.QREL[I] = 0.0
        Magboltz.QSATT[I] = 0.0
        Magboltz.QSUM[I] = 0.0
        for J in range(Magboltz.NGAS):
            Magboltz.QSUM[I] = Magboltz.QSUM[I] + Magboltz.QION[J][I] + QATT[J][I]
            Magboltz.QSATT[I] = Magboltz.QSATT[I] + QATT[J][I]
            Magboltz.QREL[I] = Magboltz.QREL[I] + Magboltz.QION[J][I] + QATT[J][I]
        for KGAS in range(6):
            for J in range(MIXOBJECT.Gases[KGAS].NIN):
                Magboltz.QSUM[I] = Magboltz.QSUM[I] + MIXOBJECT.Gases[KGAS].QIN[J][I] * Magboltz.ANN[KGAS]
        Magboltz.Mixobject = MIXOBJECT
        return Magboltz
