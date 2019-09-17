from PyBoltz_mert cimport PyBoltz_mert
from libc.math cimport sin, cos, acos, asin, log, sqrt
from Gasmix_mert cimport Gasmix_mert
from Ang cimport Ang
import sys
import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef MixerT_mert(PyBoltz_mert object):
    """
    This function sets up the given PyBoltz object with the right values for the requested gas mixture. It uses the Gasmix object
    to get the momentum cross sections and all the needed values.

    The object parameter is the PyBoltz object to be setup.
    """
    cdef double AttachmentCrossSection[6][4000]
    cdef Gasmix_mert MixObject
    cdef int  IE, GasIndex, NP, p, sum, J, i, j, KION, JJ, IL, I
    ECHARG = 1.602176565e-19

    object.ElectronEnergyStep = object.FinalElectronEnergy / float(object.EnergySteps)

    EHALF = object.ElectronEnergyStep / 2

    object.E[0] = EHALF
    for i in range(1, 4000):
        object.E[i] = EHALF + object.ElectronEnergyStep * i
        object.SqrtEnergy[i] = sqrt(object.E[i])
    object.SqrtEnergy[0] = sqrt(EHALF)

    MixObject = Gasmix_mert()
    MixObject.InitWithInfo(object.GasIDs, object.InelasticCrossSectionPerGas, object.N_Inelastic, object.PenningFraction,
                           object.E, object.SqrtEnergy, object.TotalCrossSection, object.RelativeIonMinusAttachCrossSection, object.InelasticCrossSection, object.ElasticCrossSection,
                           object.DENSY, 0, object.NumberOfGases, object.EnergySteps, object.WhichAngularModel, object.ElectronEnergyStep,
                           object.FinalElectronEnergy, object.ThermalEnergy, object.RhydbergConst, object.TemperatureCentigrade, object.PressureTorr, object.EnablePenning, object.PIR2)
    MixObject.A = object.A
    MixObject.D = object.D
    MixObject.A1 = object.A1
    MixObject.EV0 = object.EV0
    MixObject.F = object.F
    MixObject.Lambda = object.Lambda
    MixObject.Run()

    #-----------------------------------------------------------------
    #     CALCULATION OF COLLISION FREQUENCIES FOR AN ARRAY OF
    #     ELECTRON ENERGIES IN THE RANGE ZERO TO FinalEnergy
    #
    #     L=1      ELASTIC NTH GAS
    #     L=2      IONISATION NTH GAS
    #     L=3      ATTACHMENT NTH GAS
    #     L=4      INELASTIC NTH GAS
    #     L=5      SUPERELASTIC NTH GAS
    #---------------------------------------------------------------
    ElectronMass = 9.10938291e-31

    for IE in range(4000):
        for GasIndex in range(object.NumberOfGases):
            object.FCION[IE] = 0.0
            object.FCATT[IE] = 0.0
            NP = 1

            object.CollisionFrequency[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].Q[1][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
            object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
            object.AngleCut[GasIndex][IE][NP - 1] = 1
            object.INDEX[GasIndex][NP - 1] = 0
            AngObject = Ang()

            if MixObject.Gases[GasIndex].KEL[1] == 1:
                ScatteringParameter1 = MixObject.Gases[GasIndex].PEElasticCrossSection[1][IE]
                AngObject.ScatteringParameter1 = ScatteringParameter1
                AngObject.AngCut()
                object.AngleCut[GasIndex][IE][NP - 1] = AngObject.ANGC
                object.ScatteringParameter[GasIndex][IE][NP - 1] = AngObject.ScatteringParameter2
                object.INDEX[GasIndex][NP - 1] = 1
            elif MixObject.Gases[GasIndex].KEL[1] == 2:
                object.ScatteringParameter[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].PEElasticCrossSection[1][IE]
                object.INDEX[GasIndex][NP - 1] = 2

            if IE == 0:
                RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                object.AMGAS[GasIndex] = 2 * ElectronMass / MixObject.Gases[GasIndex].E[1]
                object.RGAS[GasIndex][NP - 1] = RGAS
                L = 1
                object.IARRY[GasIndex][NP - 1] = L
                object.EnergyLevels[GasIndex][NP - 1] = 0.0
                object.IPN[GasIndex][NP - 1] = 0

                object.PenningFraction[GasIndex][0][NP - 1] = 0.0
                object.PenningFraction[GasIndex][1][NP - 1] = 0.0
                object.PenningFraction[GasIndex][2][NP - 1] = 0.0

            # IONISATION
            if object.FinalElectronEnergy >= MixObject.Gases[GasIndex].E[2]:
                if MixObject.Gases[GasIndex].N_Ionization <= 1:
                    NP += 1
                    object.CollisionFrequency[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].Q[2][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                    object.FCION[IE] = object.FCION[IE] + object.CollisionFrequency[GasIndex][IE][NP - 1]
                    object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                    object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                    object.INDEX[GasIndex][NP - 1] = 0
                    if MixObject.Gases[GasIndex].KEL[2] == 1:
                        ScatteringParameter1 = MixObject.Gases[GasIndex].PEElasticCrossSection[2][IE]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.AngCut()
                        object.AngleCut[GasIndex][IE][NP - 1] = AngObject.ANGC
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = AngObject.ScatteringParameter2
                        object.INDEX[GasIndex][NP - 1] = 1
                    elif MixObject.Gases[GasIndex].KEL[2] == 2:
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].PEElasticCrossSection[2][IE]
                        object.INDEX[GasIndex][NP - 1] = 2
                elif MixObject.Gases[GasIndex].N_Ionization > 1:
                    for KION in range(MixObject.Gases[GasIndex].N_Ionization):
                        NP += 1
                        object.CollisionFrequency[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].IonizationCrossSection[KION][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                        object.FCION[IE] = object.FCION[IE] + object.CollisionFrequency[GasIndex][IE][NP - 1]
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                        object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                        object.INDEX[GasIndex][NP - 1] = 0
                        if MixObject.Gases[0].KEL[2] == 1:
                            ScatteringParameter1 = MixObject.Gases[GasIndex].PEIonizationCrossSection[KION][IE]
                            AngObject.ScatteringParameter1 = ScatteringParameter1
                            AngObject.AngCut()
                            object.AngleCut[GasIndex][IE][NP - 1] = AngObject.ANGC
                            object.ScatteringParameter[GasIndex][IE][NP - 1] = AngObject.ScatteringParameter2
                            object.INDEX[GasIndex][NP - 1] = 1
                        elif MixObject.Gases[0].KEL[2] == 2:
                            object.ScatteringParameter[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].PEIonizationCrossSection[KION][IE]
                            object.INDEX[GasIndex][NP - 1] = 2
                if IE == 0:
                    if MixObject.Gases[GasIndex].N_Ionization <= 1:
                        RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                        object.RGAS[GasIndex][NP - 1] = RGAS
                        object.EnergyLevels[GasIndex][NP - 1] = MixObject.Gases[GasIndex].E[2] / RGAS
                        object.WPL[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EB[0]
                        object.NC0[GasIndex][NP - 1] = MixObject.Gases[GasIndex].NC0[0]
                        object.EC0[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EC0[0]
                        object.NG1[GasIndex][NP - 1] = MixObject.Gases[GasIndex].NG1[0]
                        object.EG1[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EG1[0]
                        object.NG2[GasIndex][NP - 1] = MixObject.Gases[GasIndex].NG2[0]
                        object.EG2[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EG2[0]
                        object.WKLM[GasIndex][NP - 1] = MixObject.Gases[GasIndex].WK[0]
                        object.EFL[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EFL[0]
                        object.IPN[GasIndex][NP - 1] = 1
                        L = 2
                        object.IARRY[GasIndex][NP - 1] = L
                        object.PenningFraction[GasIndex][0][NP - 1] = 0.0
                        object.PenningFraction[GasIndex][1][NP - 1] = 0.0
                        object.PenningFraction[GasIndex][2][NP - 1] = 0.0
                    elif MixObject.Gases[GasIndex].N_Ionization > 1:
                        NP = NP - MixObject.Gases[GasIndex].N_Ionization
                        for KION in range(MixObject.Gases[GasIndex].N_Ionization):
                            NP = NP + 1
                            RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                            object.RGAS[GasIndex][NP - 1] = RGAS
                            object.EnergyLevels[GasIndex][NP - 1] = MixObject.Gases[GasIndex].IonizationEnergy[KION] / RGAS
                            object.WPL[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EB[KION]
                            object.NC0[GasIndex][NP - 1] = MixObject.Gases[GasIndex].NC0[KION]
                            object.EC0[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EC0[KION]
                            object.EG2[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EG2[KION]
                            object.NG1[GasIndex][NP - 1] = MixObject.Gases[GasIndex].NG1[KION]
                            object.EG1[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EG1[KION]
                            object.NG2[GasIndex][NP - 1] = MixObject.Gases[GasIndex].NG2[KION]
                            object.WKLM[GasIndex][NP - 1] = MixObject.Gases[GasIndex].WK[KION]
                            object.EFL[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EFL[KION]
                            object.IPN[GasIndex][NP - 1] = 1
                            L = 2
                            object.IARRY[GasIndex][NP - 1] = L
                            object.PenningFraction[GasIndex][0][NP - 1] = 0.0
                            object.PenningFraction[GasIndex][1][NP - 1] = 0.0
                            object.PenningFraction[GasIndex][2][NP - 1] = 0.0
            # ATTACHMENT
            if object.FinalElectronEnergy >= MixObject.Gases[GasIndex].E[3]:
                if MixObject.Gases[GasIndex].N_Attachment <= 1:
                    NP += 1
                    object.CollisionFrequency[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].Q[3][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                    object.FCATT[IE] = object.FCATT[IE] + object.CollisionFrequency[GasIndex][IE][NP - 1]
                    object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                    object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                    if IE == 0:
                        RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                        object.RGAS[GasIndex][NP - 1] = RGAS
                        object.EnergyLevels[GasIndex][NP - 1] = 0.0
                        object.INDEX[GasIndex][NP - 1] = 0
                        object.IPN[GasIndex][NP - 1] = -1
                        L = 3
                        object.IARRY[GasIndex][NP - 1] = L
                        object.PenningFraction[GasIndex][0][NP - 1] = 0.0
                        object.PenningFraction[GasIndex][1][NP - 1] = 0.0
                        object.PenningFraction[GasIndex][2][NP - 1] = 0.0

                elif MixObject.Gases[GasIndex].N_Attachment > 1:
                    for JJ in range(int(MixObject.Gases[GasIndex].N_Attachment)):
                        NP += 1
                        object.CollisionFrequency[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].AttachmentCrossSection[JJ][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                        object.FCATT[IE] = object.FCATT[IE] + object.CollisionFrequency[GasIndex][IE][NP - 1]
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                        object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                        if IE == 0:
                            RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                            object.RGAS[GasIndex][NP - 1] = RGAS
                            object.EnergyLevels[GasIndex][NP - 1] = 0.0
                            object.INDEX[GasIndex][NP - 1] = 0
                            object.IPN[GasIndex][NP - 1] = -1
                            L = 3
                            object.IARRY[GasIndex][NP - 1] = L
                            object.PenningFraction[GasIndex][0][NP - 1] = 0.0
                            object.PenningFraction[GasIndex][1][NP - 1] = 0.0
                            object.PenningFraction[GasIndex][2][NP - 1] = 0.0
            # INELASTIC AND SUPERELASTIC
            if MixObject.Gases[GasIndex].N_Inelastic > 0:
                for J in range(int(MixObject.Gases[GasIndex].N_Inelastic)):
                    NP = NP + 1
                    object.CollisionFrequency[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                    object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                    object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                    object.INDEX[GasIndex][NP - 1] = 0
                    if MixObject.Gases[GasIndex].KIN[J] == 1:
                        ScatteringParameter1 = MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][IE]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.AngCut()
                        object.AngleCut[GasIndex][IE][NP - 1] = AngObject.ANGC
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = AngObject.ScatteringParameter2
                        object.INDEX[GasIndex][NP - 1] = 1
                    elif MixObject.Gases[GasIndex].KIN[J] == 2:
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][IE]
                        object.INDEX[GasIndex][NP - 1] = 2
                    if IE == 0:
                        RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                        object.RGAS[GasIndex][NP - 1] = RGAS
                        object.EnergyLevels[GasIndex][NP - 1] = MixObject.Gases[GasIndex].EnergyLevels[J] / RGAS
                        L = 4
                        if MixObject.Gases[GasIndex].EnergyLevels[J] < 0:
                            L = 5
                        object.IPN[GasIndex][NP - 1] = 0
                        object.IARRY[GasIndex][NP - 1] = L
                        object.PenningFraction[GasIndex][0][NP - 1] = MixObject.Gases[GasIndex].PenningFraction[0][J]
                        object.PenningFraction[GasIndex][1][NP - 1] = MixObject.Gases[GasIndex].PenningFraction[1][J] * 1.0e-16 / sqrt(3)
                        object.PenningFraction[GasIndex][2][NP - 1] = MixObject.Gases[GasIndex].PenningFraction[2][J]


            object.NumMomCrossSectionPoints[GasIndex] = NP
            object.ISIZE[GasIndex] = 1
            for I in range(1, 9):
                if object.NumMomCrossSectionPoints[GasIndex] >= 2 ** I:
                    object.ISIZE[GasIndex] = 2 ** I
                else:
                    break
            # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
            object.TotalCollisionFrequency[GasIndex][IE] = 0.0
            for p in range(int(object.NumMomCrossSectionPoints[GasIndex])):
                object.TotalCollisionFrequency[GasIndex][IE] = object.TotalCollisionFrequency[GasIndex][IE] + object.CollisionFrequency[GasIndex][IE][p]
                if object.CollisionFrequency[GasIndex][IE][p] < 0:
                    print ("WARNING NEGATIVE COLLISION FREQUENCY at gas " +str(p)+"  "+ str(IE))

            for p in range(int(object.NumMomCrossSectionPoints[GasIndex])):
                if object.TotalCollisionFrequency[GasIndex][IE] == 0:
                    object.CollisionFrequency[GasIndex][IE][p] = 0
                else:
                    object.CollisionFrequency[GasIndex][IE][p] = object.CollisionFrequency[GasIndex][IE][p] / object.TotalCollisionFrequency[GasIndex][IE]

            for p in range(1, int(object.NumMomCrossSectionPoints[GasIndex])):
                object.CollisionFrequency[GasIndex][IE][p] = object.CollisionFrequency[GasIndex][IE][p] + object.CollisionFrequency[GasIndex][IE][p - 1]
            object.FCATT[IE] = object.FCATT[IE] * object.SqrtEnergy[IE]
            object.FCION[IE] = object.FCION[IE] * object.SqrtEnergy[IE]
            object.TotalCollisionFrequency[GasIndex][IE] = object.TotalCollisionFrequency[GasIndex][IE] * object.SqrtEnergy[IE]

    # CALCULATION OF NULL COLLISION FREQUENCIES
    for IE in range(4000):
        sum = 0
        for i in range(object.NumberOfGases):
            object.NumMomCrossSectionPointsNull[i] = MixObject.Gases[i].N_Null
            sum += int(object.NumMomCrossSectionPointsNull[i])

            if sum == 0:
                break
        for i in range(object.NumberOfGases):
            if object.NumMomCrossSectionPointsNull[i] > 0:
                for J in range(int(object.NumMomCrossSectionPointsNull[i])):
                    object.SCLENUL[i][J] = MixObject.Gases[i].SCLN[J]
                    object.NullCollisionFreq[i][IE][J] = MixObject.Gases[i].NullCrossSection[J][IE] * object.VMoleculesPerCm3PerGas[i] * object.SCLENUL[i][J]

            # CALCULATE NULL COLLISION FREQUENCY FOR EACH GAS COMPONENT

        for GasIndex in range(object.NumberOfGases):
            object.TotalCollisionFrequencyNull[GasIndex][IE] = 0.0
            for IL in range(int(object.NumMomCrossSectionPointsNull[GasIndex])):
                object.TotalCollisionFrequencyNull[GasIndex][IE] = object.TotalCollisionFrequencyNull[GasIndex][IE] + object.NullCollisionFreq[GasIndex][IE][IL]
                if object.NullCollisionFreq[GasIndex][IE][IL] < 0:
                    print "WARNING NEGATIVE NULL COLLISION FREQUENCY"
            for IL in range(int(object.NumMomCrossSectionPointsNull[GasIndex])):
                if object.TotalCollisionFrequencyNull[GasIndex][IE] == 0:
                    object.NullCollisionFreq[GasIndex][IE][IL] = 0.0
                else:
                    object.NullCollisionFreq[GasIndex][IE][IL] = object.NullCollisionFreq[GasIndex][IE][IL] / object.TotalCollisionFrequencyNull[GasIndex][IE]

            for IL in range(1, int(object.NumMomCrossSectionPointsNull[GasIndex])):
                object.NullCollisionFreq[GasIndex][IE][IL] = object.NullCollisionFreq[GasIndex][IE][IL] + object.NullCollisionFreq[GasIndex][IE][IL - 1]
            object.TotalCollisionFrequencyNull[GasIndex][IE] = object.TotalCollisionFrequencyNull[GasIndex][IE] * object.SqrtEnergy[IE]

    KELSum = 0

    for GasIndex in range(object.NumberOfGases):
        for J in range(6):
            KELSum += MixObject.Gases[GasIndex].KEL[J]

    for GasIndex in range(object.NumberOfGases):
        for J in range(250):
            KELSum += MixObject.Gases[GasIndex].KIN[J]

    if KELSum > 0:
        object.AnisotropicDetected = 1

    # CALCULATE NULL COLLISION FREQUENCIES FOR EACH GAS COMPONENT
    tt = 0
    FAKEnergyLevels = abs(object.FakeIonizations) / object.NumberOfGases
    for GasIndex in range(object.NumberOfGases):
        object.MaxCollisionFreq[GasIndex] = 0.0
        for IE in range(4000):
            if object.TotalCollisionFrequency[GasIndex][IE] + object.TotalCollisionFrequencyNull[GasIndex][IE] + FAKEnergyLevels >= object.MaxCollisionFreq[GasIndex]:
                object.MaxCollisionFreq[GasIndex] = object.TotalCollisionFrequency[GasIndex][IE] + object.TotalCollisionFrequencyNull[GasIndex][IE] + FAKEnergyLevels

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
        object.TotalCrossSection[I] = object.MoleculesPerCm3PerGas[0] * MixObject.Gases[0].Q[0][I] + object.MoleculesPerCm3PerGas[1] * MixObject.Gases[1].Q[0][I] + \
                         object.MoleculesPerCm3PerGas[2] * MixObject.Gases[2].Q[0][I] + object.MoleculesPerCm3PerGas[3] * MixObject.Gases[3].Q[0][I] + \
                         object.MoleculesPerCm3PerGas[4] * MixObject.Gases[4].Q[0][I] + object.MoleculesPerCm3PerGas[5] * MixObject.Gases[5].Q[0][I]
        object.ElasticCrossSection[I] = object.MoleculesPerCm3PerGas[0] * MixObject.Gases[0].Q[1][I] + object.MoleculesPerCm3PerGas[1] * MixObject.Gases[1].Q[1][I] + \
                        object.MoleculesPerCm3PerGas[2] * MixObject.Gases[2].Q[1][I] + object.MoleculesPerCm3PerGas[3] * MixObject.Gases[3].Q[1][I] + \
                        object.MoleculesPerCm3PerGas[4] * MixObject.Gases[4].Q[1][I] + object.MoleculesPerCm3PerGas[5] * MixObject.Gases[5].Q[1][I]

        for GasIndex in range(6):
            object.IonizationCrossSection[GasIndex][I] = MixObject.Gases[GasIndex].Q[2][I] * object.MoleculesPerCm3PerGas[GasIndex]
            AttachmentCrossSection[GasIndex][I] = MixObject.Gases[GasIndex].Q[3][I] * object.MoleculesPerCm3PerGas[GasIndex]
        object.RelativeIonMinusAttachCrossSection[I] = 0.0
        object.AttachmentSectionSum[I] = 0.0
        object.CrossSectionSum[I] = 0.0
        for J in range(object.NumberOfGases):
            object.CrossSectionSum[I] = object.CrossSectionSum[I] + object.IonizationCrossSection[J][I] + AttachmentCrossSection[J][I]
            object.AttachmentSectionSum[I] = object.AttachmentSectionSum[I] + AttachmentCrossSection[J][I]
            object.RelativeIonMinusAttachCrossSection[I] = object.RelativeIonMinusAttachCrossSection[I] + object.IonizationCrossSection[J][I] - AttachmentCrossSection[J][I]
        for GasIndex in range(6):
            for J in range(int(MixObject.Gases[GasIndex].N_Inelastic)):
                object.CrossSectionSum[I] = object.CrossSectionSum[I] + MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][I] * object.MoleculesPerCm3PerGas[GasIndex]
    return
