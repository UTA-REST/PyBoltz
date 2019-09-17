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
cpdef Mixer_mert(PyBoltz_mert object):
    """
    This function sets up the given PyBoltz object with the right values for the requested gas mixture. It uses the Gasmix object
    to get the momentum cross sections and all the needed values.

    The object parameter is the PyBoltz object to be setup.
    """
    cdef double AttachmentCrossSection[6][4000], ECHARG, JHI, JLOW, EHI, F2, BP, ELOW
    cdef Gasmix_mert MixObject
    cdef int  IE, GasIndex, NP, p, sum, J, i, j, KION, JJ, IL, I
    ECHARG = 1.602176565e-19

    object.ElectronEnergyStep = object.FinalElectronEnergy / object.EnergySteps

    EHALF = object.ElectronEnergyStep / 2

    object.E[0] = EHALF
    for i in range(1, 4000):
        object.E[i] = EHALF + object.ElectronEnergyStep * i
        object.SqrtEnergy[i] = sqrt(object.E[i])
    object.SqrtEnergy[0] = sqrt(EHALF)
    MixObject = Gasmix_mert()
    MixObject.InitWithInfo(object.GasIDs, object.InelasticCrossSectionPerGas, object.N_Inelastic,
                           object.PenningFraction,
                           object.E, object.SqrtEnergy, object.TotalCrossSection,
                           object.RelativeIonMinusAttachCrossSection, object.InelasticCrossSection,
                           object.ElasticCrossSection,
                           object.DENSY, 0, object.NumberOfGases, object.EnergySteps, object.WhichAngularModel,
                           object.ElectronEnergyStep,
                           object.FinalElectronEnergy, object.ThermalEnergy, object.RhydbergConst,
                           object.TemperatureCentigrade, object.PressureTorr, object.EnablePenning, object.PIR2)
    MixObject.A = object.A
    MixObject.D = object.D
    MixObject.A1 = object.A1
    MixObject.EV0 = object.EV0
    MixObject.F = object.F
    MixObject.Lambda = object.Lambda
    MixObject.Run()

    ElectronMass = 9.10938291e-31
    for IE in range(4000):
        NP = 0
        for GasIndex in range(object.NumberOfGases):
            object.CollisionFrequencyNT[IE][NP] = MixObject.Gases[GasIndex].Q[1][IE] * object.VMoleculesPerCm3PerGas[
                GasIndex]
            object.ScatteringParameterNT[IE][NP] = 0.5
            object.AngleCutNT[IE][NP] = 1
            object.INDEXNT[NP] = 0
            AngObject = Ang()

            if MixObject.Gases[GasIndex].KEL[1] == 1:
                ScatteringParameter1 = MixObject.Gases[GasIndex].PEElasticCrossSection[1][IE]
                AngObject.ScatteringParameter1 = ScatteringParameter1
                AngObject.AngCut()
                object.AngleCutNT[IE][NP] = AngObject.ANGC
                object.ScatteringParameterNT[IE][NP] = AngObject.ScatteringParameter2
                object.INDEXNT[NP] = 1
            elif MixObject.Gases[GasIndex].KEL[1] == 2:
                object.ScatteringParameterNT[IE][NP] = MixObject.Gases[GasIndex].PEElasticCrossSection[1][IE]
                object.INDEXNT[NP] = 2

            if IE == 0:
                RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                object.RGASNT[NP] = RGAS
                L = 1
                object.IARRYNT[NP] = L
                object.EnergyLevelsNT[NP] = 0.0
                object.IPNNT[NP] = 0

                object.PenningFractionNT[0][NP] = 0.0
                object.PenningFractionNT[1][NP] = 0.0
                object.PenningFractionNT[2][NP] = 0.0
                # IONISATION

            if object.FinalElectronEnergy >= MixObject.Gases[GasIndex].E[2]:
                if MixObject.Gases[GasIndex].N_Ionization <= 1:
                    NP += 1
                    object.CollisionFrequencyNT[IE][NP] = MixObject.Gases[GasIndex].Q[2][IE] * \
                                                          object.VMoleculesPerCm3PerGas[GasIndex]
                    object.FCION[IE] = object.FCION[IE] + object.CollisionFrequencyNT[IE][NP]
                    object.ScatteringParameterNT[IE][NP] = 0.5
                    object.AngleCutNT[IE][NP] = 1.0
                    object.INDEXNT[NP] = 0
                    if MixObject.Gases[GasIndex].KEL[2] == 1:
                        ScatteringParameter1 = MixObject.Gases[GasIndex].PEElasticCrossSection[2][IE]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.AngCut()
                        object.AngleCutNT[IE][NP] = AngObject.ANGC
                        object.ScatteringParameterNT[IE][NP] = AngObject.ScatteringParameter2
                        object.INDEXNT[NP] = 1
                    elif MixObject.Gases[GasIndex].KEL[2] == 2:
                        object.ScatteringParameterNT[IE][NP] = MixObject.Gases[GasIndex].PEElasticCrossSection[2][IE]
                        object.INDEXNT[NP] = 2
                elif MixObject.Gases[GasIndex].N_Ionization > 1:
                    for KION in range(MixObject.Gases[GasIndex].N_Ionization):
                        NP += 1
                        object.CollisionFrequencyNT[IE][NP] = MixObject.Gases[GasIndex].IonizationCrossSection[KION][
                                                                  IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                        object.FCION[IE] = object.FCION[IE] + object.CollisionFrequencyNT[IE][NP]
                        object.ScatteringParameterNT[IE][NP] = 0.5
                        object.AngleCutNT[IE][NP] = 1.0
                        object.INDEXNT[NP] = 0
                        if MixObject.Gases[GasIndex].KEL[2] == 1:
                            ScatteringParameter1 = MixObject.Gases[GasIndex].PEIonizationCrossSection[KION][IE]
                            AngObject.ScatteringParameter1 = ScatteringParameter1
                            AngObject.AngCut()
                            object.AngleCutNT[IE][NP] = AngObject.ANGC
                            object.ScatteringParameterNT[IE][NP] = AngObject.ScatteringParameter2
                            object.INDEXNT[NP] = 1
                        elif MixObject.Gases[GasIndex].KEL[2] == 2:
                            object.ScatteringParameterNT[IE][NP] = \
                            MixObject.Gases[GasIndex].PEIonizationCrossSection[KION][IE]
                            object.INDEXNT[NP] = 2

                if IE == 0:
                    if MixObject.Gases[GasIndex].N_Ionization <= 1:
                        RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EnergyLevelsNT[NP] = MixObject.Gases[GasIndex].E[2] / RGAS
                        object.WPLNT[NP] = MixObject.Gases[GasIndex].EB[0]
                        object.NC0NT[NP] = MixObject.Gases[GasIndex].NC0[0]
                        object.EC0NT[NP] = MixObject.Gases[GasIndex].EC0[0]
                        object.NG1NT[NP] = MixObject.Gases[GasIndex].NG1[0]
                        object.EG1NT[NP] = MixObject.Gases[GasIndex].EG1[0]
                        object.EG2NT[NP] = MixObject.Gases[GasIndex].EG2[0]
                        object.NG2NT[NP] = MixObject.Gases[GasIndex].NG2[0]
                        object.EFLNT[NP] = MixObject.Gases[GasIndex].EFL[0]
                        object.WKLMNT[NP] = MixObject.Gases[GasIndex].WK[0]
                        object.IPNNT[NP] = 1
                        L = 2
                        object.IARRYNT[NP] = L
                        object.PenningFractionNT[0][NP] = 0.0
                        object.PenningFractionNT[1][NP] = 0.0
                        object.PenningFractionNT[2][NP] = 0.0
                    elif MixObject.Gases[GasIndex].N_Ionization > 1:
                        NP = NP - MixObject.Gases[GasIndex].N_Ionization
                        for KION in range(MixObject.Gases[GasIndex].N_Ionization):
                            NP = NP + 1
                            RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                            object.RGASNT[NP] = RGAS
                            object.EnergyLevelsNT[NP] = MixObject.Gases[GasIndex].IonizationEnergy[KION] / RGAS
                            object.WPLNT[NP] = MixObject.Gases[GasIndex].EB[KION]
                            object.NC0NT[NP] = MixObject.Gases[GasIndex].NC0[KION]
                            object.EC0NT[NP] = MixObject.Gases[GasIndex].EC0[KION]
                            object.NG1NT[NP] = MixObject.Gases[GasIndex].NG1[KION]
                            object.EG2NT[NP] = MixObject.Gases[GasIndex].EG2[KION]
                            object.EFLNT[NP] = MixObject.Gases[GasIndex].EFL[KION]
                            object.EG1NT[NP] = MixObject.Gases[GasIndex].EG1[KION]
                            object.NG2NT[NP] = MixObject.Gases[GasIndex].NG2[KION]
                            object.WKLMNT[NP] = MixObject.Gases[GasIndex].WK[KION]
                            object.IPNNT[NP] = 1
                            L = 2
                            object.IARRYNT[NP] = L
                            object.PenningFractionNT[0][NP] = 0.0
                            object.PenningFractionNT[1][NP] = 0.0
                            object.PenningFractionNT[2][NP] = 0.0

            if object.FinalElectronEnergy >= MixObject.Gases[GasIndex].E[3]:
                if MixObject.Gases[GasIndex].N_Attachment <= 1:
                    NP += 1
                    object.CollisionFrequencyNT[IE][NP] = MixObject.Gases[GasIndex].Q[3][IE] * \
                                                          object.VMoleculesPerCm3PerGas[GasIndex]
                    object.FCATT[IE] = object.FCATT[IE] + object.CollisionFrequencyNT[IE][NP]
                    object.ScatteringParameterNT[IE][NP] = 0.5
                    object.AngleCutNT[IE][NP] = 1.0
                    if IE == 0:
                        RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EnergyLevelsNT[NP] = 0.0
                        object.INDEXNT[NP] = 0
                        object.IPNNT[NP] = -1
                        L = 3
                        object.IARRYNT[NP] = L
                        object.PenningFractionNT[0][NP] = 0.0
                        object.PenningFractionNT[1][NP] = 0.0
                        object.PenningFractionNT[2][NP] = 0.0
                elif MixObject.Gases[GasIndex].N_Attachment > 1:
                    for JJ in range(int(MixObject.Gases[GasIndex].N_Attachment)):
                        NP += 1
                        object.CollisionFrequencyNT[IE][NP] = MixObject.Gases[GasIndex].AttachmentCrossSection[JJ][IE] * \
                                                              object.VMoleculesPerCm3PerGas[GasIndex]
                        object.FCATT[IE] = object.FCATT[IE] + object.CollisionFrequencyNT[IE][NP]
                        object.ScatteringParameterNT[IE][NP] = 0.5
                        object.AngleCutNT[IE][NP] = 1.0
                        if IE == 0:
                            RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                            object.RGASNT[NP] = RGAS
                            object.EnergyLevelsNT[NP] = 0.0
                            object.INDEXNT[NP] = 0
                            object.IPNNT[NP] = -1
                            L = 3
                            object.IARRYNT[NP] = L
                            object.PenningFractionNT[0][NP] = 0.0
                            object.PenningFractionNT[1][NP] = 0.0
                            object.PenningFractionNT[2][NP] = 0.0

            # INELASTIC AND SUPERELASTIC
            if MixObject.Gases[GasIndex].N_Inelastic > 0:
                for J in range(int(MixObject.Gases[GasIndex].N_Inelastic)):
                    NP = NP + 1
                    object.CollisionFrequencyNT[IE][NP] = MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][IE] * \
                                                          object.VMoleculesPerCm3PerGas[GasIndex]
                    object.ScatteringParameterNT[IE][NP] = 0.5
                    object.AngleCutNT[IE][NP] = 1.0
                    object.INDEXNT[NP] = 0
                    if MixObject.Gases[GasIndex].KIN[J] == 1:

                        ScatteringParameter1 = MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][IE]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.AngCut()
                        object.AngleCutNT[IE][NP] = AngObject.ANGC
                        object.ScatteringParameterNT[IE][NP] = AngObject.ScatteringParameter2
                        object.INDEXNT[NP] = 1
                    elif MixObject.Gases[GasIndex].KIN[J] == 2:

                        object.ScatteringParameterNT[IE][NP] = \
                        MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][IE]
                        object.INDEXNT[NP] = 2
                    if IE == 0:

                        RGAS = 1 + MixObject.Gases[GasIndex].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EnergyLevelsNT[NP] = MixObject.Gases[GasIndex].EnergyLevels[J] / RGAS
                        L = 4

                        if MixObject.Gases[GasIndex].EnergyLevels[J] < 0:
                            L = 5
                        object.IPNNT[NP] = 0
                        object.IARRYNT[NP] = L
                        object.PenningFractionNT[0][NP] = MixObject.Gases[GasIndex].PenningFraction[0][J]
                        object.PenningFractionNT[1][NP] = MixObject.Gases[GasIndex].PenningFraction[1][
                                                              J] * 1.0e-16 / sqrt(3)
                        object.PenningFractionNT[2][NP] = MixObject.Gases[GasIndex].PenningFraction[2][J]

            NP += 1

        object.NumMomCrossSectionPointsNT = NP
        object.ISIZENT = 1
        for I in range(1, 9):
            if object.NumMomCrossSectionPointsNT >= 2 ** I:
                object.ISIZENT = 2 ** I
            else:
                break

        # CALCULATION OF TOTAL COLLISION FREQUENCY FOR EACH GAS COMPONENT
        object.TotalCollisionFrequencyNT[IE] = 0.0
        for P in range(int(object.NumMomCrossSectionPointsNT)):
            object.TotalCollisionFrequencyNT[IE] = object.TotalCollisionFrequencyNT[IE] + \
                                                   object.CollisionFrequencyNT[IE][P]
            if object.CollisionFrequencyNT[IE][P] < 0:
                print("WARNING NEGATIVE COLLISION FREQUENCY")

        for P in range(int(object.NumMomCrossSectionPointsNT)):
            if object.TotalCollisionFrequencyNT[IE] != 0.0:
                object.CollisionFrequencyNT[IE][P] /= object.TotalCollisionFrequencyNT[IE]
            else:
                object.CollisionFrequencyNT[IE][P] = 0.0

        for P in range(1, int(object.NumMomCrossSectionPointsNT)):
            object.CollisionFrequencyNT[IE][P] += object.CollisionFrequencyNT[IE][P - 1]
        object.FCATT[IE] *= object.SqrtEnergy[IE]
        object.FCION[IE] *= object.SqrtEnergy[IE]
        object.TotalCollisionFrequencyNT[IE] *= object.SqrtEnergy[IE]

        NP = 0
        object.NumMomCrossSectionPointsNullNT = 0
        N_NullSum = 0.0
        for I in range(object.NumberOfGases):
            N_NullSum += MixObject.Gases[GasIndex].N_Null
        if N_NullSum != 0:
            for I in range(object.NumberOfGases):
                if MixObject.Gases[GasIndex].N_Null > 0:
                    for J in range(MixObject.Gases[GasIndex].N_Null):
                        object.SCLENULNT[NP] = MixObject.Gases[GasIndex].SCLN[J]
                        object.NullCollisionFreqNT[IE][NP] = MixObject.Gases[GasIndex].NullCrossSection[J][IE] * \
                                                             object.VMoleculesPerCm3PerGas[GasIndex] * \
                                                             object.SCLENULNT[NP]
                        NP += 1
            object.NumMomCrossSectionPointsNullNT = NP + 1
            object.TotalCollisionFrequencyNullNT[IE] = 0.0
            for P in range(int(object.NumMomCrossSectionPointsNullNT)):
                object.TotalCollisionFrequencyNullNT[IE] = object.TotalCollisionFrequencyNullNT[IE] + \
                                                           object.NullCollisionFreqNT[IE][P]
                if object.NullCollisionFreqNT[IE][P] < 0:
                    print("WARNING NEGATIVE NULL COLLISION FREQUENCY")

            for P in range(int(object.NumMomCrossSectionPointsNullNT)):
                if object.TotalCollisionFrequencyNullNT[IE] != 0.0:
                    object.NullCollisionFreqNT[IE][P] /= object.TotalCollisionFrequencyNullNT[IE]
                else:
                    object.NullCollisionFreqNT[IE][P] = 0.0

            for P in range(1, int(object.NumMomCrossSectionPointsNullNT)):
                object.NullCollisionFreqNT[IE][P] += object.NullCollisionFreqNT[IE][P - 1]
            object.TotalCollisionFrequencyNullNT[IE] *= object.SqrtEnergy[IE]

    KELSum = 0

    for GasIndex in range(object.NumberOfGases):
        for J in range(6):
            KELSum += MixObject.Gases[GasIndex].KEL[J]

    for GasIndex in range(object.NumberOfGases):
        for J in range(250):
            KELSum += MixObject.Gases[GasIndex].KIN[J]

    if KELSum > 0:
        object.AnisotropicDetected = 1

    BP = object.EField ** 2 * object.CONST1
    F2 = object.EField * object.CONST3
    ELOW = object.MaxCollisionTime * (object.MaxCollisionTime * BP - F2 * sqrt(
        0.5 * object.FinalElectronEnergy)) / object.ElectronEnergyStep - 1
    ELOW = min(ELOW, object.SmallNumber)
    EHI = object.MaxCollisionTime * (object.MaxCollisionTime * BP + F2 * sqrt(
        0.5 * object.FinalElectronEnergy)) / object.ElectronEnergyStep + 1
    if EHI > 10000:
        EHI = 10000
    for l in range(8):
        I = l + 1
        JLOW = 4000 - 500 * (9 - I) + 1 + int(ELOW)
        JHI = 4000 - 500 * (8 - I) + int(EHI)
        JLOW = max(JLOW, 0)
        JHI = min(JHI, 4000)
        for J in range(int(JLOW - 1), int(JHI)):
            if (object.TotalCollisionFrequencyNT[J] + object.TotalCollisionFrequencyNullNT[J] + abs(
                    object.FakeIonizations)) > object.MaxCollisionFreqNT[l]:
                object.MaxCollisionFreqNT[l] = object.TotalCollisionFrequencyNT[J] + \
                                               object.TotalCollisionFrequencyNullNT[J] + abs(object.FakeIonizations)
    for I in range(object.EnergySteps):
        object.TotalCrossSection[I] = object.MoleculesPerCm3PerGas[0] * MixObject.Gases[0].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[1] * MixObject.Gases[1].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[2] * MixObject.Gases[2].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[3] * MixObject.Gases[3].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[4] * MixObject.Gases[4].Q[0][I] + \
                                      object.MoleculesPerCm3PerGas[5] * MixObject.Gases[5].Q[0][I]
        object.ElasticCrossSection[I] = object.MoleculesPerCm3PerGas[0] * MixObject.Gases[0].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[1] * MixObject.Gases[1].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[2] * MixObject.Gases[2].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[3] * MixObject.Gases[3].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[4] * MixObject.Gases[4].Q[1][I] + \
                                        object.MoleculesPerCm3PerGas[5] * MixObject.Gases[5].Q[1][I]

        for GasIndex in range(object.NumberOfGases):
            object.IonizationCrossSection[GasIndex][I] = MixObject.Gases[GasIndex].Q[2][I] * \
                                                         object.MoleculesPerCm3PerGas[GasIndex]
            AttachmentCrossSection[GasIndex][I] = MixObject.Gases[GasIndex].Q[3][I] * object.MoleculesPerCm3PerGas[
                GasIndex]
            if MixObject.Gases[GasIndex].N_Ionization > 1:
                object.IonizationCrossSection[GasIndex][I] = 0.0
                for KION in range(MixObject.Gases[GasIndex].N_Ionization):
                    object.IonizationCrossSection[GasIndex][I] += \
                    MixObject.Gases[GasIndex].IonizationCrossSection[KION][I] * object.MoleculesPerCm3PerGas[GasIndex]
        object.RelativeIonMinusAttachCrossSection[I] = 0.0
        object.AttachmentSectionSum[I] = 0.0
        object.CrossSectionSum[I] = 0.0
        for J in range(object.NumberOfGases):
            object.CrossSectionSum[I] = object.CrossSectionSum[I] + object.IonizationCrossSection[J][I] + \
                                        AttachmentCrossSection[J][I]
            object.AttachmentSectionSum[I] = object.AttachmentSectionSum[I] + AttachmentCrossSection[J][I]
            object.RelativeIonMinusAttachCrossSection[I] = object.RelativeIonMinusAttachCrossSection[I] + \
                                                           object.IonizationCrossSection[J][I] - \
                                                           AttachmentCrossSection[J][I]
        for GasIndex in range(6):
            for J in range(int(MixObject.Gases[GasIndex].N_Inelastic)):
                object.CrossSectionSum[I] = object.CrossSectionSum[I] + \
                                            MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][I] * \
                                            object.MoleculesPerCm3PerGas[GasIndex]
