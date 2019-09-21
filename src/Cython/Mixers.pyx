from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt
from Gasmix cimport Gasmix
from Ang cimport Ang
import sys
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
    cdef double AttachmentCrossSection[6][4000], ECHARG, JHI, JLOW, EHI, F2, BP, ELOW
    cdef int  IE, GasIndex, NP, p, sum, J, i, j, KION, JJ, IL, I
    ECHARG = 1.602176565e-19


    object.ElectronEnergyStep = object.FinalElectronEnergy / object.EnergySteps

    EHALF = object.ElectronEnergyStep / 2

    object.E[0] = EHALF
    for i in range(1, 4000):
        object.E[i] = EHALF + object.ElectronEnergyStep * i
        object.SqrtEnergy[i] = sqrt(object.E[i])
    object.SqrtEnergy[0] = sqrt(EHALF)
    object.MixObject.InitWithInfo(object.GasIDs, object.InelasticCrossSectionPerGas, object.N_Inelastic, object.PenningFraction,
                           object.E, object.SqrtEnergy, object.TotalCrossSection, object.RelativeIonMinusAttachCrossSection, object.InelasticCrossSection, object.ElasticCrossSection,
                           object.DENSY, 0, object.NumberOfGases, object.EnergySteps, object.WhichAngularModel, object.ElectronEnergyStep,
                           object.FinalElectronEnergy, object.ThermalEnergy, object.RhydbergConst, object.TemperatureCentigrade, object.PressureTorr, object.EnablePenning, object.PIR2)
    object.MixObject.Run()

    ElectronMass = 9.10938291e-31
    for IE in range(4000):
        NP = 0
        for GasIndex in range(object.NumberOfGases):
            object.CollisionFrequencyNT[IE][NP] = object.MixObject.Gases[GasIndex].Q[1][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
            object.ScatteringParameterNT[IE][NP] = 0.5
            object.AngleCutNT[IE][NP] = 1
            object.INDEXNT[NP] = 0
            AngObject = Ang()

            if object.MixObject.Gases[GasIndex].KEL[1] == 1:
                ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEElasticCrossSection[1][IE]
                AngObject.ScatteringParameter1 = ScatteringParameter1
                AngObject.CalcAngCut()
                object.AngleCutNT[IE][NP] = AngObject.AngCut
                object.ScatteringParameterNT[IE][NP] = AngObject.ScatteringParameter2
                object.INDEXNT[NP] = 1
            elif object.MixObject.Gases[GasIndex].KEL[1] == 2:
                object.ScatteringParameterNT[IE][NP] = object.MixObject.Gases[GasIndex].PEElasticCrossSection[1][IE]
                object.INDEXNT[NP] = 2

            if IE == 0:
                RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                object.RGASNT[NP] = RGAS
                L = 1
                object.IARRYNT[NP] = L
                object.EnergyLevelsNT[NP] = 0.0
                object.IPNNT[NP] = 0

                object.PenningFractionNT[0][NP] = 0.0
                object.PenningFractionNT[1][NP] = 0.0
                object.PenningFractionNT[2][NP] = 0.0
                # IONISATION

            if object.FinalElectronEnergy >= object.MixObject.Gases[GasIndex].E[2]:
                if object.MixObject.Gases[GasIndex].N_Ionization <= 1:
                    NP += 1
                    object.CollisionFrequencyNT[IE][NP] = object.MixObject.Gases[GasIndex].Q[2][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                    object.FCION[IE] = object.FCION[IE] + object.CollisionFrequencyNT[IE][NP]
                    object.ScatteringParameterNT[IE][NP] = 0.5
                    object.AngleCutNT[IE][NP] = 1.0
                    object.INDEXNT[NP] = 0
                    if object.MixObject.Gases[GasIndex].KEL[2] == 1:
                        ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEElasticCrossSection[2][IE]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.CalcAngCut()
                        object.AngleCutNT[IE][NP] = AngObject.AngCut
                        object.ScatteringParameterNT[IE][NP] = AngObject.ScatteringParameter2
                        object.INDEXNT[NP] = 1
                    elif object.MixObject.Gases[GasIndex].KEL[2] == 2:
                        object.ScatteringParameterNT[IE][NP] = object.MixObject.Gases[GasIndex].PEElasticCrossSection[2][IE]
                        object.INDEXNT[NP] = 2
                elif object.MixObject.Gases[GasIndex].N_Ionization > 1:
                    for KION in range(object.MixObject.Gases[GasIndex].N_Ionization):
                        NP += 1
                        object.CollisionFrequencyNT[IE][NP] = object.MixObject.Gases[GasIndex].IonizationCrossSection[KION][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                        object.FCION[IE] = object.FCION[IE] + object.CollisionFrequencyNT[IE][NP]
                        object.ScatteringParameterNT[IE][NP] = 0.5
                        object.AngleCutNT[IE][NP] = 1.0
                        object.INDEXNT[NP] = 0
                        if object.MixObject.Gases[GasIndex].KEL[2] == 1:
                            ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEIonizationCrossSection[KION][IE]
                            AngObject.ScatteringParameter1 = ScatteringParameter1
                            AngObject.CalcAngCut()
                            object.AngleCutNT[IE][NP] = AngObject.AngCut
                            object.ScatteringParameterNT[IE][NP] = AngObject.ScatteringParameter2
                            object.INDEXNT[NP] = 1
                        elif object.MixObject.Gases[GasIndex].KEL[2] == 2:
                            object.ScatteringParameterNT[IE][NP] = object.MixObject.Gases[GasIndex].PEIonizationCrossSection[KION][IE]
                            object.INDEXNT[NP] = 2

                if IE == 0:
                    if object.MixObject.Gases[GasIndex].N_Ionization <= 1:
                        RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EnergyLevelsNT[NP] = object.MixObject.Gases[GasIndex].E[2] / RGAS
                        object.WPLNT[NP] = object.MixObject.Gases[GasIndex].EB[0]
                        object.NC0NT[NP] = object.MixObject.Gases[GasIndex].NC0[0]
                        object.EC0NT[NP] = object.MixObject.Gases[GasIndex].EC0[0]
                        object.NG1NT[NP] = object.MixObject.Gases[GasIndex].NG1[0]
                        object.EG1NT[NP] = object.MixObject.Gases[GasIndex].EG1[0]
                        object.EG2NT[NP] = object.MixObject.Gases[GasIndex].EG2[0]
                        object.NG2NT[NP] = object.MixObject.Gases[GasIndex].NG2[0]
                        object.EFLNT[NP] = object.MixObject.Gases[GasIndex].EFL[0]
                        object.WKLMNT[NP] = object.MixObject.Gases[GasIndex].WK[0]
                        object.IPNNT[NP] = 1
                        L = 2
                        object.IARRYNT[NP] = L
                        object.PenningFractionNT[0][NP] = 0.0
                        object.PenningFractionNT[1][NP] = 0.0
                        object.PenningFractionNT[2][NP] = 0.0
                    elif object.MixObject.Gases[GasIndex].N_Ionization > 1:
                        NP = NP - object.MixObject.Gases[GasIndex].N_Ionization
                        for KION in range(object.MixObject.Gases[GasIndex].N_Ionization):
                            NP = NP + 1
                            RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                            object.RGASNT[NP] = RGAS
                            object.EnergyLevelsNT[NP] = object.MixObject.Gases[GasIndex].IonizationEnergy[KION] / RGAS
                            object.WPLNT[NP] = object.MixObject.Gases[GasIndex].EB[KION]
                            object.NC0NT[NP] = object.MixObject.Gases[GasIndex].NC0[KION]
                            object.EC0NT[NP] = object.MixObject.Gases[GasIndex].EC0[KION]
                            object.NG1NT[NP] = object.MixObject.Gases[GasIndex].NG1[KION]
                            object.EG2NT[NP] = object.MixObject.Gases[GasIndex].EG2[KION]
                            object.EFLNT[NP] = object.MixObject.Gases[GasIndex].EFL[KION]
                            object.EG1NT[NP] = object.MixObject.Gases[GasIndex].EG1[KION]
                            object.NG2NT[NP] = object.MixObject.Gases[GasIndex].NG2[KION]
                            object.WKLMNT[NP] = object.MixObject.Gases[GasIndex].WK[KION]
                            object.IPNNT[NP] = 1
                            L = 2
                            object.IARRYNT[NP] = L
                            object.PenningFractionNT[0][NP] = 0.0
                            object.PenningFractionNT[1][NP] = 0.0
                            object.PenningFractionNT[2][NP] = 0.0

            if object.FinalElectronEnergy >= object.MixObject.Gases[GasIndex].E[3]:
                if object.MixObject.Gases[GasIndex].N_Attachment <= 1:
                    NP += 1
                    object.CollisionFrequencyNT[IE][NP] = object.MixObject.Gases[GasIndex].Q[3][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                    object.FCATT[IE] = object.FCATT[IE] + object.CollisionFrequencyNT[IE][NP]
                    object.ScatteringParameterNT[IE][NP] = 0.5
                    object.AngleCutNT[IE][NP] = 1.0
                    if IE == 0:
                        RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EnergyLevelsNT[NP] = 0.0
                        object.INDEXNT[NP] = 0
                        object.IPNNT[NP] = -1
                        L = 3
                        object.IARRYNT[NP] = L
                        object.PenningFractionNT[0][NP] = 0.0
                        object.PenningFractionNT[1][NP] = 0.0
                        object.PenningFractionNT[2][NP] = 0.0
                elif object.MixObject.Gases[GasIndex].N_Attachment > 1:
                    for JJ in range(int(object.MixObject.Gases[GasIndex].N_Attachment)):
                        NP += 1
                        object.CollisionFrequencyNT[IE][NP] = object.MixObject.Gases[GasIndex].AttachmentCrossSection[JJ][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                        object.FCATT[IE] = object.FCATT[IE] + object.CollisionFrequencyNT[IE][NP]
                        object.ScatteringParameterNT[IE][NP] = 0.5
                        object.AngleCutNT[IE][NP] = 1.0
                        if IE == 0:
                            RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
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
            if object.MixObject.Gases[GasIndex].N_Inelastic > 0:
                for J in range(int(object.MixObject.Gases[GasIndex].N_Inelastic)):
                    NP = NP + 1
                    object.CollisionFrequencyNT[IE][NP] = object.MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                    object.ScatteringParameterNT[IE][NP] = 0.5
                    object.AngleCutNT[IE][NP] = 1.0
                    object.INDEXNT[NP] = 0
                    if object.MixObject.Gases[GasIndex].KIN[J] == 1:

                        ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][IE]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.CalcAngCut()
                        object.AngleCutNT[IE][NP] = AngObject.AngCut
                        object.ScatteringParameterNT[IE][NP] = AngObject.ScatteringParameter2
                        object.INDEXNT[NP] = 1
                    elif object.MixObject.Gases[GasIndex].KIN[J] == 2:

                        object.ScatteringParameterNT[IE][NP] = object.MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][IE]
                        object.INDEXNT[NP] = 2
                    if IE == 0:

                        RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGASNT[NP] = RGAS
                        object.EnergyLevelsNT[NP] = object.MixObject.Gases[GasIndex].EnergyLevels[J] / RGAS
                        L = 4

                        if object.MixObject.Gases[GasIndex].EnergyLevels[J] < 0:
                            L = 5
                        object.IPNNT[NP] = 0
                        object.IARRYNT[NP] = L
                        object.PenningFractionNT[0][NP] = object.MixObject.Gases[GasIndex].PenningFraction[0][J]
                        object.PenningFractionNT[1][NP] = object.MixObject.Gases[GasIndex].PenningFraction[1][J] * 1.0e-16 / sqrt(3)
                        object.PenningFractionNT[2][NP] = object.MixObject.Gases[GasIndex].PenningFraction[2][J]

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
            object.TotalCollisionFrequencyNT[IE] = object.TotalCollisionFrequencyNT[IE] + object.CollisionFrequencyNT[IE][P]
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
            N_NullSum += object.MixObject.Gases[GasIndex].N_Null
        if N_NullSum != 0:
            for I in range(object.NumberOfGases):
                if object.MixObject.Gases[GasIndex].N_Null > 0:
                    for J in range(object.MixObject.Gases[GasIndex].N_Null):
                        object.SCLENULNT[NP] = object.MixObject.Gases[GasIndex].SCLN[J]
                        object.NullCollisionFreqNT[IE][NP] = object.MixObject.Gases[GasIndex].NullCrossSection[J][IE] * object.VMoleculesPerCm3PerGas[GasIndex] * \
                                               object.SCLENULNT[NP]
                        NP+=1
            object.NumMomCrossSectionPointsNullNT = NP + 1
            object.TotalCollisionFrequencyNullNT[IE] = 0.0
            for P in range(int(object.NumMomCrossSectionPointsNullNT)):
                object.TotalCollisionFrequencyNullNT[IE] = object.TotalCollisionFrequencyNullNT[IE] + object.NullCollisionFreqNT[IE][P]
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
            KELSum += object.MixObject.Gases[GasIndex].KEL[J]

    for GasIndex in range(object.NumberOfGases):
        for J in range(250):
            KELSum += object.MixObject.Gases[GasIndex].KIN[J]

    if KELSum > 0:
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
            if (object.TotalCollisionFrequencyNT[J] + object.TotalCollisionFrequencyNullNT[J] + abs(object.FakeIonizations)) > object.MaxCollisionFreqNT[l]:
                object.MaxCollisionFreqNT[l] = object.TotalCollisionFrequencyNT[J] + object.TotalCollisionFrequencyNullNT[J] + abs(object.FakeIonizations)
    for I in range(object.EnergySteps):
        object.TotalCrossSection[I] = object.MoleculesPerCm3PerGas[0] * object.MixObject.Gases[0].Q[0][I] + object.MoleculesPerCm3PerGas[1] * object.MixObject.Gases[1].Q[0][I] + \
                         object.MoleculesPerCm3PerGas[2] * object.MixObject.Gases[2].Q[0][I] + object.MoleculesPerCm3PerGas[3] * object.MixObject.Gases[3].Q[0][I] + \
                         object.MoleculesPerCm3PerGas[4] * object.MixObject.Gases[4].Q[0][I] + object.MoleculesPerCm3PerGas[5] * object.MixObject.Gases[5].Q[0][I]
        object.ElasticCrossSection[I] = object.MoleculesPerCm3PerGas[0] * object.MixObject.Gases[0].Q[1][I] + object.MoleculesPerCm3PerGas[1] * object.MixObject.Gases[1].Q[1][I] + \
                        object.MoleculesPerCm3PerGas[2] * object.MixObject.Gases[2].Q[1][I] + object.MoleculesPerCm3PerGas[3] * object.MixObject.Gases[3].Q[1][I] + \
                        object.MoleculesPerCm3PerGas[4] * object.MixObject.Gases[4].Q[1][I] + object.MoleculesPerCm3PerGas[5] * object.MixObject.Gases[5].Q[1][I]

        for GasIndex in range(object.NumberOfGases):
            object.IonizationCrossSection[GasIndex][I] = object.MixObject.Gases[GasIndex].Q[2][I] * object.MoleculesPerCm3PerGas[GasIndex]
            AttachmentCrossSection[GasIndex][I] = object.MixObject.Gases[GasIndex].Q[3][I] * object.MoleculesPerCm3PerGas[GasIndex]
            if object.MixObject.Gases[GasIndex].N_Ionization > 1:
                object.IonizationCrossSection[GasIndex][I] = 0.0
                for KION in range(object.MixObject.Gases[GasIndex].N_Ionization):
                    object.IonizationCrossSection[GasIndex][I] += object.MixObject.Gases[GasIndex].IonizationCrossSection[KION][I] * object.MoleculesPerCm3PerGas[GasIndex]
        object.RelativeIonMinusAttachCrossSection[I] = 0.0
        object.AttachmentSectionSum[I] = 0.0
        object.CrossSectionSum[I] = 0.0
        for J in range(object.NumberOfGases):
            object.CrossSectionSum[I] = object.CrossSectionSum[I] + object.IonizationCrossSection[J][I] + AttachmentCrossSection[J][I]
            object.AttachmentSectionSum[I] = object.AttachmentSectionSum[I] + AttachmentCrossSection[J][I]
            object.RelativeIonMinusAttachCrossSection[I] = object.RelativeIonMinusAttachCrossSection[I] + object.IonizationCrossSection[J][I] - AttachmentCrossSection[J][I]
        for GasIndex in range(6):
            for J in range(int(object.MixObject.Gases[GasIndex].N_Inelastic)):
                object.CrossSectionSum[I] = object.CrossSectionSum[I] + object.MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][I] * object.MoleculesPerCm3PerGas[GasIndex]


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
    cdef double AttachmentCrossSection[6][4000]
    cdef int  IE, GasIndex, NP, p, sum, J, i, j, KION, JJ, IL, I
    ECHARG = 1.602176565e-19

    object.ElectronEnergyStep = object.FinalElectronEnergy / float(object.EnergySteps)

    EHALF = object.ElectronEnergyStep / 2

    object.E[0] = EHALF
    for i in range(1, 4000):
        object.E[i] = EHALF + object.ElectronEnergyStep * i
        object.SqrtEnergy[i] = sqrt(object.E[i])
    object.SqrtEnergy[0] = sqrt(EHALF)

    object.MixObject = Gasmix()
    object.MixObject.InitWithInfo(object.GasIDs, object.InelasticCrossSectionPerGas, object.N_Inelastic, object.PenningFraction,
                           object.E, object.SqrtEnergy, object.TotalCrossSection, object.RelativeIonMinusAttachCrossSection, object.InelasticCrossSection, object.ElasticCrossSection,
                           object.DENSY, 0, object.NumberOfGases, object.EnergySteps, object.WhichAngularModel, object.ElectronEnergyStep,
                           object.FinalElectronEnergy, object.ThermalEnergy, object.RhydbergConst, object.TemperatureCentigrade, object.PressureTorr, object.EnablePenning, object.PIR2)
    object.MixObject.Run()

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

            object.CollisionFrequency[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].Q[1][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
            object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
            object.AngleCut[GasIndex][IE][NP - 1] = 1
            object.INDEX[GasIndex][NP - 1] = 0
            AngObject = Ang()

            if object.MixObject.Gases[GasIndex].KEL[1] == 1:
                ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEElasticCrossSection[1][IE]
                AngObject.ScatteringParameter1 = ScatteringParameter1
                AngObject.CalcAngCut()
                object.AngleCut[GasIndex][IE][NP - 1] = AngObject.AngCut
                object.ScatteringParameter[GasIndex][IE][NP - 1] = AngObject.ScatteringParameter2
                object.INDEX[GasIndex][NP - 1] = 1
            elif object.MixObject.Gases[GasIndex].KEL[1] == 2:
                object.ScatteringParameter[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].PEElasticCrossSection[1][IE]
                object.INDEX[GasIndex][NP - 1] = 2

            if IE == 0:
                RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                object.AMGAS[GasIndex] = 2 * ElectronMass / object.MixObject.Gases[GasIndex].E[1]
                object.RGAS[GasIndex][NP - 1] = RGAS
                L = 1
                object.IARRY[GasIndex][NP - 1] = L
                object.EnergyLevels[GasIndex][NP - 1] = 0.0
                object.IPN[GasIndex][NP - 1] = 0

                object.PenningFraction[GasIndex][0][NP - 1] = 0.0
                object.PenningFraction[GasIndex][1][NP - 1] = 0.0
                object.PenningFraction[GasIndex][2][NP - 1] = 0.0

            # IONISATION
            if object.FinalElectronEnergy >= object.MixObject.Gases[GasIndex].E[2]:
                if object.MixObject.Gases[GasIndex].N_Ionization <= 1:
                    NP += 1
                    object.CollisionFrequency[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].Q[2][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                    object.FCION[IE] = object.FCION[IE] + object.CollisionFrequency[GasIndex][IE][NP - 1]
                    object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                    object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                    object.INDEX[GasIndex][NP - 1] = 0
                    if object.MixObject.Gases[GasIndex].KEL[2] == 1:
                        ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEElasticCrossSection[2][IE]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.CalcAngCut()
                        object.AngleCut[GasIndex][IE][NP - 1] = AngObject.AngCut
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = AngObject.ScatteringParameter2
                        object.INDEX[GasIndex][NP - 1] = 1
                    elif object.MixObject.Gases[GasIndex].KEL[2] == 2:
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].PEElasticCrossSection[2][IE]
                        object.INDEX[GasIndex][NP - 1] = 2
                elif object.MixObject.Gases[GasIndex].N_Ionization > 1:
                    for KION in range(object.MixObject.Gases[GasIndex].N_Ionization):
                        NP += 1
                        object.CollisionFrequency[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].IonizationCrossSection[KION][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                        object.FCION[IE] = object.FCION[IE] + object.CollisionFrequency[GasIndex][IE][NP - 1]
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                        object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                        object.INDEX[GasIndex][NP - 1] = 0
                        if object.MixObject.Gases[0].KEL[2] == 1:
                            ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEIonizationCrossSection[KION][IE]
                            AngObject.ScatteringParameter1 = ScatteringParameter1
                            AngObject.CalcAngCut()
                            object.AngleCut[GasIndex][IE][NP - 1] = AngObject.AngCut
                            object.ScatteringParameter[GasIndex][IE][NP - 1] = AngObject.ScatteringParameter2
                            object.INDEX[GasIndex][NP - 1] = 1
                        elif object.MixObject.Gases[0].KEL[2] == 2:
                            object.ScatteringParameter[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].PEIonizationCrossSection[KION][IE]
                            object.INDEX[GasIndex][NP - 1] = 2
                if IE == 0:
                    if object.MixObject.Gases[GasIndex].N_Ionization <= 1:
                        RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGAS[GasIndex][NP - 1] = RGAS
                        object.EnergyLevels[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].E[2] / RGAS
                        object.WPL[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EB[0]
                        object.NC0[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].NC0[0]
                        object.EC0[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EC0[0]
                        object.NG1[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].NG1[0]
                        object.EG1[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EG1[0]
                        object.NG2[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].NG2[0]
                        object.EG2[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EG2[0]
                        object.WKLM[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].WK[0]
                        object.EFL[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EFL[0]
                        object.IPN[GasIndex][NP - 1] = 1
                        L = 2
                        object.IARRY[GasIndex][NP - 1] = L
                        object.PenningFraction[GasIndex][0][NP - 1] = 0.0
                        object.PenningFraction[GasIndex][1][NP - 1] = 0.0
                        object.PenningFraction[GasIndex][2][NP - 1] = 0.0
                    elif object.MixObject.Gases[GasIndex].N_Ionization > 1:
                        NP = NP - object.MixObject.Gases[GasIndex].N_Ionization
                        for KION in range(object.MixObject.Gases[GasIndex].N_Ionization):
                            NP = NP + 1
                            RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                            object.RGAS[GasIndex][NP - 1] = RGAS
                            object.EnergyLevels[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].IonizationEnergy[KION] / RGAS
                            object.WPL[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EB[KION]
                            object.NC0[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].NC0[KION]
                            object.EC0[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EC0[KION]
                            object.EG2[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EG2[KION]
                            object.NG1[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].NG1[KION]
                            object.EG1[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EG1[KION]
                            object.NG2[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].NG2[KION]
                            object.WKLM[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].WK[KION]
                            object.EFL[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EFL[KION]
                            object.IPN[GasIndex][NP - 1] = 1
                            L = 2
                            object.IARRY[GasIndex][NP - 1] = L
                            object.PenningFraction[GasIndex][0][NP - 1] = 0.0
                            object.PenningFraction[GasIndex][1][NP - 1] = 0.0
                            object.PenningFraction[GasIndex][2][NP - 1] = 0.0
            # ATTACHMENT
            if object.FinalElectronEnergy >= object.MixObject.Gases[GasIndex].E[3]:
                if object.MixObject.Gases[GasIndex].N_Attachment <= 1:
                    NP += 1
                    object.CollisionFrequency[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].Q[3][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                    object.FCATT[IE] = object.FCATT[IE] + object.CollisionFrequency[GasIndex][IE][NP - 1]
                    object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                    object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                    if IE == 0:
                        RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGAS[GasIndex][NP - 1] = RGAS
                        object.EnergyLevels[GasIndex][NP - 1] = 0.0
                        object.INDEX[GasIndex][NP - 1] = 0
                        object.IPN[GasIndex][NP - 1] = -1
                        L = 3
                        object.IARRY[GasIndex][NP - 1] = L
                        object.PenningFraction[GasIndex][0][NP - 1] = 0.0
                        object.PenningFraction[GasIndex][1][NP - 1] = 0.0
                        object.PenningFraction[GasIndex][2][NP - 1] = 0.0

                elif object.MixObject.Gases[GasIndex].N_Attachment > 1:
                    for JJ in range(int(object.MixObject.Gases[GasIndex].N_Attachment)):
                        NP += 1
                        object.CollisionFrequency[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].AttachmentCrossSection[JJ][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                        object.FCATT[IE] = object.FCATT[IE] + object.CollisionFrequency[GasIndex][IE][NP - 1]
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                        object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                        if IE == 0:
                            RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
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
            if object.MixObject.Gases[GasIndex].N_Inelastic > 0:
                for J in range(int(object.MixObject.Gases[GasIndex].N_Inelastic)):
                    NP = NP + 1
                    object.CollisionFrequency[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][IE] * object.VMoleculesPerCm3PerGas[GasIndex]
                    object.ScatteringParameter[GasIndex][IE][NP - 1] = 0.5
                    object.AngleCut[GasIndex][IE][NP - 1] = 1.0
                    object.INDEX[GasIndex][NP - 1] = 0
                    if object.MixObject.Gases[GasIndex].KIN[J] == 1:
                        ScatteringParameter1 = object.MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][IE]
                        AngObject.ScatteringParameter1 = ScatteringParameter1
                        AngObject.CalcAngCut()
                        object.AngleCut[GasIndex][IE][NP - 1] = AngObject.AngCut
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = AngObject.ScatteringParameter2
                        object.INDEX[GasIndex][NP - 1] = 1
                    elif object.MixObject.Gases[GasIndex].KIN[J] == 2:
                        object.ScatteringParameter[GasIndex][IE][NP - 1] = object.MixObject.Gases[GasIndex].PEInelasticCrossSectionPerGas[J][IE]
                        object.INDEX[GasIndex][NP - 1] = 2
                    if IE == 0:
                        RGAS = 1 + object.MixObject.Gases[GasIndex].E[1] / 2
                        object.RGAS[GasIndex][NP - 1] = RGAS
                        object.EnergyLevels[GasIndex][NP - 1] = object.MixObject.Gases[GasIndex].EnergyLevels[J] / RGAS
                        L = 4
                        if object.MixObject.Gases[GasIndex].EnergyLevels[J] < 0:
                            L = 5
                        object.IPN[GasIndex][NP - 1] = 0
                        object.IARRY[GasIndex][NP - 1] = L
                        object.PenningFraction[GasIndex][0][NP - 1] = object.MixObject.Gases[GasIndex].PenningFraction[0][J]
                        object.PenningFraction[GasIndex][1][NP - 1] = object.MixObject.Gases[GasIndex].PenningFraction[1][J] * 1.0e-16 / sqrt(3)
                        object.PenningFraction[GasIndex][2][NP - 1] = object.MixObject.Gases[GasIndex].PenningFraction[2][J]


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
            object.NumMomCrossSectionPointsNull[i] = object.MixObject.Gases[i].N_Null
            sum += int(object.NumMomCrossSectionPointsNull[i])

            if sum == 0:
                break
        for i in range(object.NumberOfGases):
            if object.NumMomCrossSectionPointsNull[i] > 0:
                for J in range(int(object.NumMomCrossSectionPointsNull[i])):
                    object.SCLENUL[i][J] = object.MixObject.Gases[i].SCLN[J]
                    object.NullCollisionFreq[i][IE][J] = object.MixObject.Gases[i].NullCrossSection[J][IE] * object.VMoleculesPerCm3PerGas[i] * object.SCLENUL[i][J]

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
            KELSum += object.MixObject.Gases[GasIndex].KEL[J]

    for GasIndex in range(object.NumberOfGases):
        for J in range(250):
            KELSum += object.MixObject.Gases[GasIndex].KIN[J]

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
        object.TotalCrossSection[I] = object.MoleculesPerCm3PerGas[0] * object.MixObject.Gases[0].Q[0][I] + object.MoleculesPerCm3PerGas[1] * object.MixObject.Gases[1].Q[0][I] + \
                         object.MoleculesPerCm3PerGas[2] * object.MixObject.Gases[2].Q[0][I] + object.MoleculesPerCm3PerGas[3] * object.MixObject.Gases[3].Q[0][I] + \
                         object.MoleculesPerCm3PerGas[4] * object.MixObject.Gases[4].Q[0][I] + object.MoleculesPerCm3PerGas[5] * object.MixObject.Gases[5].Q[0][I]
        object.ElasticCrossSection[I] = object.MoleculesPerCm3PerGas[0] * object.MixObject.Gases[0].Q[1][I] + object.MoleculesPerCm3PerGas[1] * object.MixObject.Gases[1].Q[1][I] + \
                        object.MoleculesPerCm3PerGas[2] * object.MixObject.Gases[2].Q[1][I] + object.MoleculesPerCm3PerGas[3] * object.MixObject.Gases[3].Q[1][I] + \
                        object.MoleculesPerCm3PerGas[4] * object.MixObject.Gases[4].Q[1][I] + object.MoleculesPerCm3PerGas[5] * object.MixObject.Gases[5].Q[1][I]

        for GasIndex in range(6):
            object.IonizationCrossSection[GasIndex][I] = object.MixObject.Gases[GasIndex].Q[2][I] * object.MoleculesPerCm3PerGas[GasIndex]
            AttachmentCrossSection[GasIndex][I] = object.MixObject.Gases[GasIndex].Q[3][I] * object.MoleculesPerCm3PerGas[GasIndex]
        object.RelativeIonMinusAttachCrossSection[I] = 0.0
        object.AttachmentSectionSum[I] = 0.0
        object.CrossSectionSum[I] = 0.0
        for J in range(object.NumberOfGases):
            object.CrossSectionSum[I] = object.CrossSectionSum[I] + object.IonizationCrossSection[J][I] + AttachmentCrossSection[J][I]
            object.AttachmentSectionSum[I] = object.AttachmentSectionSum[I] + AttachmentCrossSection[J][I]
            object.RelativeIonMinusAttachCrossSection[I] = object.RelativeIonMinusAttachCrossSection[I] + object.IonizationCrossSection[J][I] - AttachmentCrossSection[J][I]
        for GasIndex in range(6):
            for J in range(int(object.MixObject.Gases[GasIndex].N_Inelastic)):
                object.CrossSectionSum[I] = object.CrossSectionSum[I] + object.MixObject.Gases[GasIndex].InelasticCrossSectionPerGas[J][I] * object.MoleculesPerCm3PerGas[GasIndex]
    return
