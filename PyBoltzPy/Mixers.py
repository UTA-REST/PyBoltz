import math
import numpy as np
from PyBoltzPy.Ang import Ang


def Mixer(object):
    """
    Set up the Boltz object with values for the requested gas mixture.
    Uses the Gasmix object to get momentum cross sections.
    Port of PyBoltz/Mixers.pyx Mixer() for the no-thermal-motion case.
    """
    AttachmentCrossSection = np.zeros((6, 4000))
    ElectronCharge = 1.602176565e-19
    ElectronMass = 9.10938291e-31

    object.ElectronEnergyStep = object.Max_Electron_Energy / object.EnergySteps

    EnergyHalf = object.ElectronEnergyStep / 2
    object.E[0] = EnergyHalf

    for i in range(1, 4000):
        object.E[i] = EnergyHalf + object.ElectronEnergyStep * i
        object.SqrtEnergy[i] = math.sqrt(object.E[i])
    object.SqrtEnergy[0] = math.sqrt(EnergyHalf)

    object.MixObject.InitWithInfo(
        object.GasIDs, object.InelasticCrossSectionPerGas, object.N_Inelastic,
        object.PenningFraction, object.E, object.SqrtEnergy,
        object.NumberOfGases, object.EnergySteps, object.Which_Angular_Model,
        object.ElectronEnergyStep, object.Max_Electron_Energy, object.ThermalEnergy,
        object.TemperatureCentigrade, object.Pressure_Torr, object.PIR2, object.RhydbergConst)
    object.MixObject.Run()

    # Pre-fetch all gas data once (avoids repeated C-struct access from Python)
    gas_data = [object.MixObject.get_gas_data(i) for i in range(6)]

    AngObject = Ang()

    for iEnergy in range(4000):
        object.IonCollisionFreq[iEnergy] = 0.0
        object.AttCollisionFreq[iEnergy] = 0.0
        iProcess = 0
        for GasIndex in range(object.NumberOfGases):
            gas = gas_data[GasIndex]

            object.CollisionFrequencyNT[iEnergy][iProcess] = (
                gas['Q'][1][iEnergy] * object.VMoleculesPerCm3PerGas[GasIndex])
            object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
            object.AngleCutNT[iEnergy][iProcess] = 1
            object.AngularModelNT[iProcess] = 0

            if gas['AngularModel'][1] == 1:
                AngObject.ScatteringParameter1 = gas['PEElasticCrossSection'][1][iEnergy]
                AngObject.CalcAngCut()
                object.AngleCutNT[iEnergy][iProcess] = AngObject.AngCut
                object.ScatteringParameterNT[iEnergy][iProcess] = AngObject.ScatteringParameter2
                object.AngularModelNT[iProcess] = 1
            elif gas['AngularModel'][1] == 2:
                object.ScatteringParameterNT[iEnergy][iProcess] = gas['PEElasticCrossSection'][1][iEnergy]
                object.AngularModelNT[iProcess] = 2

            if iEnergy == 0:
                RGas = 1 + gas['E'][1] / 2
                object.RGasNT[iProcess] = RGas
                object.InteractionTypeNT[iProcess] = 1
                object.EnergyLevelsNT[iProcess] = 0.0
                object.ElectronNumChangeNT[iProcess] = 0
                object.PenningFractionNT[0][iProcess] = 0.0
                object.PenningFractionNT[1][iProcess] = 0.0
                object.PenningFractionNT[2][iProcess] = 0.0

            # IONISATION
            if object.Max_Electron_Energy >= gas['E'][2]:
                if gas['N_Ionization'] <= 1:
                    iProcess += 1
                    object.CollisionFrequencyNT[iEnergy][iProcess] = (
                        gas['Q'][2][iEnergy] * object.VMoleculesPerCm3PerGas[GasIndex])
                    object.IonCollisionFreq[iEnergy] += object.CollisionFrequencyNT[iEnergy][iProcess - 1]
                    object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                    object.AngleCutNT[iEnergy][iProcess] = 1.0
                    object.AngularModelNT[iProcess] = 0
                    if gas['AngularModel'][2] == 1:
                        AngObject.ScatteringParameter1 = gas['PEElasticCrossSection'][2][iEnergy]
                        AngObject.CalcAngCut()
                        object.AngleCutNT[iEnergy][iProcess] = AngObject.AngCut
                        object.ScatteringParameterNT[iEnergy][iProcess] = AngObject.ScatteringParameter2
                        object.AngularModelNT[iProcess] = 1
                    elif gas['AngularModel'][2] == 2:
                        object.ScatteringParameterNT[iEnergy][iProcess] = gas['PEElasticCrossSection'][2][iEnergy]
                        object.AngularModelNT[iProcess] = 2
                elif gas['N_Ionization'] > 1:
                    for iIonizaton in range(gas['N_Ionization']):
                        iProcess += 1
                        object.CollisionFrequencyNT[iEnergy][iProcess] = (
                            gas['IonizationCrossSection'][iIonizaton][iEnergy] *
                            object.VMoleculesPerCm3PerGas[GasIndex])
                        object.IonCollisionFreq[iEnergy] += object.CollisionFrequencyNT[iEnergy][iProcess - 1]
                        object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                        object.AngleCutNT[iEnergy][iProcess] = 1.0
                        object.AngularModelNT[iProcess] = 0
                        if gas['AngularModel'][2] == 1:
                            AngObject.ScatteringParameter1 = gas['PEIonizationCrossSection'][iIonizaton][iEnergy]
                            AngObject.CalcAngCut()
                            object.AngleCutNT[iEnergy][iProcess] = AngObject.AngCut
                            object.ScatteringParameterNT[iEnergy][iProcess] = AngObject.ScatteringParameter2
                            object.AngularModelNT[iProcess] = 1
                        elif gas['AngularModel'][2] == 2:
                            object.ScatteringParameterNT[iEnergy][iProcess] = (
                                gas['PEIonizationCrossSection'][iIonizaton][iEnergy])
                            object.AngularModelNT[iProcess] = 2

                if iEnergy == 0:
                    if gas['N_Ionization'] <= 1:
                        RGas = 1 + gas['E'][1] / 2
                        object.RGasNT[iProcess] = RGas
                        object.EnergyLevelsNT[iProcess] = gas['E'][2] / RGas
                        object.NC0NT[iProcess] = gas['NC0'][0]
                        object.EC0NT[iProcess] = gas['EC0'][0]
                        object.WPLNT[iProcess] = gas['EOBY'][0]
                        object.ElectronNumChangeNT[iProcess] = 1
                        object.InteractionTypeNT[iProcess] = 2
                        object.PenningFractionNT[0][iProcess] = 0.0
                        object.PenningFractionNT[1][iProcess] = 0.0
                        object.PenningFractionNT[2][iProcess] = 0.0
                    elif gas['N_Ionization'] > 1:
                        iProcess = iProcess - gas['N_Ionization']
                        for iIonizaton in range(gas['N_Ionization']):
                            iProcess = iProcess + 1
                            RGas = 1 + gas['E'][1] / 2
                            object.RGasNT[iProcess] = RGas
                            object.EnergyLevelsNT[iProcess] = gas['IonizationEnergy'][iIonizaton] / RGas
                            object.EC0NT[iProcess] = gas['EC0'][iIonizaton]
                            object.NC0NT[iProcess] = gas['NC0'][iIonizaton]
                            object.WPLNT[iProcess] = gas['EOBY'][iIonizaton]
                            object.ElectronNumChangeNT[iProcess] = 1
                            object.InteractionTypeNT[iProcess] = 2
                            object.PenningFractionNT[0][iProcess] = 0.0
                            object.PenningFractionNT[1][iProcess] = 0.0
                            object.PenningFractionNT[2][iProcess] = 0.0

            # ATTACHMENT
            if object.Max_Electron_Energy >= gas['E'][3]:
                if gas['N_Attachment'] <= 1:
                    iProcess += 1
                    object.CollisionFrequencyNT[iEnergy][iProcess] = (
                        gas['Q'][3][iEnergy] * object.VMoleculesPerCm3PerGas[GasIndex])
                    object.AttCollisionFreq[iEnergy] += object.CollisionFrequencyNT[iEnergy][iProcess - 1]
                    object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                    object.AngleCutNT[iEnergy][iProcess] = 1.0
                    if iEnergy == 0:
                        RGas = 1 + gas['E'][1] / 2
                        object.RGasNT[iProcess] = RGas
                        object.EnergyLevelsNT[iProcess] = 0.0
                        object.AngularModelNT[iProcess] = 0
                        object.ElectronNumChangeNT[iProcess] = -1
                        object.InteractionTypeNT[iProcess] = 3
                        object.PenningFractionNT[0][iProcess] = 0.0
                        object.PenningFractionNT[1][iProcess] = 0.0
                        object.PenningFractionNT[2][iProcess] = 0.0
                elif gas['N_Attachment'] > 1:
                    for JJ in range(int(gas['N_Attachment'])):
                        iProcess += 1
                        object.CollisionFrequencyNT[iEnergy][iProcess] = (
                            gas['AttachmentCrossSection'][JJ][iEnergy] *
                            object.VMoleculesPerCm3PerGas[GasIndex])
                        object.AttCollisionFreq[iEnergy] += object.CollisionFrequencyNT[iEnergy][iProcess - 1]
                        object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                        object.AngleCutNT[iEnergy][iProcess] = 1.0
                        if iEnergy == 0:
                            RGas = 1 + gas['E'][1] / 2
                            object.RGasNT[iProcess] = RGas
                            object.EnergyLevelsNT[iProcess] = 0.0
                            object.AngularModelNT[iProcess] = 0
                            object.ElectronNumChangeNT[iProcess] = -1
                            object.InteractionTypeNT[iProcess] = 3
                            object.PenningFractionNT[0][iProcess] = 0.0
                            object.PenningFractionNT[1][iProcess] = 0.0
                            object.PenningFractionNT[2][iProcess] = 0.0

            # INELASTIC AND SUPERELASTIC
            if gas['N_Inelastic'] > 0:
                for J in range(int(gas['N_Inelastic'])):
                    iProcess = iProcess + 1
                    object.CollisionFrequencyNT[iEnergy][iProcess] = (
                        gas['InelasticCrossSectionPerGas'][J][iEnergy] *
                        object.VMoleculesPerCm3PerGas[GasIndex])
                    object.ScatteringParameterNT[iEnergy][iProcess] = 0.5
                    object.AngleCutNT[iEnergy][iProcess] = 1.0
                    object.AngularModelNT[iProcess] = 0
                    if gas['KIN'][J] == 1:
                        AngObject.ScatteringParameter1 = gas['PEInelasticCrossSectionPerGas'][J][iEnergy]
                        AngObject.CalcAngCut()
                        object.AngleCutNT[iEnergy][iProcess] = AngObject.AngCut
                        object.ScatteringParameterNT[iEnergy][iProcess] = AngObject.ScatteringParameter2
                        object.AngularModelNT[iProcess] = 1
                    elif gas['KIN'][J] == 2:
                        object.ScatteringParameterNT[iEnergy][iProcess] = (
                            gas['PEInelasticCrossSectionPerGas'][J][iEnergy])
                        object.AngularModelNT[iProcess] = 2
                    if iEnergy == 0:
                        RGas = 1 + gas['E'][1] / 2
                        object.RGasNT[iProcess] = RGas
                        object.EnergyLevelsNT[iProcess] = gas['EnergyLevels'][J] / RGas
                        L = 4
                        if gas['EnergyLevels'][J] < 0:
                            L = 5
                        object.ElectronNumChangeNT[iProcess] = 0
                        object.InteractionTypeNT[iProcess] = L
                        object.PenningFractionNT[0][iProcess] = gas['PenningFraction'][0][J]
                        object.PenningFractionNT[1][iProcess] = (
                            gas['PenningFraction'][1][J] * 1.0e-16 / math.sqrt(3))
                        object.PenningFractionNT[2][iProcess] = gas['PenningFraction'][2][J]

            iProcess += 1

        object.NumMomCrossSectionPointsNT = iProcess
        object.ISIZENT = 1
        for I in range(1, 9):
            if object.NumMomCrossSectionPointsNT >= 2 ** I:
                object.ISIZENT = 2 ** I
            else:
                break

        # CALCULATION OF TOTAL COLLISION FREQUENCY
        object.TotalCollisionFrequencyNT[iEnergy] = 0.0
        for P in range(int(object.NumMomCrossSectionPointsNT)):
            object.TotalCollisionFrequencyNT[iEnergy] += object.CollisionFrequencyNT[iEnergy][P]
            if object.CollisionFrequencyNT[iEnergy][P] < 0:
                print("WARNING NEGATIVE COLLISION FREQUENCY")

        for P in range(int(object.NumMomCrossSectionPointsNT)):
            if object.TotalCollisionFrequencyNT[iEnergy] != 0.0:
                object.CollisionFrequencyNT[iEnergy][P] /= object.TotalCollisionFrequencyNT[iEnergy]
            else:
                object.CollisionFrequencyNT[iEnergy][P] = 0.0

        for P in range(1, int(object.NumMomCrossSectionPointsNT)):
            object.CollisionFrequencyNT[iEnergy][P] += object.CollisionFrequencyNT[iEnergy][P - 1]
        object.TotalCollisionFrequencyNT[iEnergy] *= object.SqrtEnergy[iEnergy]
        object.IonCollisionFreq[iEnergy] *= object.SqrtEnergy[iEnergy]
        object.AttCollisionFreq[iEnergy] *= object.SqrtEnergy[iEnergy]

        # NULL COLLISION FREQUENCIES (uses GasIndex from last gas — same as original code)
        iProcess = 0
        object.NumMomCrossSectionPointsNullNT = 0
        N_NullSum = 0.0
        for I in range(object.NumberOfGases):
            N_NullSum += gas_data[GasIndex]['N_Null']  # GasIndex is last gas (same bug as original)
        if N_NullSum != 0:
            for I in range(object.NumberOfGases):
                if gas_data[GasIndex]['N_Null'] > 0:
                    for J in range(gas_data[GasIndex]['N_Null']):
                        object.ScaleNullNT[iProcess] = gas_data[GasIndex]['ScaleNull'][J]
                        object.NullCollisionFreqNT[iEnergy][iProcess] = (
                            gas_data[GasIndex]['NullCrossSection'][J][iEnergy] *
                            object.VMoleculesPerCm3PerGas[GasIndex] *
                            object.ScaleNullNT[iProcess])
                        iProcess += 1
            object.NumMomCrossSectionPointsNullNT = iProcess + 1
            object.TotalCollisionFrequencyNullNT[iEnergy] = 0.0
            for P in range(int(object.NumMomCrossSectionPointsNullNT)):
                object.TotalCollisionFrequencyNullNT[iEnergy] += object.NullCollisionFreqNT[iEnergy][P]
                if object.NullCollisionFreqNT[iEnergy][P] < 0:
                    print("WARNING NEGATIVE NULL COLLISION FREQUENCY")

            for P in range(int(object.NumMomCrossSectionPointsNullNT)):
                if object.TotalCollisionFrequencyNullNT[iEnergy] != 0.0:
                    object.NullCollisionFreqNT[iEnergy][P] /= object.TotalCollisionFrequencyNullNT[iEnergy]
                else:
                    object.NullCollisionFreqNT[iEnergy][P] = 0.0

            for P in range(1, int(object.NumMomCrossSectionPointsNullNT)):
                object.NullCollisionFreqNT[iEnergy][P] += object.NullCollisionFreqNT[iEnergy][P - 1]
            object.TotalCollisionFrequencyNullNT[iEnergy] *= object.SqrtEnergy[iEnergy]

    AngularModelSum = 0
    for GasIndex in range(object.NumberOfGases):
        for J in range(6):
            AngularModelSum += gas_data[GasIndex]['AngularModel'][J]

    for GasIndex in range(object.NumberOfGases):
        for J in range(250):
            AngularModelSum += gas_data[GasIndex]['KIN'][J]

    if AngularModelSum > 0:
        object.AnisotropicDetected = 1

    BP = object.EField ** 2 * object.CONST1
    F2 = object.EField * object.CONST3
    EnergyLow = (object.MaxCollisionTime * (object.MaxCollisionTime * BP -
                 F2 * math.sqrt(0.5 * object.Max_Electron_Energy)) /
                 object.ElectronEnergyStep - 1)
    EnergyLow = min(EnergyLow, object.SmallNumber)
    EnergyHigh = (object.MaxCollisionTime * (object.MaxCollisionTime * BP +
                  F2 * math.sqrt(0.5 * object.Max_Electron_Energy)) /
                  object.ElectronEnergyStep + 1)
    if EnergyHigh > 10000:
        EnergyHigh = 10000

    for l in range(8):
        I = l + 1
        JLOW = 4000 - 500 * (9 - I) + 1 + int(EnergyLow)
        JHI = 4000 - 500 * (8 - I) + int(EnergyHigh)
        JLOW = max(JLOW, 0)
        JHI = min(JHI, 4000)
        for J in range(int(JLOW - 1), int(JHI)):
            val = (object.TotalCollisionFrequencyNT[J] +
                   object.TotalCollisionFrequencyNullNT[J] +
                   abs(object.FakeIonizations))
            if val > object.MaxCollisionFreqNT[l]:
                object.MaxCollisionFreqNT[l] = val

    for I in range(object.EnergySteps):
        object.TotalCrossSection[I] = sum(
            object.MoleculesPerCm3PerGas[g] * gas_data[g]['Q'][0][I]
            for g in range(6))
        object.ElasticCrossSection[I] = sum(
            object.MoleculesPerCm3PerGas[g] * gas_data[g]['Q'][1][I]
            for g in range(6))

        for GasIndex in range(object.NumberOfGases):
            object.IonizationCrossSection[GasIndex][I] = (
                gas_data[GasIndex]['Q'][2][I] * object.MoleculesPerCm3PerGas[GasIndex])
            AttachmentCrossSection[GasIndex][I] = (
                gas_data[GasIndex]['Q'][3][I] * object.MoleculesPerCm3PerGas[GasIndex])
            if gas_data[GasIndex]['N_Ionization'] > 1:
                object.IonizationCrossSection[GasIndex][I] = 0.0
                for iIonizaton in range(gas_data[GasIndex]['N_Ionization']):
                    object.IonizationCrossSection[GasIndex][I] += (
                        gas_data[GasIndex]['IonizationCrossSection'][iIonizaton][I] *
                        object.MoleculesPerCm3PerGas[GasIndex])

        object.RelativeIonMinusAttachCrossSection[I] = 0.0
        object.AttachmentSectionSum[I] = 0.0
        object.CrossSectionSum[I] = 0.0
        for J in range(object.NumberOfGases):
            object.CrossSectionSum[I] += (object.IonizationCrossSection[J][I] +
                                          AttachmentCrossSection[J][I])
            object.AttachmentSectionSum[I] += AttachmentCrossSection[J][I]
            object.RelativeIonMinusAttachCrossSection[I] += (
                object.IonizationCrossSection[J][I] - AttachmentCrossSection[J][I])

        for GasIndex in range(6):
            for J in range(int(gas_data[GasIndex]['N_Inelastic'])):
                object.CrossSectionSum[I] += (
                    gas_data[GasIndex]['InelasticCrossSectionPerGas'][J][I] *
                    object.MoleculesPerCm3PerGas[GasIndex])
