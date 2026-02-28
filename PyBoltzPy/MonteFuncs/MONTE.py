import math
import random
import numpy as np
from PyBoltzPy.MBSorts import MBSort


def run(Object):
    """
    Calculate collision events and update diffusion and velocity.
    No magnetic field, no thermal motion variant.
    Pure Python port of PyBoltz/MonteFuncs/MONTE.pyx.
    """
    Object.VelocityX = 0.0
    Object.VelocityY = 0.0
    Object.VelocityErrorX = 0.0
    Object.VelocityErrorY = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.TimeSum = 0.0

    I = 0
    ST1 = 0.0
    ST2 = 0.0
    SumE2 = 0.0
    SumXX = 0.0
    SumYY = 0.0
    SumZZ = 0.0
    SumVX = 0.0
    SumVY = 0.0
    Z_LastSample = 0.0
    ST_LastSample = 0.0
    ST1_LastSample = 0.0
    ST2_LastSample = 0.0
    SumZZ_LastSample = 0.0
    SumXX_LastSample = 0.0
    SumYY_LastSample = 0.0
    SumVX_LastSample = 0.0
    SumVY_LastSample = 0.0
    SME2_LastSample = 0.0
    TDash = 0.0
    SVX_LastSample = 0.0
    SVY_LastSample = 0.0

    # Arrays storing X,Y,Z,T at every real collision
    CollT = [0.0] * 2000000
    CollX = [0.0] * 2000000
    CollY = [0.0] * 2000000
    CollZ = [0.0] * 2000000

    # Arrays storing drift param estimates from each sample
    Num_Samples = int(Object.Num_Samples)
    DriftVelPerSample = [0.0] * Num_Samples
    MeanEnergyPerSample = [0.0] * Num_Samples
    DiffZZPerSample = [0.0] * Num_Samples
    DiffYYPerSample = [0.0] * Num_Samples
    DiffXXPerSample = [0.0] * Num_Samples

    EBefore = Object.InitialElectronEnergy
    NumDecorLengths = 0
    NumCollisions = 0
    IEXTRA = 0

    TEMP = np.zeros(4000)
    for J in range(4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]
    ABSFAKEI = abs(Object.FakeIonizations)
    Object.FakeIonizations = 0

    # Constants
    BP = Object.EField ** 2 * Object.CONST1
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3
    Sqrt2M = Object.CONST3 * 0.01
    TwoM = Sqrt2M ** 2
    TwoPi = 2.0 * math.pi

    # Initial direction cosines
    DirCosineZ1 = math.cos(Object.AngleFromZ)
    DirCosineX1 = math.sin(Object.AngleFromZ) * math.cos(Object.AngleFromX)
    DirCosineY1 = math.sin(Object.AngleFromZ) * math.sin(Object.AngleFromX)

    VelBefore = Sqrt2M * math.sqrt(EBefore)
    VelXBefore = DirCosineX1 * VelBefore
    VelYBefore = DirCosineY1 * VelBefore
    VelZBefore = DirCosineZ1 * VelBefore

    if Object.Console_Output_Flag:
        print('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format(
            "Velocity", "Position", "Time", "Energy", "DIFXX", "DIFYY", "DIFZZ"))

    CollisionsPerSample = int(Object.MaxNumberOfCollisions / Num_Samples)

    INTEM = 8
    DeltaE = Object.Max_Electron_Energy / float(INTEM)

    for iSample in range(Num_Samples):
        for iCollision in range(CollisionsPerSample):
            while True:
                RandomNum = random.random()
                I = int(EBefore / DeltaE) + 1
                I = min(I, INTEM) - 1
                TLIM = Object.MaxCollisionFreqNT[I]

                T = -1 * math.log(RandomNum) / TLIM + TDash
                Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T
                TDash = T
                AP = DirCosineZ1 * F2 * math.sqrt(EBefore)
                EAfter = EBefore + (AP + BP * T) * T
                iEnergyBin = int(EAfter / Object.ElectronEnergyStep)
                iEnergyBin = min(iEnergyBin, 3999)

                if TEMP[iEnergyBin] > TLIM:
                    TDash += math.log(RandomNum) / TLIM
                    Object.MaxCollisionFreqNT[I] *= 1.05
                    continue

                RandomNum = random.random()
                Test1 = Object.TotalCollisionFrequencyNT[iEnergyBin] / TLIM

                if RandomNum > Test1:
                    Test2 = TEMP[iEnergyBin] / TLIM
                    if RandomNum < Test2:
                        if Object.NumMomCrossSectionPointsNullNT == 0:
                            continue
                        RandomNum = random.random()
                        I = 0
                        while Object.NullCollisionFreqNT[iEnergyBin][I] < RandomNum:
                            I += 1
                        Object.ICOLNNNT[I] += 1
                        continue
                    else:
                        Test3 = (TEMP[iEnergyBin] + ABSFAKEI) / TLIM
                        if RandomNum < Test3:
                            Object.FakeIonizations += 1
                            continue
                        continue
                else:
                    break

            # Real collision
            NumCollisions += 1
            Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * TDash
            TDash = 0.0

            T2 = T * T
            A = AP * T
            B = BP * T2

            SumE2 = SumE2 + T * (EBefore + A / 2.0 + B / 3.0)

            VelocityBefore = Sqrt2M * math.sqrt(EBefore)
            VelXBefore = DirCosineX1 * VelocityBefore
            VelYBefore = DirCosineY1 * VelocityBefore
            VelZBefore = DirCosineZ1 * VelocityBefore

            if Object.BFieldMode == 1:
                VelocityRatio = math.sqrt(EBefore / EAfter)
                DirCosineX2 = DirCosineX1 * VelocityRatio
                DirCosineY2 = DirCosineY1 * VelocityRatio
                DirCosineZ2 = DirCosineZ1 * VelocityRatio + T * F2 / (2.0 * math.sqrt(EAfter))

                A_pos = T * VelocityBefore
                Object.X = Object.X + DirCosineX1 * A_pos
                Object.Y = Object.Y + DirCosineY1 * A_pos
                Object.Z = Object.Z + DirCosineZ1 * A_pos + T2 * F1

            elif Object.BFieldMode == 2:
                VelocityRatio = math.sqrt(EBefore / EAfter)
                WBT = Object.AngularSpeedOfRotation * T
                CosWT = math.cos(WBT)
                SinWT = math.sin(WBT)
                VelAfter = Sqrt2M * math.sqrt(EAfter)
                A_pos = T * VelocityBefore
                DirCosineX2 = (VelXBefore * CosWT - VelYBefore * SinWT) / VelAfter
                DirCosineY2 = (VelYBefore * CosWT + VelXBefore * SinWT) / VelAfter
                DirCosineZ2 = DirCosineZ1 * VelocityRatio + T * F2 / (2.0 * math.sqrt(EAfter))
                Object.X += (VelXBefore * SinWT - VelYBefore * (1 - CosWT)) / Object.AngularSpeedOfRotation
                Object.Y += (VelYBefore * SinWT + VelXBefore * (1 - CosWT)) / Object.AngularSpeedOfRotation
                Object.Z += DirCosineZ1 * A_pos + T2 * F1

            Object.TimeSum = Object.TimeSum + T

            iTimeBin = int(T)
            iTimeBin = min(iTimeBin, 299)
            Object.CollisionTimes[iTimeBin] += 1
            Object.CollisionEnergies[iEnergyBin] += 1
            Object.VelocityZ = Object.Z / Object.TimeSum

            SumVX = SumVX + VelXBefore * VelXBefore * T2
            SumVY = SumVY + VelYBefore * VelYBefore * T2

            if NumDecorLengths != 0:
                CollsToLookBack = 0
                for iCorr in range(int(Object.Decor_Lookbacks)):
                    ST2 += T
                    DecorDistance = NumCollisions + CollsToLookBack
                    if DecorDistance > Object.Decor_Colls:
                        DecorDistance = DecorDistance - Object.Decor_Colls
                    TDiff = Object.TimeSum - CollT[DecorDistance - 1]

                    SumXX = SumXX + ((Object.X - CollX[DecorDistance - 1]) ** 2) * T / TDiff
                    SumYY = SumYY + ((Object.Y - CollY[DecorDistance - 1]) ** 2) * T / TDiff

                    if iSample >= 2:
                        ST1 += T
                        SumZZ = SumZZ + (
                            (Object.Z - CollZ[DecorDistance - 1] - Object.VelocityZ * TDiff) ** 2
                        ) * T / TDiff
                    CollsToLookBack += Object.Decor_Step

            CollX[NumCollisions - 1] = Object.X
            CollY[NumCollisions - 1] = Object.Y
            CollZ[NumCollisions - 1] = Object.Z
            CollT[NumCollisions - 1] = Object.TimeSum

            if NumCollisions >= Object.Decor_Colls:
                NumDecorLengths += 1
                NumCollisions = 0

            # Pick collision type
            RandomNum = random.random()
            I = MBSort(I, RandomNum, iEnergyBin, Object)
            while Object.CollisionFrequencyNT[iEnergyBin][I] < RandomNum:
                I = I + 1

            S1 = Object.RGasNT[I]
            EI = Object.EnergyLevelsNT[I]
            if Object.ElectronNumChangeNT[I] > 0:
                RandomNum = random.random()
                EXTRA = RandomNum * (EAfter - EI)
                EI = EXTRA + EI
                IEXTRA += int(Object.NC0NT[I])

            IPT = int(Object.InteractionTypeNT[I])
            Object.CollisionsPerGasPerTypeNT[IPT - 1] += 1
            Object.ICOLNNT[I] += 1

            if EAfter < EI:
                EI = EAfter - 0.0001

            if Object.Enable_Penning != 0:
                if Object.PenningFractionNT[0][I] != 0:
                    RAN = random.random()
                    if RAN <= Object.PenningFractionNT[0][I]:
                        IEXTRA += 1

            S2 = (S1 ** 2) / (S1 - 1.0)

            RandomNum = random.random()
            if Object.AngularModelNT[I] == 1:
                RandomNum1 = random.random()
                CosTheta = 1.0 - RandomNum * Object.AngleCutNT[iEnergyBin][I]
                if RandomNum1 > Object.ScatteringParameterNT[iEnergyBin][I]:
                    CosTheta = -1 * CosTheta
            elif Object.AngularModelNT[I] == 2:
                EPSI = Object.ScatteringParameterNT[iEnergyBin][I]
                CosTheta = 1 - (2 * RandomNum * (1 - EPSI) / (1 + EPSI * (1 - 2 * RandomNum)))
            else:
                CosTheta = 1 - 2 * RandomNum

            Theta = math.acos(CosTheta)

            RandomNum = random.random()
            Phi = TwoPi * RandomNum
            SinPhi = math.sin(Phi)
            CosPhi = math.cos(Phi)

            ARG1 = max(1 - S1 * EI / EAfter, Object.SmallNumber)
            D = 1.0 - CosTheta * math.sqrt(ARG1)
            U = (S1 - 1) * (S1 - 1) / ARG1

            EBefore = max(EAfter * (1 - EI / (S1 * EAfter) - 2 * D / S2), Object.SmallNumber)
            Q = min(math.sqrt((EAfter / EBefore) * ARG1) / S1, 1.0)

            Object.AngleFromZ = math.asin(Q * math.sin(Theta))
            CosZAngle = math.cos(Object.AngleFromZ)

            if CosTheta < 0 and CosTheta * CosTheta > U:
                CosZAngle = -1 * CosZAngle
            SinZAngle = math.sin(Object.AngleFromZ)
            DirCosineZ2 = min(DirCosineZ2, 1)

            ARGZ = math.sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
            if ARGZ == 0:
                DirCosineZ1 = CosZAngle
                DirCosineX1 = CosPhi * SinZAngle
                DirCosineY1 = SinPhi * SinZAngle
            else:
                DirCosineZ1 = DirCosineZ2 * CosZAngle + ARGZ * SinZAngle * SinPhi
                DirCosineY1 = (DirCosineY2 * CosZAngle +
                               (SinZAngle / ARGZ) * (DirCosineX2 * CosPhi - DirCosineY2 * DirCosineZ2 * SinPhi))
                DirCosineX1 = (DirCosineX2 * CosZAngle -
                               (SinZAngle / ARGZ) * (DirCosineY2 * CosPhi + DirCosineX2 * DirCosineZ2 * SinPhi))

        # End of inner collision loop — put into right units
        Object.VelocityZ *= 1e9
        Object.MeanElectronEnergy = SumE2 / Object.TimeSum
        Object.LongitudinalDiffusion = 0.0

        if Object.AnisotropicDetected == 0:
            Object.DiffusionX = 5e15 * SumVX / Object.TimeSum
            Object.DiffusionY = 5e15 * SumVY / Object.TimeSum
            DiffXXPerSample[iSample] = 5.0e15 * (SumVX - SVX_LastSample) / (Object.TimeSum - ST_LastSample)
            DiffYYPerSample[iSample] = 5.0e15 * (SumVY - SVY_LastSample) / (Object.TimeSum - ST_LastSample)
        else:
            if ST2 != 0.0:
                Object.DiffusionY = 5e15 * SumYY / ST2
                Object.DiffusionX = 5e15 * SumXX / ST2
                DiffXXPerSample[iSample] = 5.0e15 * (SumXX - SumXX_LastSample) / (ST2 - ST2_LastSample)
                DiffYYPerSample[iSample] = 5.0e15 * (SumYY - SumYY_LastSample) / (ST2 - ST2_LastSample)
                print(SumXX, SumYY, SumXX_LastSample, SumYY_LastSample, ST2, ST2_LastSample)
            else:
                DiffXXPerSample[iSample] = 0.0
                DiffYYPerSample[iSample] = 0.0

        if ST1 != 0.0:
            Object.DiffusionZ = 5.0e15 * SumZZ / ST1
            DiffZZPerSample[iSample] = 5.0e15 * (SumZZ - SumZZ_LastSample) / (ST1 - ST1_LastSample)
        else:
            DiffZZPerSample[iSample] = 0.0

        DriftVelPerSample[iSample] = (Object.Z - Z_LastSample) / (Object.TimeSum - ST_LastSample) * 1.0e9
        MeanEnergyPerSample[iSample] = (SumE2 - SME2_LastSample) / (Object.TimeSum - ST_LastSample)

        Z_LastSample = Object.Z
        ST_LastSample = Object.TimeSum
        ST1_LastSample = ST1
        ST2_LastSample = ST2
        SVX_LastSample = SumVX
        SVY_LastSample = SumVY
        SumZZ_LastSample = SumZZ
        SumYY_LastSample = SumYY
        SumXX_LastSample = SumXX
        SME2_LastSample = SumE2

        if Object.Console_Output_Flag:
            print('{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(
                Object.VelocityZ, Object.Z, Object.TimeSum, Object.MeanElectronEnergy,
                Object.DiffusionX, Object.DiffusionY, Object.DiffusionZ))

        if Object.CollisionEnergies[3999] > (1000 * float(iSample + 1)):
            raise ValueError("WARNING ENERGY OUT OF RANGE, INCREASE ELECTRON ENERGY INTEGRATION RANGE")

    # Calculate errors from the Num_Samples data points
    SumV_Samples = 0.0
    SumE_Samples = 0.0
    SumV2_Samples = 0.0
    SumE2_Samples = 0.0
    SumDZZ_Samples = 0.0
    SumDXX_Samples = 0.0
    SumDYY_Samples = 0.0
    SumDXX2_Samples = 0.0
    SumDYY2_Samples = 0.0
    SumDZZ2_Samples = 0.0

    for K in range(Num_Samples):
        SumV_Samples = SumV_Samples + DriftVelPerSample[K]
        SumE_Samples = SumE_Samples + MeanEnergyPerSample[K]
        SumV2_Samples = SumV2_Samples + DriftVelPerSample[K] * DriftVelPerSample[K]
        SumE2_Samples = SumE2_Samples + MeanEnergyPerSample[K] * MeanEnergyPerSample[K]
        SumDXX_Samples = SumDXX_Samples + DiffXXPerSample[K]
        SumDYY_Samples = SumDYY_Samples + DiffYYPerSample[K]
        SumDYY2_Samples = SumDYY2_Samples + DiffYYPerSample[K] ** 2
        SumDXX2_Samples = SumDXX2_Samples + DiffXXPerSample[K] ** 2
        if K >= 2:
            SumDZZ_Samples = SumDZZ_Samples + DiffZZPerSample[K]
            SumDZZ2_Samples += DiffZZPerSample[K] ** 2

    # Guard against zero denominators and negative sqrt args (Cython produces NaN/inf silently)
    def _safe_pct_err(variance_num, denom):
        if denom == 0.0:
            return 0.0
        arg = max(0.0, variance_num)
        return 100 * math.sqrt(arg) / denom

    Object.VelocityErrorZ = _safe_pct_err(
        (SumV2_Samples - SumV_Samples ** 2 / Num_Samples) / (Num_Samples - 1), Object.VelocityZ)
    Object.MeanElectronEnergyError = _safe_pct_err(
        (SumE2_Samples - SumE_Samples ** 2 / Num_Samples) / (Num_Samples - 1), Object.MeanElectronEnergy)
    Object.ErrorDiffusionX = _safe_pct_err(
        (SumDXX2_Samples - SumDXX_Samples ** 2 / Num_Samples) / (Num_Samples - 1), Object.DiffusionX)
    Object.ErrorDiffusionY = _safe_pct_err(
        (SumDYY2_Samples - SumDYY_Samples ** 2 / Num_Samples) / (Num_Samples - 1), Object.DiffusionY)
    Object.ErrorDiffusionZ = _safe_pct_err(
        (SumDZZ2_Samples - SumDZZ_Samples ** 2 / Num_Samples - 2) / (Num_Samples - 2 - 1), Object.DiffusionZ)

    Object.VelocityErrorZ = Object.VelocityErrorZ / math.sqrt(Num_Samples)
    Object.MeanElectronEnergyError = Object.MeanElectronEnergyError / math.sqrt(Num_Samples)
    Object.ErrorDiffusionX = Object.ErrorDiffusionX / math.sqrt(Num_Samples)
    Object.ErrorDiffusionY = Object.ErrorDiffusionY / math.sqrt(Num_Samples)
    Object.ErrorDiffusionZ = Object.ErrorDiffusionZ / math.sqrt(Num_Samples - 2)

    Object.LongitudinalDiffusion = Object.DiffusionZ
    Object.TransverseDiffusion = (Object.DiffusionX + Object.DiffusionY) / 2

    # Convert to cm/sec
    Object.VelocityZ *= 1.0e5
    Object.LongitudinalDiffusionError = Object.ErrorDiffusionZ
    Object.TransverseDiffusionError = (Object.ErrorDiffusionX + Object.ErrorDiffusionY) / 2.0

    # Townsend coefficients (uses CollisionsPerGasPerType, same as Cython original)
    Attachment = 0.0
    Ionization = 0.0
    for I in range(Object.NumberOfGases):
        Attachment += Object.CollisionsPerGasPerType[I][2]
        Ionization += Object.CollisionsPerGasPerType[I][1]
    Ionization += IEXTRA

    Object.AttachmentRateError = 0.0
    if Attachment != 0:
        Object.AttachmentRateError = 100 * math.sqrt(Attachment) / Attachment
    Object.AttachmentRate = Attachment / (Object.TimeSum * Object.VelocityZ) * 1e12

    Object.IonisationRateError = 0.0
    if Ionization != 0:
        Object.IonisationRateError = 100 * math.sqrt(Ionization) / Ionization
    Object.IonisationRate = Ionization / (Object.TimeSum * Object.VelocityZ) * 1e12

    return
