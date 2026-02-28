import math
import numpy as np
import numba


@numba.njit(cache=True)
def _mbsort(I, R2, IE, CollisionFreqNT, NumMomCrossSectionPointsNT, ISIZENT):
    iStep = int(ISIZENT)
    Increment = 0
    for K in range(12):
        I = Increment
        if iStep == 2:
            if I == 0:
                return I
            return I - 1
        I = Increment + iStep
        if I <= NumMomCrossSectionPointsNT:
            if CollisionFreqNT[IE, I - 1] < R2:
                Increment = Increment + iStep
        iStep = int(iStep / 2)
    if I == 0:
        return I
    return I - 1


@numba.njit(cache=True)
def _safe_pct_err(variance_num, denom):
    if denom == 0.0:
        return 0.0
    arg = variance_num if variance_num > 0.0 else 0.0
    return 100.0 * math.sqrt(arg) / denom


@numba.njit(cache=True)
def _monte_core(
    # Arrays modified in-place
    MaxCollisionFreqNT,
    CollisionTimes,
    CollisionEnergies,
    ICOLNNNT,
    CollisionsPerGasPerTypeNT,
    ICOLNNT,
    CollisionsPerGasPerType,
    # Read-only arrays
    TotalCollisionFreqNullNT,
    TotalCollisionFreqNT,
    NullCollisionFreqNT,
    CollisionFreqNT,
    RGasNT, EnergyLevelsNT, ElectronNumChangeNT,
    NC0NT, InteractionTypeNT, PenningFractionNT,
    AngularModelNT, AngleCutNT, ScatteringParameterNT,
    # Scalar inputs
    FakeIonizations_in,
    EField, CONST1, CONST2, CONST3,
    AngleFromZ_in, AngleFromX,
    MaxNumberOfCollisions, Num_Samples,
    Max_Electron_Energy, ElectronEnergyStep,
    BFieldMode, AngularSpeedOfRotation,
    Decor_Lookbacks, Decor_Colls, Decor_Step,
    NumMomCrossSectionPointsNullNT,
    NumMomCrossSectionPointsNT,
    ISIZENT,
    InitialElectronEnergy,
    MeanCollisionTime_in,
    Enable_Penning,
    AnisotropicDetected,
    SmallNumber,
    NumberOfGases,
):
    X = 0.0; Y = 0.0; Z = 0.0; TimeSum = 0.0
    VelocityZ = 0.0
    MeanCollisionTime = MeanCollisionTime_in
    FakeIonizations = FakeIonizations_in
    AngleFromZ = AngleFromZ_in

    I = 0
    ST1 = 0.0; ST2 = 0.0; SumE2 = 0.0
    SumXX = 0.0; SumYY = 0.0; SumZZ = 0.0
    SumVX = 0.0; SumVY = 0.0
    Z_LastSample = 0.0; ST_LastSample = 0.0
    ST1_LastSample = 0.0; ST2_LastSample = 0.0
    SumZZ_LastSample = 0.0; SumXX_LastSample = 0.0
    SumYY_LastSample = 0.0; SumVX_LastSample = 0.0
    SumVY_LastSample = 0.0; SME2_LastSample = 0.0
    TDash = 0.0; SVX_LastSample = 0.0; SVY_LastSample = 0.0

    CollT = np.zeros(2000000)
    CollX = np.zeros(2000000)
    CollY = np.zeros(2000000)
    CollZ = np.zeros(2000000)

    DriftVelPerSample = np.zeros(Num_Samples)
    MeanEnergyPerSample = np.zeros(Num_Samples)
    DiffZZPerSample = np.zeros(Num_Samples)
    DiffYYPerSample = np.zeros(Num_Samples)
    DiffXXPerSample = np.zeros(Num_Samples)

    EBefore = InitialElectronEnergy
    NumDecorLengths = 0
    NumCollisions = 0
    IEXTRA = 0

    TEMP = np.zeros(4000)
    for J in range(4000):
        TEMP[J] = TotalCollisionFreqNullNT[J] + TotalCollisionFreqNT[J]
    ABSFAKEI = abs(FakeIonizations)
    FakeIonizations = 0

    BP = EField ** 2 * CONST1
    F1 = EField * CONST2
    F2 = EField * CONST3
    Sqrt2M = CONST3 * 0.01
    TwoPi = 2.0 * math.pi

    DirCosineZ1 = math.cos(AngleFromZ)
    DirCosineX1 = math.sin(AngleFromZ) * math.cos(AngleFromX)
    DirCosineY1 = math.sin(AngleFromZ) * math.sin(AngleFromX)

    # Initialise so Numba can infer types even if BFieldMode branch never runs
    DirCosineX2 = 0.0; DirCosineY2 = 0.0; DirCosineZ2 = 0.0

    CollisionsPerSample = int(MaxNumberOfCollisions / Num_Samples)
    INTEM = 8
    DeltaE = Max_Electron_Energy / float(INTEM)

    MeanElectronEnergy = 0.0
    DiffusionX = 0.0; DiffusionY = 0.0; DiffusionZ = 0.0

    for iSample in range(Num_Samples):
        for iCollision in range(CollisionsPerSample):
            while True:
                RandomNum = np.random.random()
                I = int(EBefore / DeltaE) + 1
                I = min(I, INTEM) - 1
                TLIM = MaxCollisionFreqNT[I]

                T = -1.0 * math.log(RandomNum) / TLIM + TDash
                MeanCollisionTime = 0.9 * MeanCollisionTime + 0.1 * T
                TDash = T
                AP = DirCosineZ1 * F2 * math.sqrt(EBefore)
                EAfter = EBefore + (AP + BP * T) * T
                iEnergyBin = int(EAfter / ElectronEnergyStep)
                iEnergyBin = min(iEnergyBin, 3999)

                if TEMP[iEnergyBin] > TLIM:
                    TDash += math.log(RandomNum) / TLIM
                    MaxCollisionFreqNT[I] *= 1.05
                    continue

                RandomNum = np.random.random()
                Test1 = TotalCollisionFreqNT[iEnergyBin] / TLIM

                if RandomNum > Test1:
                    Test2 = TEMP[iEnergyBin] / TLIM
                    if RandomNum < Test2:
                        if NumMomCrossSectionPointsNullNT == 0:
                            continue
                        RandomNum = np.random.random()
                        I = 0
                        while NullCollisionFreqNT[iEnergyBin, I] < RandomNum:
                            I += 1
                        ICOLNNNT[I] += 1.0
                        continue
                    else:
                        Test3 = (TEMP[iEnergyBin] + ABSFAKEI) / TLIM
                        if RandomNum < Test3:
                            FakeIonizations += 1
                            continue
                        continue
                else:
                    break

            # Real collision
            NumCollisions += 1
            MeanCollisionTime = 0.9 * MeanCollisionTime + 0.1 * TDash
            TDash = 0.0

            T2 = T * T
            A = AP * T
            B = BP * T2
            SumE2 = SumE2 + T * (EBefore + A / 2.0 + B / 3.0)

            VelocityBefore = Sqrt2M * math.sqrt(EBefore)
            VelXBefore = DirCosineX1 * VelocityBefore
            VelYBefore = DirCosineY1 * VelocityBefore

            if BFieldMode == 1:
                VelocityRatio = math.sqrt(EBefore / EAfter)
                DirCosineX2 = DirCosineX1 * VelocityRatio
                DirCosineY2 = DirCosineY1 * VelocityRatio
                DirCosineZ2 = DirCosineZ1 * VelocityRatio + T * F2 / (2.0 * math.sqrt(EAfter))
                A_pos = T * VelocityBefore
                X = X + DirCosineX1 * A_pos
                Y = Y + DirCosineY1 * A_pos
                Z = Z + DirCosineZ1 * A_pos + T2 * F1
            elif BFieldMode == 2:
                VelocityRatio = math.sqrt(EBefore / EAfter)
                WBT = AngularSpeedOfRotation * T
                CosWT = math.cos(WBT)
                SinWT = math.sin(WBT)
                VelAfter = Sqrt2M * math.sqrt(EAfter)
                A_pos = T * VelocityBefore
                DirCosineX2 = (VelXBefore * CosWT - VelYBefore * SinWT) / VelAfter
                DirCosineY2 = (VelYBefore * CosWT + VelXBefore * SinWT) / VelAfter
                DirCosineZ2 = DirCosineZ1 * VelocityRatio + T * F2 / (2.0 * math.sqrt(EAfter))
                X += (VelXBefore * SinWT - VelYBefore * (1.0 - CosWT)) / AngularSpeedOfRotation
                Y += (VelYBefore * SinWT + VelXBefore * (1.0 - CosWT)) / AngularSpeedOfRotation
                Z += DirCosineZ1 * A_pos + T2 * F1

            TimeSum = TimeSum + T
            iTimeBin = int(T)
            iTimeBin = min(iTimeBin, 299)
            CollisionTimes[iTimeBin] += 1.0
            CollisionEnergies[iEnergyBin] += 1.0
            VelocityZ = Z / TimeSum
            SumVX = SumVX + VelXBefore * VelXBefore * T2
            SumVY = SumVY + VelYBefore * VelYBefore * T2

            if NumDecorLengths != 0:
                CollsToLookBack = 0
                for iCorr in range(Decor_Lookbacks):
                    ST2 += T
                    DecorDistance = NumCollisions + CollsToLookBack
                    if DecorDistance > Decor_Colls:
                        DecorDistance = DecorDistance - Decor_Colls
                    TDiff = TimeSum - CollT[DecorDistance - 1]
                    SumXX = SumXX + ((X - CollX[DecorDistance - 1]) ** 2) * T / TDiff
                    SumYY = SumYY + ((Y - CollY[DecorDistance - 1]) ** 2) * T / TDiff
                    if iSample >= 2:
                        ST1 += T
                        SumZZ = SumZZ + (
                            (Z - CollZ[DecorDistance - 1] - VelocityZ * TDiff) ** 2
                        ) * T / TDiff
                    CollsToLookBack += Decor_Step

            CollX[NumCollisions - 1] = X
            CollY[NumCollisions - 1] = Y
            CollZ[NumCollisions - 1] = Z
            CollT[NumCollisions - 1] = TimeSum

            if NumCollisions >= Decor_Colls:
                NumDecorLengths += 1
                NumCollisions = 0

            RandomNum = np.random.random()
            I = _mbsort(I, RandomNum, iEnergyBin, CollisionFreqNT, NumMomCrossSectionPointsNT, ISIZENT)
            while CollisionFreqNT[iEnergyBin, I] < RandomNum:
                I += 1

            S1 = RGasNT[I]
            EI = EnergyLevelsNT[I]
            if ElectronNumChangeNT[I] > 0:
                RandomNum = np.random.random()
                EXTRA = RandomNum * (EAfter - EI)
                EI = EXTRA + EI
                IEXTRA += int(NC0NT[I])

            IPT = int(InteractionTypeNT[I])
            CollisionsPerGasPerTypeNT[IPT - 1] += 1.0
            ICOLNNT[I] += 1.0

            if EAfter < EI:
                EI = EAfter - 0.0001

            if Enable_Penning != 0:
                if PenningFractionNT[0, I] != 0.0:
                    RAN = np.random.random()
                    if RAN <= PenningFractionNT[0, I]:
                        IEXTRA += 1

            S2 = (S1 ** 2) / (S1 - 1.0)

            RandomNum = np.random.random()
            if AngularModelNT[I] == 1:
                RandomNum1 = np.random.random()
                CosTheta = 1.0 - RandomNum * AngleCutNT[iEnergyBin, I]
                if RandomNum1 > ScatteringParameterNT[iEnergyBin, I]:
                    CosTheta = -1.0 * CosTheta
            elif AngularModelNT[I] == 2:
                EPSI = ScatteringParameterNT[iEnergyBin, I]
                CosTheta = 1.0 - (2.0 * RandomNum * (1.0 - EPSI) / (1.0 + EPSI * (1.0 - 2.0 * RandomNum)))
            else:
                CosTheta = 1.0 - 2.0 * RandomNum

            Theta = math.acos(CosTheta)
            RandomNum = np.random.random()
            Phi = TwoPi * RandomNum
            SinPhi = math.sin(Phi)
            CosPhi = math.cos(Phi)

            ARG1 = max(1.0 - S1 * EI / EAfter, SmallNumber)
            D = 1.0 - CosTheta * math.sqrt(ARG1)
            U = (S1 - 1.0) * (S1 - 1.0) / ARG1

            EBefore = max(EAfter * (1.0 - EI / (S1 * EAfter) - 2.0 * D / S2), SmallNumber)
            Q = min(math.sqrt((EAfter / EBefore) * ARG1) / S1, 1.0)

            AngleFromZ = math.asin(Q * math.sin(Theta))
            CosZAngle = math.cos(AngleFromZ)
            if CosTheta < 0.0 and CosTheta * CosTheta > U:
                CosZAngle = -1.0 * CosZAngle
            SinZAngle = math.sin(AngleFromZ)
            DirCosineZ2 = min(DirCosineZ2, 1.0)

            ARGZ = math.sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
            if ARGZ == 0.0:
                DirCosineZ1 = CosZAngle
                DirCosineX1 = CosPhi * SinZAngle
                DirCosineY1 = SinPhi * SinZAngle
            else:
                DirCosineZ1 = DirCosineZ2 * CosZAngle + ARGZ * SinZAngle * SinPhi
                DirCosineY1 = (DirCosineY2 * CosZAngle +
                               (SinZAngle / ARGZ) * (DirCosineX2 * CosPhi - DirCosineY2 * DirCosineZ2 * SinPhi))
                DirCosineX1 = (DirCosineX2 * CosZAngle -
                               (SinZAngle / ARGZ) * (DirCosineY2 * CosPhi + DirCosineX2 * DirCosineZ2 * SinPhi))

        # End of sample — bookkeeping
        VelocityZ *= 1e9
        MeanElectronEnergy = SumE2 / TimeSum

        if AnisotropicDetected == 0:
            DiffusionX = 5e15 * SumVX / TimeSum
            DiffusionY = 5e15 * SumVY / TimeSum
            DiffXXPerSample[iSample] = 5.0e15 * (SumVX - SVX_LastSample) / (TimeSum - ST_LastSample)
            DiffYYPerSample[iSample] = 5.0e15 * (SumVY - SVY_LastSample) / (TimeSum - ST_LastSample)
        else:
            if ST2 != 0.0:
                DiffusionY = 5e15 * SumYY / ST2
                DiffusionX = 5e15 * SumXX / ST2
                DiffXXPerSample[iSample] = 5.0e15 * (SumXX - SumXX_LastSample) / (ST2 - ST2_LastSample)
                DiffYYPerSample[iSample] = 5.0e15 * (SumYY - SumYY_LastSample) / (ST2 - ST2_LastSample)
            else:
                DiffXXPerSample[iSample] = 0.0
                DiffYYPerSample[iSample] = 0.0

        if ST1 != 0.0:
            DiffusionZ = 5.0e15 * SumZZ / ST1
            DiffZZPerSample[iSample] = 5.0e15 * (SumZZ - SumZZ_LastSample) / (ST1 - ST1_LastSample)
        else:
            DiffZZPerSample[iSample] = 0.0

        DriftVelPerSample[iSample] = (Z - Z_LastSample) / (TimeSum - ST_LastSample) * 1.0e9
        MeanEnergyPerSample[iSample] = (SumE2 - SME2_LastSample) / (TimeSum - ST_LastSample)

        Z_LastSample = Z; ST_LastSample = TimeSum
        ST1_LastSample = ST1; ST2_LastSample = ST2
        SVX_LastSample = SumVX; SVY_LastSample = SumVY
        SumZZ_LastSample = SumZZ; SumYY_LastSample = SumYY
        SumXX_LastSample = SumXX; SME2_LastSample = SumE2

        if CollisionEnergies[3999] > (1000.0 * float(iSample + 1)):
            raise ValueError("WARNING ENERGY OUT OF RANGE, INCREASE ELECTRON ENERGY INTEGRATION RANGE")

    # Error calculation from per-sample statistics
    SumV = 0.0; SumE = 0.0; SumV2 = 0.0; SumE2s = 0.0
    SumDZZ = 0.0; SumDXX = 0.0; SumDYY = 0.0
    SumDXX2 = 0.0; SumDYY2 = 0.0; SumDZZ2 = 0.0

    for K in range(Num_Samples):
        SumV += DriftVelPerSample[K]
        SumE += MeanEnergyPerSample[K]
        SumV2 += DriftVelPerSample[K] ** 2
        SumE2s += MeanEnergyPerSample[K] ** 2
        SumDXX += DiffXXPerSample[K]
        SumDYY += DiffYYPerSample[K]
        SumDYY2 += DiffYYPerSample[K] ** 2
        SumDXX2 += DiffXXPerSample[K] ** 2
        if K >= 2:
            SumDZZ += DiffZZPerSample[K]
            SumDZZ2 += DiffZZPerSample[K] ** 2

    VelocityErrorZ = _safe_pct_err(
        (SumV2 - SumV ** 2 / Num_Samples) / (Num_Samples - 1), VelocityZ)
    MeanElectronEnergyError = _safe_pct_err(
        (SumE2s - SumE ** 2 / Num_Samples) / (Num_Samples - 1), MeanElectronEnergy)
    ErrorDiffusionX = _safe_pct_err(
        (SumDXX2 - SumDXX ** 2 / Num_Samples) / (Num_Samples - 1), DiffusionX)
    ErrorDiffusionY = _safe_pct_err(
        (SumDYY2 - SumDYY ** 2 / Num_Samples) / (Num_Samples - 1), DiffusionY)
    ErrorDiffusionZ = _safe_pct_err(
        (SumDZZ2 - SumDZZ ** 2 / Num_Samples - 2.0) / (Num_Samples - 2 - 1), DiffusionZ)

    VelocityErrorZ /= math.sqrt(float(Num_Samples))
    MeanElectronEnergyError /= math.sqrt(float(Num_Samples))
    ErrorDiffusionX /= math.sqrt(float(Num_Samples))
    ErrorDiffusionY /= math.sqrt(float(Num_Samples))
    ErrorDiffusionZ /= math.sqrt(float(Num_Samples - 2))

    LongitudinalDiffusion = DiffusionZ
    TransverseDiffusion = (DiffusionX + DiffusionY) / 2.0
    VelocityZ *= 1.0e5
    TransverseDiffusionError = (ErrorDiffusionX + ErrorDiffusionY) / 2.0
    LongitudinalDiffusionError = ErrorDiffusionZ

    Attachment = 0.0
    Ionization = 0.0
    for Idx in range(NumberOfGases):
        Attachment += CollisionsPerGasPerType[Idx, 2]
        Ionization += CollisionsPerGasPerType[Idx, 1]
    Ionization += float(IEXTRA)

    AttachmentRateError = 0.0
    if Attachment != 0.0:
        AttachmentRateError = 100.0 * math.sqrt(Attachment) / Attachment
    AttachmentRate = Attachment / (TimeSum * VelocityZ) * 1e12 if VelocityZ != 0.0 else 0.0

    IonisationRateError = 0.0
    if Ionization != 0.0:
        IonisationRateError = 100.0 * math.sqrt(Ionization) / Ionization
    IonisationRate = Ionization / (TimeSum * VelocityZ) * 1e12 if VelocityZ != 0.0 else 0.0

    return (
        X, Y, Z, TimeSum, VelocityZ, MeanCollisionTime, float(FakeIonizations),
        MeanElectronEnergy, MeanElectronEnergyError,
        DiffusionX, DiffusionY, DiffusionZ,
        VelocityErrorZ, ErrorDiffusionX, ErrorDiffusionY, ErrorDiffusionZ,
        TransverseDiffusion, LongitudinalDiffusion,
        TransverseDiffusionError, LongitudinalDiffusionError,
        AttachmentRate, AttachmentRateError,
        IonisationRate, IonisationRateError,
    )


def run(Object):
    """
    Drop-in replacement for MONTE.run() using Numba JIT compilation.
    First call incurs compilation overhead; subsequent calls are fast.
    """
    Object.VelocityX = 0.0
    Object.VelocityY = 0.0
    Object.VelocityErrorX = 0.0
    Object.VelocityErrorY = 0.0

    result = _monte_core(
        # Arrays modified in-place
        Object.MaxCollisionFreqNT,
        Object.CollisionTimes,
        Object.CollisionEnergies,
        Object.ICOLNNNT,
        Object.CollisionsPerGasPerTypeNT,
        Object.ICOLNNT,
        Object.CollisionsPerGasPerType,
        # Read-only arrays
        Object.TotalCollisionFrequencyNullNT,
        Object.TotalCollisionFrequencyNT,
        Object.NullCollisionFreqNT,
        Object.CollisionFrequencyNT,
        Object.RGasNT, Object.EnergyLevelsNT, Object.ElectronNumChangeNT,
        Object.NC0NT, Object.InteractionTypeNT, Object.PenningFractionNT,
        Object.AngularModelNT, Object.AngleCutNT, Object.ScatteringParameterNT,
        # Scalars
        float(Object.FakeIonizations),
        float(Object.EField), float(Object.CONST1), float(Object.CONST2), float(Object.CONST3),
        float(Object.AngleFromZ), float(Object.AngleFromX),
        int(Object.MaxNumberOfCollisions), int(Object.Num_Samples),
        float(Object.Max_Electron_Energy), float(Object.ElectronEnergyStep),
        int(Object.BFieldMode), float(Object.AngularSpeedOfRotation),
        int(Object.Decor_Lookbacks), int(Object.Decor_Colls), int(Object.Decor_Step),
        int(Object.NumMomCrossSectionPointsNullNT),
        int(Object.NumMomCrossSectionPointsNT),
        float(Object.ISIZENT),
        float(Object.InitialElectronEnergy),
        float(Object.MeanCollisionTime),
        int(Object.Enable_Penning),
        int(Object.AnisotropicDetected),
        float(Object.SmallNumber),
        int(Object.NumberOfGases),
    )

    (Object.X, Object.Y, Object.Z, Object.TimeSum,
     Object.VelocityZ, Object.MeanCollisionTime, Object.FakeIonizations,
     Object.MeanElectronEnergy, Object.MeanElectronEnergyError,
     Object.DiffusionX, Object.DiffusionY, Object.DiffusionZ,
     Object.VelocityErrorZ, Object.ErrorDiffusionX, Object.ErrorDiffusionY, Object.ErrorDiffusionZ,
     Object.TransverseDiffusion, Object.LongitudinalDiffusion,
     Object.TransverseDiffusionError, Object.LongitudinalDiffusionError,
     Object.AttachmentRate, Object.AttachmentRateError,
     Object.IonisationRate, Object.IonisationRateError) = result
