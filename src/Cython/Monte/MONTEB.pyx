from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow
from libc.string cimport memset
from PyBoltz cimport drand48
from MBSorts cimport MBSort
from libc.stdlib cimport malloc, free
import cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = drand48(dummy)
    return r

# The functionality of MONTEB mirrors MONTE.  The latter is more extensively documented in the code
#  and we refer developers there for more information.

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at temp =  TemperatureCentigrade.

    This function is used when the magnetic field is perpendicular to the electric field in the z direction.    
    
    The object parameter is the PyBoltz object to have the output results and to be used in the simulation.
    """
    cdef long long I, NumDecorLengths, NumCollisions, IEXTRA, IMBPT, K, J, iCollisionM, iSample, iCollision, GasIndex, IE, IT, CollsToLookBack, IPT, iCorr,DecorDistance
    cdef double ST1, RandomSeed,ST2, SumE2, SumXX, SumYY, SumZZ, SumVX, SumVY, Z_LastSample, ST_LastSample, ST1_LastSample, ST2_LastSample, SumZZ_LastSample, SumXX_LastSample, SumYY_LastSample, SVX_LastSample, SVY_LastSample, SME2_LastSample, TDash
    cdef double ABSFAKEI, DirCosineZ1, DirCosineX1, DirCosineY1, VelXBefore, VelYBefore, VelZBefore, BP, F1, F2, TwoPi, DirCosineX2, DirCosineY2, DirCosineZ2, VelXAfter, VelYAfter, VelZAfter, DZCOM, DYCOM, DXCOM, Theta,
    cdef double  EBefore, Sqrt2M, TwoM, AP, CONST6, RandomNum, GasVelX, GasVelY, GasVelZ, VEX, VEY, VEZ, COMEnergy, Test1, Test2, Test3, VelBeforeM1
    cdef double T2, A, B, CONST7, S1, EI, R9, EXTRA, RAN, RandomNum1, CosTheta, EPSI, Phi, SinPhi, CosPhi, ARG1, D, Q, CosZAngle, U, CosSquareTheta, SinZAngle, VXLAB, VYLAB, VZLAB
    cdef double SumV_Samples, SumE_Samples, SumV2_Samples, SumE2_Samples, SumDXX_Samples, SumDYY_Samples, SumDXX2_Samples, SumDYY2_Samples, SumDZZ_Samples, SumDZZ2_Samples, Attachment, Ionization, E,SumYZ,SumLS,SumTS
    cdef double SumYZ_LastSample,SLN_LastSample,STR_LastSample,MeanEnergy_LastSample,EFieldTimes100, MeanEnergy
    cdef double *CollT, *CollX, *CollY, *CollZ, *DriftVelPerSampleZ, *MeanEnergyPerSample, *DiffZZPerSample, *DiffYYPerSample, *DiffXXPerSample,*DiffYZPerSample,*DiffLonPerSample,*DriftVelPerSampleYZ, *DiffTranPerSample,TEMP[4000]
    CollT = <double *> malloc(2000000 * sizeof(double))
    memset(CollT, 0, 2000000 * sizeof(double))
    CollX = <double *> malloc(2000000 * sizeof(double))
    memset(CollX, 0, 2000000 * sizeof(double))

    CollY = <double *> malloc(2000000 * sizeof(double))
    memset(CollY, 0, 2000000 * sizeof(double))

    CollZ = <double *> malloc(2000000 * sizeof(double))
    memset(CollZ, 0, 2000000 * sizeof(double))

    DriftVelPerSampleZ = <double *> malloc(10 * sizeof(double))
    memset(DriftVelPerSampleZ, 0, 10 * sizeof(double))

    DriftVelPerSampleY = <double *> malloc(10 * sizeof(double))
    memset(DriftVelPerSampleY, 0, 10 * sizeof(double))

    MeanEnergyPerSample = <double *> malloc(10 * sizeof(double))
    memset(MeanEnergyPerSample, 0, 10 * sizeof(double))

    DiffZZPerSample = <double *> malloc(10 * sizeof(double))
    memset(DiffZZPerSample, 0, 10 * sizeof(double))

    DiffYYPerSample = <double *> malloc(10 * sizeof(double))
    memset(DiffYYPerSample, 0, 10 * sizeof(double))

    DiffXXPerSample = <double *> malloc(10 * sizeof(double))
    memset(DiffXXPerSample, 0, 10 * sizeof(double))

    DiffYZPerSample = <double *> malloc(10 * sizeof(double))
    memset(DiffYZPerSample, 0, 10 * sizeof(double))

    DiffLonPerSample = <double *> malloc(10 * sizeof(double))
    memset(DiffLonPerSample, 0, 10 * sizeof(double))


    DiffTranPerSample  = <double *> malloc(10 * sizeof(double))
    memset(DiffTranPerSample, 0, 10 * sizeof(double))

    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))
    for J in range(4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]

    Object.VelocityX = 0.0
    Object.VelocityErrorX = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.DiffusionXZ = 0.0
    Object.DiffusionXY = 0.0
    Object.ErrorDiffusionXZ = 0.0
    Object.ErrorDiffusionXY = 0.0
    Object.TimeSum = 0.0
    ST1 = 0.0
    ST2 = 0.0
    SumXX = 0.0
    SumYY = 0.0
    SumZZ = 0.0
    SumYZ = 0.0
    SumLS = 0.0
    SumTS = 0.0
    SumVX = 0.0
    Z_LastSample = 0.0
    Y_LastSample = 0.0
    ST_LastSample = 0.0
    ST1_LastSample = 0.0
    ST2_LastSample = 0.0
    SumZZ_LastSample = 0.0
    SumXX_LastSample = 0.0
    SumYY_LastSample = 0.0
    SumYZ_LastSample = 0.0
    SVX_LastSample = 0.0
    SLN_LastSample = 0.0
    STR_LastSample = 0.0
    MeanEnergy_LastSample = 0.0

    EFieldTimes100 = Object.EField * 100
    RandomSeed = Object.Random_Seed
    EBefore = Object.InitialElectronEnergy
    INTEM = 8
    NumDecorLengths = 0
    NumCollisions = 0
    IEXTRA = 0
    TDash = 0.0
    Sqrt2M = Object.CONST3 * 0.01

    ABSFAKEI = Object.FakeIonizations
    Object.FakeIonizations = 0

    TwoPi = 2 * acos(-1)
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    VelTotal = Sqrt2M * sqrt(EBefore)
    VelXBefore = DirCosineX1 * VelTotal
    VelYBefore = DirCosineY1 * VelTotal
    VelZBefore = DirCosineZ1 * VelTotal

    iCollisionM = <long long>(Object.MaxNumberOfCollisions / Object.Num_Samples)

    DELTAE = Object.Max_Electron_Energy / float(INTEM)
    if Object.Console_Output_Flag:
        print('{:^12s}{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity Z", "Velocity Y", "Energy",
                                                                       "DIFXX", "DIFYY", "DIFZZ", "DIFYZ","DIFLNG","DIFTRN"))
    for iSample in range(int(Object.Num_Samples)):
        for iCollision in range(int(iCollisionM)):
            while True:
                RandomNum = random_uniform(RandomSeed)
                I = int(EBefore / DELTAE) + 1
                I = min(I, INTEM) - 1
                TLIM = Object.MaxCollisionFreqNT[I]
                T = -1 * log(RandomNum) / TLIM + TDash
                Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T
                TDash = T
                WBT = Object.AngularSpeedOfRotation * T
                CosWT = cos(WBT)
                SinWT = sin(WBT)
                DZ = (VelZBefore * SinWT + (Object.EFieldOverBField - VelYBefore) * (1 - CosWT)) / Object.AngularSpeedOfRotation
                E = EBefore + DZ * EFieldTimes100
                IE = int(E / Object.ElectronEnergyStep)
                IE = min(IE, 3999)
                if TEMP[IE] > TLIM:
                    TDash += log(RandomNum) / TLIM
                    Object.MaxCollisionFreqNT[I] *= 1.05
                    continue

                RandomNum = random_uniform(RandomSeed)
                Test1 = Object.TotalCollisionFrequencyNT[IE] / TLIM

                if RandomNum > Test1:
                    Test2 = TEMP[IE] / TLIM
                    if RandomNum < Test2:
                        if Object.NumMomCrossSectionPointsNullNT == 0:
                            continue
                        RandomNum = random_uniform(RandomSeed)
                        I = 0
                        while Object.NullCollisionFreqNT[IE][I] < RandomNum:
                            I += 1

                        Object.ICOLNNNT[I] += 1
                        continue
                    else:
                        Test3 = (TEMP[IE] + ABSFAKEI) / TLIM
                        if RandomNum < Test3:
                            # FAKE IONISATION INCREMENT COUNTER
                            Object.FakeIonizations += 1
                            continue
                        continue
                else:
                    break

            T2 = T ** 2
            TDash = 0.0
            VelXAfter = VelXBefore
            VelYAfter = (VelYBefore - Object.EFieldOverBField) * CosWT + VelZBefore * SinWT + Object.EFieldOverBField
            VelZAfter = VelZBefore * CosWT - (VelYBefore - Object.EFieldOverBField) * SinWT
            VelTotal = sqrt(VelXAfter ** 2 + VelYAfter ** 2 + VelZAfter ** 2)
            DirCosineX2 = VelXAfter / VelTotal
            DirCosineY2 = VelYAfter / VelTotal
            DirCosineZ2 = VelZAfter / VelTotal
            NumCollisions += 1

            Object.X += VelXBefore * T
            Object.Y += Object.EFieldOverBField * T + ((VelYBefore - Object.EFieldOverBField) * SinWT + VelZBefore * (1 - CosWT)) / Object.AngularSpeedOfRotation
            Object.Z += DZ
            Object.TimeSum += T
            IT = int(T)
            IT = min(IT, 299)
            Object.CollisionTimes[IT] += 1
            Object.CollisionEnergies[IE] += 1
            Object.VelocityZ = Object.Z / Object.TimeSum
            Object.VelocityY = Object.Y / Object.TimeSum
            SumVX += (VelXBefore ** 2) * T2
            if NumDecorLengths != 0:
                CollsToLookBack = 0
                for J in range(int(Object.Decor_Lookbacks)):
                    ST2 = ST2 + T
                    DecorDistance = NumCollisions + CollsToLookBack
                    if DecorDistance > Object.Decor_Colls:
                        DecorDistance = DecorDistance - Object.Decor_Colls
                    TDiff = Object.TimeSum - CollT[DecorDistance-1]
                    SumXX += ((Object.X - CollX[DecorDistance-1]) ** 2) * T / TDiff
                    CollsToLookBack += Object.Decor_Step
                    if iSample >= 2:
                        ST1 += T
                        SumZZ += ((Object.Z - CollZ[DecorDistance-1] - Object.VelocityZ * TDiff) ** 2) * T / TDiff
                        SumYY += ((Object.Y - CollY[DecorDistance-1] - Object.VelocityY * TDiff) ** 2) * T / TDiff
                        SumYZ += (Object.Z - CollZ[DecorDistance-1] - Object.VelocityZ * TDiff) * (
                                Object.Y - CollY[DecorDistance-1] - Object.VelocityY * TDiff) * T / TDiff
                        A2 = (Object.VelocityZ * TDiff) ** 2 + (Object.VelocityY * TDiff) ** 2
                        B2 = (Object.Z - Object.VelocityZ * TDiff - CollZ[DecorDistance-1]) ** 2 + (
                                Object.Y - Object.VelocityY * TDiff - CollY[DecorDistance-1]) ** 2
                        C2 = (Object.Z - CollZ[DecorDistance-1]) ** 2 + (Object.Y - CollY[DecorDistance-1]) ** 2
                        DL2 = (A2 + B2 - C2) ** 2 / (4 * A2)
                        DT2 = B2 - DL2
                        SumLS += DL2 * T / TDiff
                        SumTS += DT2 * T / TDiff
            CollX[NumCollisions-1] = Object.X
            CollY[NumCollisions-1] = Object.Y
            CollZ[NumCollisions-1] = Object.Z
            CollT[NumCollisions-1] = Object.TimeSum
            if NumCollisions >= Object.Decor_Colls:
                NumDecorLengths += 1
                NumCollisions = 0

            RandomNum = random_uniform(RandomSeed)

            I = MBSort(I, RandomNum,  IE, Object)
            while Object.CollisionFrequencyNT[IE][I] < RandomNum:
                I = I + 1

            S1 = Object.RGasNT[I]
            EI = Object.EnergyLevelsNT[I]
            if Object.ElectronNumChangeNT[I] > 0:
                R9 = random_uniform(RandomSeed)
                EXTRA = R9 * (E - EI)
                EI = EXTRA + EI
                IEXTRA += <long long>(Object.NC0NT[I])
            IPT = <long long>(Object.InteractionTypeNT[I])
            Object.CollisionsPerGasPerTypeNT[int(IPT)] += 1
            Object.ICOLNNT[I] += 1
            if E < EI:
                EI = E - 0.0001

            if Object.Enable_Penning != 0:
                if Object.PenningFractionNT[0][I] != 0:
                    RAN = random_uniform(RandomSeed)
                    if RAN <= Object.PenningFractionNT[0][I]:
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

            RandomNum = random_uniform(RandomSeed)
            if Object.AngularModelNT[I] == 1:
                RandomNum1 = random_uniform(RandomSeed)
                CosTheta = 1.0 - RandomNum * Object.AngleCutNT[IE][I]
                if RandomNum1 > Object.ScatteringParameterNT[IE][I]:
                    CosTheta = -1 * CosTheta
            elif Object.AngularModelNT[I] == 2:
                EPSI = Object.ScatteringParameterNT[IE][I]
                CosTheta = 1 - (2 * RandomNum * (1 - EPSI) / (1 + EPSI * (1 - 2 * RandomNum)))
            else:
                CosTheta = 1 - 2 * RandomNum
            Theta = acos(CosTheta)
            RandomNum = random_uniform(RandomSeed)
            Phi = TwoPi * RandomNum
            SinPhi = sin(Phi)
            CosPhi = cos(Phi)
            ARG1 = 1 - S1 * EI / E
            ARG1 = max(ARG1, Object.SmallNumber)
            D = 1 - CosTheta * sqrt(ARG1)
            EBefore = E * (1 - EI / (S1 * E) - 2 * D / S2)
            EBefore = max(EBefore, Object.SmallNumber)
            Q = sqrt((E / EBefore) * ARG1) / S1
            Q = min(Q, 1)
            Object.AngleFromZ = asin(Q * sin(Theta))
            CosZAngle = cos(Object.AngleFromZ)
            U = (S1 - 1) * (S1 - 1) / ARG1
            CosSquareTheta = CosTheta * CosTheta
            if CosTheta < 0 and CosSquareTheta > U:
                CosZAngle = -1 * CosZAngle
            SinZAngle = sin(Object.AngleFromZ)
            DirCosineZ2 = min(DirCosineZ2, 1)
            VelTotal = Sqrt2M * sqrt(EBefore)
            ARGZ = sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
            if ARGZ == 0:
                DirCosineZ1 = CosZAngle
                DirCosineX1 = CosPhi * SinZAngle
                DirCosineY1 = SinPhi * SinZAngle
            else:
                DirCosineZ1 = DirCosineZ2 * CosZAngle + ARGZ * SinZAngle * SinPhi
                DirCosineY1 = DirCosineY2 * CosZAngle + (SinZAngle / ARGZ) * (DirCosineX2 * CosPhi - DirCosineY2 * DirCosineZ2 * SinPhi)
                DirCosineX1 = DirCosineX2 * CosZAngle - (SinZAngle / ARGZ) * (DirCosineY2 * CosPhi + DirCosineX2 * DirCosineZ2 * SinPhi)
            VelXBefore = DirCosineX1 * VelTotal
            VelYBefore = DirCosineY1 * VelTotal
            VelZBefore = DirCosineZ1 * VelTotal
        Object.VelocityZ *= 1e9
        Object.VelocityY *= 1e9
        if ST2 != 0.0:
            Object.DiffusionX = 5e15 * SumXX / ST2
        if ST1 != 0.0:
            Object.DiffusionZ = 5e15 * SumZZ / ST1
            Object.DiffusionY = 5e15 * SumYY / ST1
            Object.DiffusionYZ = -5e15 * SumYZ / ST1
            Object.LongitudinalDiffusion = 5e15 * SumLS / ST1
            Object.TransverseDiffusion = 5e15 * SumTS / ST1
        if Object.AnisotropicDetected == 0:
            Object.DiffusionX = 5e15 * SumVX / Object.TimeSum
        MeanEnergy = 0.0
        for IK in range(4000):
            MeanEnergy += Object.E[IK] * Object.CollisionEnergies[IK] / Object.TotalCollisionFrequencyNT[IK]
        Object.MeanElectronEnergy = MeanEnergy / Object.TimeSum
        DriftVelPerSampleZ[iSample] = (Object.Z - Z_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        DriftVelPerSampleY[iSample] = (Object.Y - Y_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        MeanEnergyPerSample[iSample] = (MeanEnergy - MeanEnergy_LastSample) / (Object.TimeSum - ST_LastSample)
        MeanEnergy_LastSample = MeanEnergy
        DiffZZPerSample[iSample] = 0.0
        DiffYYPerSample[iSample] = 0.0
        DiffYZPerSample[iSample] = 0.0
        DiffLonPerSample[iSample] = 0.0
        DiffTranPerSample[iSample] = 0.0
        if iSample > 1:
            DiffZZPerSample[iSample] = 5e15 * (SumZZ - SumZZ_LastSample) / (ST1 - ST1_LastSample)
            DiffYYPerSample[iSample] = 5e15 * (SumYY - SumYY_LastSample) / (ST1 - ST1_LastSample)
            DiffYZPerSample[iSample] = 5e15 * (SumYZ - SumYZ_LastSample) / (ST1 - ST1_LastSample)
            DiffLonPerSample[iSample] = 5e15 * (SumLS - SLN_LastSample) / (ST1 - ST1_LastSample)
            DiffTranPerSample[iSample] = 5e15 * (SumTS - STR_LastSample) / (ST1 - ST1_LastSample)
        DiffXXPerSample[iSample] = 5e15 * (SumXX - SumXX_LastSample) / (ST2 - ST2_LastSample)
        if Object.AnisotropicDetected == 0:
            DiffXXPerSample[iSample] = 5e15 * (SumVX - SVX_LastSample) / (Object.TimeSum - ST_LastSample)
        Z_LastSample = Object.Z
        Y_LastSample = Object.Y
        ST_LastSample = Object.TimeSum
        ST1_LastSample = ST1
        ST2_LastSample = ST2
        SVX_LastSample = SumVX
        SumZZ_LastSample = SumZZ
        SumXX_LastSample = SumXX
        SumYY_LastSample = SumYY
        SumYZ_LastSample = SumYZ
        SLN_LastSample = SumLS
        STR_LastSample = SumTS
        if Object.Console_Output_Flag:
            print('{:^12.1f}{:^12.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(Object.VelocityZ,Object.VelocityY,
                                                                                    Object.MeanElectronEnergy, Object.DiffusionX, Object.DiffusionY,
                                                                                    Object.DiffusionZ,Object.DiffusionYZ,Object.LongitudinalDiffusion,Object.TransverseDiffusion))
    SumV_Samples = 0.0
    TDriftVelPerSampleY = 0.0
    SumE_Samples = 0.0
    SumV2_Samples = 0.0
    T2DriftVelPerSampleY = 0.0
    SumE2_Samples = 0.0
    SumDZZ_Samples = 0.0
    SumDYY_Samples = 0.0
    SumDXX_Samples = 0.0
    TYCollZ = 0.0
    TLNST = 0.0
    TTRST = 0.0
    SumDZZ2_Samples = 0.0
    SumDYY2_Samples = 0.0
    SumDXX2_Samples = 0.0
    T2YCollZ = 0.0
    T2LNST = 0.0
    T2TRST = 0.0

    for K in range(10):
        SumV_Samples = SumV_Samples + DriftVelPerSampleZ[K]
        TDriftVelPerSampleY = TDriftVelPerSampleY + DriftVelPerSampleY[K]
        SumE_Samples = SumE_Samples + MeanEnergyPerSample[K]
        SumV2_Samples = SumV2_Samples + DriftVelPerSampleZ[K] * DriftVelPerSampleZ[K]
        T2DriftVelPerSampleY = T2DriftVelPerSampleY + DriftVelPerSampleY[K] * DriftVelPerSampleY[K]
        SumE2_Samples = SumE2_Samples + MeanEnergyPerSample[K] * MeanEnergyPerSample[K]
        SumDXX_Samples += DiffXXPerSample[K]
        SumDXX2_Samples += DiffXXPerSample[K] ** 2
        if K >= 2:
            SumDZZ_Samples = SumDZZ_Samples + DiffZZPerSample[K]
            SumDYY_Samples = SumDYY_Samples + DiffYYPerSample[K]
            TYCollZ = TYCollZ + DiffYZPerSample[K]
            TLNST = TLNST + DiffLonPerSample[K]
            TTRST = TTRST + DiffTranPerSample[K]
            SumDZZ2_Samples += DiffZZPerSample[K] ** 2
            SumDYY2_Samples += DiffYYPerSample[K] ** 2
            T2YCollZ += DiffYZPerSample[K] ** 2
            T2LNST += DiffLonPerSample[K] ** 2
            T2TRST += DiffTranPerSample[K] ** 2
    Object.VelocityErrorZ = 100 * sqrt((SumV2_Samples - SumV_Samples * SumV_Samples / Object.Num_Samples) / (Object.Num_Samples - 1)) / Object.VelocityZ
    Object.VelocityErrorY = 100 * sqrt((T2DriftVelPerSampleY - TDriftVelPerSampleY * TDriftVelPerSampleY / Object.Num_Samples) / (Object.Num_Samples - 1)) / abs(Object.VelocityY)
    Object.MeanElectronEnergyError = 100 * sqrt((SumE2_Samples - SumE_Samples * SumE_Samples / Object.Num_Samples) / (Object.Num_Samples - 1)) / Object.MeanElectronEnergy
    Object.ErrorDiffusionX = 100 * sqrt((SumDXX2_Samples - SumDXX_Samples * SumDXX_Samples / Object.Num_Samples) / (Object.Num_Samples - 1)) / Object.DiffusionX
    Object.ErrorDiffusionY = 100 * sqrt((SumDYY2_Samples - SumDYY_Samples * SumDYY_Samples / Object.Num_Samples) / (Object.Num_Samples - 1)) / Object.DiffusionY
    Object.ErrorDiffusionZ = 100 * sqrt((SumDZZ2_Samples - SumDZZ_Samples * SumDZZ_Samples / (Object.Num_Samples - 2)) / (Object.Num_Samples - 3)) / Object.DiffusionZ
    Object.ErrorDiffusionYZ = 100 * sqrt((T2YCollZ - TYCollZ * TYCollZ / (Object.Num_Samples - 2)) / (Object.Num_Samples - 3)) / abs(Object.DiffusionYZ)
    Object.LongitudinalDiffusionError = 100 * sqrt((T2LNST - TLNST * TLNST / (Object.Num_Samples - 2)) / (Object.Num_Samples - 3)) / Object.LongitudinalDiffusion
    Object.TransverseDiffusionError = 100 * sqrt((T2TRST - TTRST * TTRST / (Object.Num_Samples - 2)) / (Object.Num_Samples - 3)) / Object.TransverseDiffusion
    Object.VelocityErrorZ = Object.VelocityErrorZ / sqrt(Object.Num_Samples)
    Object.VelocityErrorY = Object.VelocityErrorY / sqrt(Object.Num_Samples)
    Object.MeanElectronEnergyError = Object.MeanElectronEnergyError / sqrt(Object.Num_Samples)
    Object.ErrorDiffusionX = Object.ErrorDiffusionX / sqrt(Object.Num_Samples)
    Object.ErrorDiffusionY = Object.ErrorDiffusionY / sqrt((Object.Num_Samples - 2))
    Object.ErrorDiffusionZ = Object.ErrorDiffusionZ / sqrt((Object.Num_Samples - 2))
    Object.ErrorDiffusionYZ = Object.ErrorDiffusionYZ / sqrt((Object.Num_Samples - 2))
    Object.LongitudinalDiffusionError = Object.LongitudinalDiffusionError / sqrt((Object.Num_Samples - 2))
    Object.TransverseDiffusionError = Object.TransverseDiffusionError / sqrt((Object.Num_Samples - 2))

    # CONVERT CM/SEC

    Object.VelocityZ *= 1e5
    Object.VelocityY *= 1e5

    Attachment = 0.0
    Ionization = 0.0
    for I in range(Object.NumberOfGases):
        Attachment += Object.CollisionsPerGasPerTypeNT[5 * (I + 1) - 3]
        Ionization += Object.CollisionsPerGasPerTypeNT[5 * (I + 1) - 4]
    Ionization += IEXTRA
    Object.AttachmentRateError = 0.0
    if Attachment != 0:
        Object.AttachmentRateError = 100 * sqrt(Attachment) / Attachment
    Object.AttachmentRate = Attachment / (Object.TimeSum * Object.VelocityZ) * 1e12
    Object.IonisationRateError = 0.0
    if Ionization != 0:
        Object.IonisationRateError = 100 * sqrt(Ionization) / Ionization
    Object.IonisationRate = Ionization / (Object.TimeSum * Object.VelocityZ) * 1e12

    return

