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

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void GenerateMaxBoltz(double RandomSeed, double API,double *RandomMaxBoltzArray):
    cdef double RAN1, RAN2, TWOPI
    cdef int J
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RandomSeed)
        RAN2 = random_uniform(RandomSeed)
        TWOPI = 2.0 * API
        RandomMaxBoltzArray[J] = sqrt(-1*log(RAN1)) * cos(RAN2 * TWOPI)
        RandomMaxBoltzArray[J + 1] = sqrt(-1*log(RAN1)) * sin(RAN2 * TWOPI)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at temp =  TemperatureCentigrade.

    This function is used when the magnetic field is parallel to the electric field in the z direction.    
    
    The object parameter is the PyBoltz object to have the output results and to be used in the simulation.
    """
    cdef long long I, NumDecorLengths,  NCOL, IEXTRA, IMBPT, K, J, iCollisionM, iSample, iCollision, GasIndex, IE, IT, CollsToLookBack, IPT, iCorr,NC_LastSampleM
    cdef double ST1, RandomSeed,ST2, SumE2, SumXX, SumYY, SumZZ, SumVX, SumVY, Z_LastSample, ST_LastSample, ST1_LastSample, ST2_LastSample, SZZ_LastSample, SXX_LastSample, SYY_LastSample, SVX_LastSample, SVY_LastSample, SME2_LastSample, TDash
    cdef double ABSFAKEI, DirCosineZ1, DirCosineX1, DirCosineY1, CX1, CY1, CZ1, BP, F1, F2, TwoPi, DirCosineX2, DirCosineY2, DirCosineZ2, CX2, CY2, CZ2, DZCOM, DYCOM, DXCOM, THETA0,
    cdef double  EBefore, Sqrt2M, TwoM, AP, CONST5, CONST6, RandomNum, VGX, VGY, VGZ, VEX, VEY, VEZ, COMEnergy, Test1, Test2, Test3, CONST11
    cdef double T2, A, B, CONST7, S1, EI, R9, EXTRA, RAN, RandomNum1, F3, EPSI, PHI0, F8, F9, ARG1, D, Q, F6, U, CSQD, F5, VXLAB, VYLAB, VZLAB
    cdef double SumV_Samples, SumE_Samples, SumV2_Samples, SumE2_Samples, SumDXX_Samples, SumDYY_Samples, SumDXX2_Samples, SumDYY2_Samples, SumDZZ_Samples, SumDZZ2_Samples, Attachment, Ionization, E,TEMP[4000]
    cdef double NumSamples

    cdef double *STO, *XST, *YST, *ZST, *WZST, *AVEST, *DFZZST, *DFYYST, *DFXXST
    STO = <double *> malloc(2000000 * sizeof(double))
    memset(STO, 0, 2000000 * sizeof(double))
    XST = <double *> malloc(2000000 * sizeof(double))
    memset(XST, 0, 2000000 * sizeof(double))

    YST = <double *> malloc(2000000 * sizeof(double))
    memset(YST, 0, 2000000 * sizeof(double))

    ZST = <double *> malloc(2000000 * sizeof(double))
    memset(ZST, 0, 2000000 * sizeof(double))

    WZST = <double *> malloc(10 * sizeof(double))
    memset(WZST, 0, 10 * sizeof(double))

    AVEST = <double *> malloc(10 * sizeof(double))
    memset(AVEST, 0, 10 * sizeof(double))

    DFZZST = <double *> malloc(10 * sizeof(double))
    memset(DFZZST, 0, 10 * sizeof(double))

    DFYYST = <double *> malloc(10 * sizeof(double))
    memset(DFYYST, 0, 10 * sizeof(double))

    DFXXST = <double *> malloc(10 * sizeof(double))
    memset(DFXXST, 0, 10 * sizeof(double))


    Object.VelocityX = 0.0
    Object.VelocityY = 0.0
    Object.VelocityErrorX = 0.0
    Object.VelocityErrorY = 0.0
    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))
    for J in range(4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNullT[J]

    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.ST = 0.0
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
    SZZ_LastSample = 0.0
    SXX_LastSample = 0.0
    SYY_LastSample = 0.0
    SVX_LastSample = 0.0
    SVY_LastSample = 0.0
    SME2_LastSample = 0.0

    RandomSeed = Object.RandomSeed
    EBefore = Object.InitialElectronEnergy
    Sqrt2M = Object.CONST3 * 0.01
    CONST5 = Object.CONST3 / 2.0

    INTEM = 8
    NumSamples = 10
    NumDecorLengths = 0
    NCOL = 0
    IEXTRA = 0

    ABSFAKEI = Object.FAKEI
    Object.FakeIonizations = 0

    # INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    # INITIAL VELOCITY
    VTOT = Sqrt2M * sqrt(EBefore)
    CX1 = DirCosineX1 * VTOT
    CY1 = DirCosineY1 * VTOT
    CZ1 = DirCosineZ1 * VTOT

    BP = Object.EField ** 2 * Object.CONST1
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3
    TwoPi = 2 * acos(-1)

    iCollisionM = <long long>(Object.MaxNumberOfCollisions / NumSamples)
    DELTAE = Object.FinalElectronEnergy / float(INTEM)
    if Object.ConsoleOutputFlag:
        print('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity", "Position", "Time", "Energy",
                                                                       "DIFXX", "DIFYY", "DIFZZ"))
    for iSample in range(int(NumSamples)):
        for iCollision in range(int(iCollisionM)):
            while True:
                RandomNum = random_uniform(RandomSeed)
                I = int(EBefore / DELTAE) + 1
                I = min(I, INTEM) - 1
                TLIM = Object.MaxCollisionFreqNT[I]
                T = -1 * log(RandomNum) / TLIM + TDash
                Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T
                TDash = T
                AP = DirCosineZ1 * F2 * sqrt(EBefore)
                E = EBefore + (AP + BP * T) * T
                IE = int(E / Object.ElectronEnergyStep)
                IE = min(IE, 3999)
                if TEMP[IE] > TLIM:
                    TDash += log(RandomNum) / TLIM
                    Object.MaxCollisionFreqNT[I] *= 1.05
                    continue

                # Test FOR REAL OR NULL COLLISION
                RandomNum = random_uniform(RandomSeed)
                Test1 = Object.TotalCollisionFrequencyNullT[IE] / TLIM

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
            WBT = Object.AngularSpeedOfRotation * T
            COSWT = cos(WBT)
            SINWT = sin(WBT)
            CONST6 = sqrt(EBefore / E)
            CX2 = CX1 * COSWT - CY1 * SINWT
            CY2 = CY1 * COSWT + CX1 * SINWT
            VTOT = Sqrt2M * sqrt(E)
            DirCosineX2 = CX2 / VTOT
            DirCosineY2 = CY2 / VTOT
            DirCosineZ2 = DirCosineZ1 * CONST6 + Object.EField * T * CONST5 / sqrt(E)
            A = AP * T
            B = BP * T2
            SumE2 = SumE2 + T * (EBefore + A / 2.0 + B / 3.0)
            CONST7 = Sqrt2M * sqrt(EBefore)
            A = T * CONST7
            NCOL += 1
            DX = (CX1 * SINWT - CY1 * (1 - COSWT)) / Object.AngularSpeedOfRotation
            Object.X += DX
            DY = (CY1 * SINWT + CX1 * (1 - COSWT)) / Object.AngularSpeedOfRotation
            Object.Y += DY
            Object.Z += DirCosineZ1 * A + T2 * F1
            Object.ST += T
            IT = int(T)
            IT = min(IT, 299)
            Object.CollisionTimes[IT] += 1
            Object.CollisionEnergies[IE] += 1
            Object.VelocityZ = Object.Z / Object.ST

            SumVX = SumVX + DX ** 2
            SumVY = SumVY + DY ** 2

            if NumDecorLengths != 0:
                CollsToLookBack = 0
                for iCorr in range(int(Object.Decor_NCORST)):
                    ST2 = ST2 + T
                    NC_LastSampleM = NCOL + CollsToLookBack
                    if NC_LastSampleM > Object.Decor_NCOLM:
                        NC_LastSampleM = NC_LastSampleM - Object.Decor_NCOLM
                    TDiff = Object.ST - STO[NC_LastSampleM-1]
                    SumXX += ((Object.X - XST[NC_LastSampleM-1]) ** 2) * T / TDiff
                    SumYY += ((Object.Y - YST[NC_LastSampleM-1]) ** 2) * T / TDiff
                    CollsToLookBack += Object.Decor_NCORLN
                    if iSample >= 2:
                        ST1 += T
                        SumZZ += ((Object.Z - ZST[NC_LastSampleM-1] - Object.VelocityZ * TDiff) ** 2) * T / TDiff
            XST[NCOL-1] = Object.X
            YST[NCOL-1] = Object.Y
            ZST[NCOL-1] = Object.Z
            STO[NCOL-1] = Object.ST
            if NCOL >= Object.Decor_NCOLM:
                NumDecorLengths += 1
                NCOL = 0

            RandomNum = random_uniform(RandomSeed)

            I = MBSort(I,  RandomNum, IE, Object)
            while Object.NullCollisionFreqT[IE][I] < RandomNum:
                I = I + 1

            S1 = Object.RGASNT[I]
            EI = Object.EnergyLevelsNT[I]

            if Object.IPNNT[I] > 0:
                R9 = random_uniform(RandomSeed)
                EXTRA = R9 * (E - EI)
                EI = EXTRA + EI
                IEXTRA += <long long>(Object.NC0NT[I])
            IPT = <long long>(Object.IARRYNT[I])
            Object.CollisionsPerGasPerTypeNT[int(IPT)] += 1
            Object.ICOLNNT[I] += 1
            if E < EI:
                EI = E - 0.0001

            if Object.EnablePenning != 0:
                if Object.PenningFractionNT[0][I] != 0:
                    RAN = random_uniform(RandomSeed)
                    if RAN <= Object.PenningFractionNT[0][I]:
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

            RandomNum = random_uniform(RandomSeed)
            if Object.INDEXNT[I] == 1:
                RandomNum1 = random_uniform(RandomSeed)
                F3 = 1.0 - RandomNum * Object.AngleCutNT[IE][I]
                if RandomNum1 > Object.ScatteringParameterNT[IE][I]:
                    F3 = -1 * F3
            elif Object.INDEXNT[I] == 2:
                EPSI = Object.ScatteringParameterNT[IE][I]
                F3 = 1 - (2 * RandomNum * (1 - EPSI) / (1 + EPSI * (1 - 2 * RandomNum)))
            else:
                F3 = 1 - 2 * RandomNum
            THETA0 = acos(F3)
            RandomNum = random_uniform(RandomSeed)
            PHI0 = TwoPi * RandomNum
            F8 = sin(PHI0)
            F9 = cos(PHI0)
            ARG1 = 1 - S1 * EI / E
            ARG1 = max(ARG1, Object.SmallNumber)
            D = 1 - F3 * sqrt(ARG1)
            EBefore = E * (1 - EI / (S1 * E) - 2 * D / S2)
            EBefore = max(EBefore, Object.SmallNumber)
            Q = sqrt((E / EBefore) * ARG1) / S1
            Q = min(Q, 1)
            Object.AngleFromZ = asin(Q * sin(THETA0))
            F6 = cos(Object.AngleFromZ)
            U = (S1 - 1) * (S1 - 1) / ARG1
            CSQD = F3 * F3
            if F3 < 0 and CSQD > U:
                F6 = -1 * F6
            F5 = sin(Object.AngleFromZ)
            DirCosineZ2 = min(DirCosineZ2, 1)
            VTOT = Sqrt2M * sqrt(EBefore)
            ARGZ = sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
            if ARGZ == 0:
                DirCosineZ1 = F6
                DirCosineX1 = F9 * F5
                DirCosineY1 = F8 * F5
            else:
                DirCosineZ1 = DirCosineZ2 * F6 + ARGZ * F5 * F8
                DirCosineY1 = DirCosineY2 * F6 + (F5 / ARGZ) * (DirCosineX2 * F9 - DirCosineY2 * DirCosineZ2 * F8)
                DirCosineX1 = DirCosineX2 * F6 - (F5 / ARGZ) * (DirCosineY2 * F9 + DirCosineX2 * DirCosineZ2 * F8)
            CX1 = DirCosineX1 * VTOT
            CY1 = DirCosineY1 * VTOT
            CZ1 = DirCosineZ1 * VTOT

        Object.VelocityZ *= 1e9
        Object.MeanElectronEnergy = SumE2 / Object.ST
        if Object.AnisotropicDetected == 0:
            Object.DiffusionX = 5e15 * SumVX / Object.ST
            Object.DiffusionY = 5e15 * SumVY / Object.ST
            DFXXST[iSample] = 5e15 * (SumVX - SVX_LastSample) / (Object.ST - ST_LastSample)
            DFYYST[iSample] = 5e15 * (SumVY - SVY_LastSample) / (Object.ST - ST_LastSample)
        else:
            if ST2 != 0.0:
                Object.DiffusionY = 5e15 * SumYY / ST2
                Object.DiffusionX = 5e15 * SumXX / ST2
                DFXXST[iSample] = 5e15 * (SumXX - SXX_LastSample) / (ST2 - ST2_LastSample)
                DFYYST[iSample] = 5e15 * (SumYY - SYY_LastSample) / (ST2 - ST2_LastSample)
            else:
                DFXXST[iSample] = 0.0
                DFYYST[iSample] = 0.0
        if ST1 != 0.0:
            Object.DiffusionZ = 5e15 * SumZZ / ST1
            DFZZST[iSample] = 5e15 * (SumZZ - SZZ_LastSample) / (ST1 - ST1_LastSample)
        else:
            DFZZST[iSample] = 0.0
        WZST[iSample] = (Object.Z - Z_LastSample) / (Object.ST - ST_LastSample) * 1e9
        AVEST[iSample] = (SumE2 - SME2_LastSample) / (Object.ST - ST_LastSample)
        Z_LastSample = Object.Z
        ST_LastSample = Object.ST
        ST1_LastSample = ST1
        ST2_LastSample = ST2
        SVX_LastSample = SumVX
        SVY_LastSample = SumVY
        SZZ_LastSample = SumZZ
        SYY_LastSample = SumYY
        SXX_LastSample = SumXX
        SME2_LastSample = SumE2
        if Object.ConsoleOutputFlag:
            print('{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(Object.VelocityZ, Object.Z, Object.ST,
                                                                                    Object.MeanElectronEnergy, Object.DiffusionX, Object.DiffusionY,
                                                                                    Object.DiffusionZ))
    SumV_Samples = 0.0
    SumE_Samples = 0.0
    SumV2_Samples = 0.0
    SumE2_Samples = 0.0
    SumDYY_Samples = 0.0
    SumDXX_Samples = 0.0

    SumDZZ_Samples = 0.0
    SumDZZ2_Samples = 0.0
    SumDYY2_Samples = 0.0
    SumDXX2_Samples = 0.0
    for K in range(10):
        SumV_Samples = SumV_Samples + WZST[K]
        SumE_Samples = SumE_Samples + AVEST[K]
        SumV2_Samples = SumV2_Samples + WZST[K] * WZST[K]
        SumE2_Samples = SumE2_Samples + AVEST[K] * AVEST[K]
        SumDXX_Samples += DFXXST[K]
        SumDYY_Samples += DFYYST[K]
        SumDXX2_Samples += DFXXST[K] ** 2
        SumDYY2_Samples += DFYYST[K] ** 2
        if K >= 2:
            SumDZZ_Samples = SumDZZ_Samples + DFZZST[K]
            SumDZZ2_Samples += DFZZST[K] ** 2
    Object.VelocityErrorZ = 100 * sqrt((SumV2_Samples - SumV_Samples * SumV_Samples / 10.0) / 9.0) / Object.VelocityZ
    Object.MeanElectronEnergyError = 100 * sqrt((SumE2_Samples - SumE_Samples * SumE_Samples / 10.0) / 9.0) / Object.MeanElectronEnergy
    Object.ErrorDiffusionX = 100 * sqrt((SumDXX2_Samples - SumDXX_Samples * SumDXX_Samples / 10.0) / 9.0) / Object.DiffusionX
    Object.ErrorDiffusionY = 100 * sqrt((SumDYY2_Samples - SumDYY_Samples * SumDYY_Samples / 10.0) / 9.0) / Object.DiffusionY
    Object.ErrorDiffusionZ = 100 * sqrt((SumDZZ2_Samples - SumDZZ_Samples * SumDZZ_Samples / 8.0) / 7.0) / Object.DiffusionZ
    Object.VelocityErrorZ = Object.VelocityErrorZ / sqrt(10)
    Object.MeanElectronEnergyError = Object.MeanElectronEnergyError / sqrt(10)
    Object.ErrorDiffusionX = Object.ErrorDiffusionX / sqrt(10)
    Object.ErrorDiffusionY = Object.ErrorDiffusionY / sqrt(10)
    Object.ErrorDiffusionZ = Object.ErrorDiffusionZ / sqrt(8)
    Object.LongitudinalDiffusion = Object.DiffusionZ
    Object.TransverseDiffusion = (Object.DiffusionX + Object.DiffusionY) / 2
    # CONVERT CM/SEC
    Object.VelocityZ *= 1e5
    Object.LongitudinalDiffusionError = Object.ErrorDiffusionZ
    Object.TransverseDiffusionError = (Object.ErrorDiffusionX + Object.ErrorDiffusionY) / 2.0

    Attachment = 0.0
    Ionization = 0.0
    for I in range(Object.NumberOfGases):
        Attachment += Object.CollisionsPerGasPerTypeNT[5 * (I + 1) - 3]
        Ionization += Object.CollisionsPerGasPerTypeNT[5 * (I + 1) - 4]
    Ionization += IEXTRA
    Object.AttachmentRateError = 0.0
    if Attachment != 0:
        Object.AttachmentRateError = 100 * sqrt(Attachment) / Attachment
    Object.AttachmentRate = Attachment / (Object.ST * Object.VelocityZ) * 1e12
    Object.IonisationRateError = 0.0
    if Ionization != 0:
        Object.IonisationRateError = 100 * sqrt(Ionization) / Ionization
    Object.IonisationRate = Ionization / (Object.ST * Object.VelocityZ) * 1e12

    return



