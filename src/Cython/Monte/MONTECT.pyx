from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
from libc.string cimport memset
from PyBoltz cimport drand48
from MBSorts cimport MBSortT
import numpy as np
cimport numpy as np
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
cdef void GenerateMaxBoltz(double RandomSeed,  double *RandomMaxBoltzArray):
    cdef double Ran1, Ran2, TwoPi
    cdef int J
    for J in range(0, 5, 2):
        Ran1 = random_uniform(RandomSeed)
        Ran2 = random_uniform(RandomSeed)
        TwoPi = 2.0 * np.pi
        RandomMaxBoltzArray[J] = sqrt(-1 * log(Ran1)) * cos(Ran2 * TwoPi)
        RandomMaxBoltzArray[J + 1] = sqrt(-1 * log(Ran1)) * sin(Ran2 * TwoPi)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at temp =  TemperatureCentigrade.

    This function is used for any magnetic field electric field in the z direction.    
    
    The object parameter is the PyBoltz object to have the output results and to be used in the simulation.
    """
    Object.VelocityX = 0.0
    Object.VelocityErrorX = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    cdef long long I, NumDecorLengths,  NumCollisions, IEXTRA, IMBPT, K, J, iCollisionM, iSample, iCollision, GasIndex, IE, IT, CollsToLookBack, IPT, iCorr, DecorDistance
    cdef double ST1, RandomSeed, ST2, SumE2, SumXX, SumYY, SumZZ, SumXZ, SumXY, Z_LastSample, ST_LastSample, ST1_LastSample, ST2_LastSample, SumZZ_LastSample, SumXX_LastSample, SumYY_LastSample, SumYZ_LastSample, SumXY_LastSample, SXZ_LastSample, SME2_LastSample, TDash
    cdef double ABSFAKEI, DirCosineZ1, DirCosineX1, DirCosineY1, VelXBefore, VelYBefore, VelZBefore, BP, F1, F2, TwoPi, DirCosineX2, DirCosineY2, DirCosineZ2, VelXAfter, VelYAfter, VelZAfter, DZCOM, DYCOM, DXCOM, Theta,
    cdef double  EBefore, Sqrt2M, TwoM, AP, CONST6, RandomNum, GasVelX, GasVelY, GasVelZ, VEX, VEY, VEZ, COMEnergy, Test1, Test2, Test3, VelBeforeM1
    cdef double T2, A, B, CONST7, S1, EI, R9, EXTRA, RAN, RandomNum1, CosTheta, EPSI, Phi, SinPhi, CosPhi, ARG1, D, Q, CosZAngle, U, CosSquareTheta, SinZAngle, VXLAB, VYLAB, VZLAB
    cdef double SumV_Samples, SumE_Samples, SumV2_Samples, SumE2_Samples, SumDXX_Samples, SumDYY_Samples, SumDZZ_Samples, TXCollY, TXCollZ, TYCollZ, SumDXX2_Samples, SumDYY2_Samples, SumDZZ2_Samples, T2XCollY, T2XCollZ, T2YCollZ, Attachment, Ionization, E, SumYZ, SumLS, SumTS
    cdef double SLN_LastSample, STR_LastSample, MeanEnergy_LastSample, EFZ100, EFX100, MeanEnergy, WZR, WYR, WXR, XR, ZR, YR, TDriftVelPerSampleY, TWCollX, T2DriftVelPerSampleY, T2WCollX
    cdef double *CollT, *CollX, *CollY, *CollZ, *DriftVelPerSampleZ, *MeanEnergyPerSample, *DiffZZPerSample, *DiffYYPerSample, *DiffXXPerSample, *DiffYZPerSample, *DFXCollY, *DiffXZPerSample, *DriftVelPerSampleYZ, *WXCollZ
    cdef double DIFXXR, DIFYYR, DIFZZR, DIFYZR, DIFXZR, DIFXYR, ZR_LastSample, YR_LastSample, XR_LastSample, SumZZR, SumYYR, SumXXR, SumXYR, SXZR, RCS, RSN, RTHETA, EOVBR



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

    WCollX = <double *> malloc(10 * sizeof(double))
    memset(WCollX, 0, 10 * sizeof(double))

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

    DFXCollY = <double *> malloc(10 * sizeof(double))
    memset(DFXCollY, 0, 10 * sizeof(double))

    DiffXZPerSample = <double *> malloc(10 * sizeof(double))
    memset(DiffXZPerSample, 0, 10 * sizeof(double))

    DIFXXR = 0.0
    DIFYYR = 0.0
    DIFZZR = 0.0
    DIFYZR = 0.0
    DIFXZR = 0.0
    I = 0
    DIFXYR = 0.0
    Object.TimeSum = 0.0
    ST1 = 0.0
    SumXX = 0.0
    SumYY = 0.0
    SumZZ = 0.0
    SumYZ = 0.0
    SumXY = 0.0
    SumXZ = 0.0
    ZR_LastSample = 0.0
    YR_LastSample = 0.0
    XR_LastSample = 0.0
    SumZZR = 0.0
    SumYYR = 0.0
    SumXXR = 0.0
    SumXYR = 0.0
    SumYZR = 0.0
    SXZR = 0.0
    ST_LastSample = 0.0
    ST1_LastSample = 0.0
    ST2_LastSample = 0.0
    SumZZ_LastSample = 0.0
    SumYY_LastSample = 0.0
    SumXX_LastSample = 0.0
    SumYZ_LastSample = 0.0
    SumXY_LastSample = 0.0
    SXZ_LastSample = 0.0

    MeanEnergy_LastSample = 0.0

    # CALC ROTATION MATRIX ANGLES
    RCS = cos((Object.BFieldAngle - 90) * np.pi / 180)
    RSN = sin((Object.BFieldAngle - 90) * np.pi / 180)

    RTHETA = Object.BFieldAngle * np.pi / 180
    EFZ100 = Object.EField * 100 * sin(RTHETA)
    EFX100 = Object.EField * 100 * cos(RTHETA)

    F1 = Object.EField * Object.CONST2 * sin(RTHETA)
    TwoPi = 2 * np.pi
    Sqrt2M = Object.CONST3 * 0.01
    TwoM = Sqrt2M ** 2
    EOVBR = Object.EFieldOverBField * sin(RTHETA)
    EBefore = Object.InitialElectronEnergy

    NumDecorLengths = 0
    NumCollisions = 0
    IEXTRA = 0
    cdef double ** TEMP = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TEMP[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TEMP[K][J] = Object.TotalCollisionFrequency[K][J] + Object.TotalCollisionFrequencyNull[K][J]
    GenerateMaxBoltz(Object.RandomSeed,  Object.RandomMaxBoltzArray)
    ABSFAKEI = Object.FakeIonizations
    Object.FakeIonizations = 0

    IMBPT = 0
    TDash = 0.0

    # INTIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    # INITIAL VELOCITY
    VelTotal = Sqrt2M * sqrt(EBefore)
    VelXBefore = DirCosineX1 * VelTotal
    VelYBefore = DirCosineY1 * VelTotal
    VelZBefore = DirCosineZ1 * VelTotal
    RandomSeed = Object.RandomSeed

    iCollisionM = <long long>(Object.MaxNumberOfCollisions / Object.NumSamples)
    if Object.ConsoleOutputFlag:
        print('{:^12s}{:^12s}{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity Z", "Velocity Y", "Velocity X","Energy",
                                                                       "DIFXX", "DIFYY", "DIFZZ", "DIFYZ","DIFXZ","DIFXY"))
    for iSample in range(int(Object.NumSamples)):
        for iCollision in range(int(iCollisionM)):
            while True:
                RandomNum = random_uniform(RandomSeed)
                T = -1 * log(RandomNum) / Object.MaxCollisionFreqTotal + TDash
                TDash = T
                WBT = Object.AngularSpeedOfRotation * T
                CosWT = cos(WBT)
                SinWT = sin(WBT)
                DZ = (VelZBefore * SinWT + (EOVBR - VelYBefore) * (1 - CosWT)) / Object.AngularSpeedOfRotation

                DX = VelXBefore * T + F1 * T * T

                E = EBefore + DZ * EFZ100 + DX * EFX100

                # CALCULATE ELECTRON VELOCITY IN LAB FRAME
                VelXAfter = VelXBefore + 2 * F1 * T
                VelYAfter = (VelYBefore - EOVBR) * CosWT + VelZBefore * SinWT + EOVBR
                VelZAfter = VelZBefore * CosWT - (VelYBefore - EOVBR) * SinWT

                # FIND NumDecorLengthsENTITY OF GAS FOR COLLISION
                GasIndex = 0
                RandomNum = random_uniform(RandomSeed)
                if Object.NumberOfGases == 1:
                    GasIndex = 0
                else:
                    while (Object.MaxCollisionFreqTotalG[GasIndex] < RandomNum):
                        GasIndex = GasIndex + 1

                IMBPT += 1
                if (IMBPT > 6):
                    GenerateMaxBoltz(Object.RandomSeed,  Object.RandomMaxBoltzArray)
                    IMBPT = 1
                GasVelX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
                IMBPT += 1
                GasVelY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
                IMBPT += 1
                GasVelZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]

                # CALCULATE ENERGY WITH STATIONARY GAS TARGET,COMEnergy
                COMEnergy = ((VelXAfter - GasVelX) ** 2 + (VelYAfter - GasVelY) ** 2 + (VelZAfter - GasVelZ) ** 2) / TwoM
                IE = int(COMEnergy / Object.ElectronEnergyStep)
                IE = min(IE, 3999)

                # Test FOR REAL OR NULL COLLISION
                RandomNum = random_uniform(RandomSeed)
                Test1 = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
                if RandomNum > Test1:
                    Test2 = TEMP[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
                    if RandomNum < Test2:
                        # Test FOR NULL LEVELS
                        if Object.NumMomCrossSectionPointsNull[GasIndex] == 0:
                            continue
                        RandomNum = random_uniform(RandomSeed)
                        I = 0
                        while Object.NullCollisionFreq[GasIndex][IE][I] < RandomNum:
                            # INCREMENT NULL SCATTER Sum
                            I += 1

                        Object.ICOLNN[GasIndex][I] += 1
                        continue
                    else:
                        Test3 = (TEMP[GasIndex][IE] + ABSFAKEI) / Object.MaxCollisionFreq[GasIndex]
                        if RandomNum < Test3:
                            # FAKE IONISATION INCREMENT COUNTER
                            Object.FakeIonizations += 1
                            continue
                        continue
                else:
                    break

            NumCollisions += 1

            Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T

            # CALCULATE DIRECTION COSINES OF ELECTRON IN 0 KELVIN FRAME
            VelBeforeM1 = 1 / (Sqrt2M * sqrt(COMEnergy))
            #     VelTotal=1.0D0/VelBeforeM1
            DXCOM = (VelXAfter - GasVelX) * VelBeforeM1
            DYCOM = (VelYAfter - GasVelY) * VelBeforeM1
            DZCOM = (VelZAfter - GasVelZ) * VelBeforeM1
            #  CALCULATE POSITIONS AT INSTANT BEFORE COLLISION
            #    ALSO UPDATE DIFFUSION  AND ENERGY CALCULATIONS.
            T2 = T ** 2
            TDash = 0.0

            Object.X += DX
            Object.Y += EOVBR * T + ((VelYBefore - EOVBR) * SinWT + VelZBefore * (1 - CosWT)) / Object.AngularSpeedOfRotation
            Object.Z += DZ
            Object.TimeSum += T
            IT = int(T)
            IT = min(IT, 299)
            Object.CollisionTimes[IT] += 1
            Object.CollisionEnergies[IE] += 1
            Object.VelocityZ = Object.Z / Object.TimeSum
            Object.VelocityY = Object.Y / Object.TimeSum
            Object.VelocityX = Object.X / Object.TimeSum
            if iSample >= 2:
                CollsToLookBack = 0
                for J in range(int(Object.Decor_LookBacks)):
                    DecorDistance = NumCollisions + CollsToLookBack
                    if DecorDistance > Object.Decor_Colls:
                        DecorDistance = DecorDistance - Object.Decor_Colls
                    ST1 += T
                    TDiff = Object.TimeSum - CollT[DecorDistance-1]
                    CollsToLookBack += Object.Decor_Step
                    SumZZ += ((Object.Z - CollZ[DecorDistance-1] - Object.VelocityZ * TDiff) ** 2) * T / TDiff
                    SumYY += ((Object.Y - CollY[DecorDistance-1] - Object.VelocityY * TDiff) ** 2) * T / TDiff
                    SumXX += ((Object.X - CollX[DecorDistance-1] - Object.VelocityX * TDiff) ** 2) * T / TDiff
                    SumYZ += (Object.Z - CollZ[DecorDistance-1] - Object.VelocityZ * TDiff) * (
                            Object.Y - CollY[DecorDistance-1] - Object.VelocityY * TDiff) * T / TDiff
                    SumXY += (Object.X - CollX[DecorDistance-1] - Object.VelocityX * TDiff) * (
                            Object.Y - CollY[DecorDistance-1] - Object.VelocityY * TDiff) * T / TDiff
                    SumXZ += (Object.X - CollX[DecorDistance-1] - Object.VelocityX * TDiff) * (
                            Object.Z - CollZ[DecorDistance-1] - Object.VelocityZ * TDiff) * T / TDiff
            CollX[NumCollisions-1] = Object.X
            CollY[NumCollisions-1] = Object.Y
            CollZ[NumCollisions-1] = Object.Z
            CollT[NumCollisions-1] = Object.TimeSum

            if NumCollisions >= Object.Decor_Colls:
                NumDecorLengths += 1
                NumCollisions = 0
            # DETERMENATION OF REAL COLLISION TYPE
            RandomNum = random_uniform(RandomSeed)

            # FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
            I = MBSortT(GasIndex, I, RandomNum, IE, Object)
            while Object.CollisionFrequency[GasIndex][IE][I] < RandomNum:
                I += 1

            S1 = Object.RGas[GasIndex][I]
            EI = Object.EnergyLevels[GasIndex][I]

            if Object.ElectronNumChange[GasIndex][I] > 0:
                #  USE FLAT DISTRIBUTION OF  ELECTRON ENERGY BETWEEN E-IonizationEnergy AND 0.0 EV
                #  SAME AS IN BOLTZMMoleculesPerCm3PerGas
                R9 = random_uniform(RandomSeed)
                EXTRA = R9 * (COMEnergy - EI)
                EI = EXTRA + EI
                # IF FLOUORESCENCE OR AUGER ADD EXTRA ELECTRONS
                IEXTRA += <long long>Object.NC0[GasIndex][I]

            #  GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
            #   COLLISION ALSO UPDATE ENERGY OF ELECTRON.
            IPT = <long long>Object.InteractionType[GasIndex][I]
            Object.CollisionsPerGasPerType[GasIndex][int(IPT)] += 1
            Object.ICOLN[GasIndex][I] += 1
            if COMEnergy < EI:
                EI = COMEnergy - 0.0001
            #IF EXCITATION THEN ADD PROBABILITY,PenningFractionC(1,I), OF TRANSFER TO GIVE
            # IONISATION OF THE OTHER GASES IN THE MIXTURE
            if Object.EnablePenning != 0:
                if Object.PenningFraction[GasIndex][0][I] != 0:
                    RAN = random_uniform(RandomSeed)
                    if RAN <= Object.PenningFraction[GasIndex][0][I]:
                        # ADD EXTRA IONISATION COLLISION
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

            # AAnisotropicDetectedTROPIC SCATTERING
            RandomNum = random_uniform(RandomSeed)
            if Object.AngularModel[GasIndex][I] == 1:
                RandomNum1 = random_uniform(RandomSeed)
                CosTheta = 1.0 - RandomNum * Object.AngleCut[GasIndex][IE][I]
                if RandomNum1 > Object.ScatteringParameter[GasIndex][IE][I]:
                    CosTheta = -1 * CosTheta
            elif Object.AngularModel[GasIndex][I] == 2:
                EPSI = Object.ScatteringParameter[GasIndex][IE][I]
                CosTheta = 1 - (2 * RandomNum * (1 - EPSI) / (1 + EPSI * (1 - 2 * RandomNum)))
            else:
                #ISOTROPIC SCATTERING
                CosTheta = 1 - 2 * RandomNum

            Theta = acos(CosTheta)
            RandomNum = random_uniform(RandomSeed)
            Phi = TwoPi * RandomNum
            SinPhi = sin(Phi)
            CosPhi = cos(Phi)
            ARG1 = 1 - S1 * EI / COMEnergy
            ARG1 = max(ARG1, Object.SmallNumber)
            D = 1 - CosTheta * sqrt(ARG1)
            EBefore = COMEnergy * (1 - EI / (S1 * COMEnergy) - 2 * D / S2)
            EBefore = max(EBefore, Object.SmallNumber)
            Q = sqrt((COMEnergy / EBefore) * ARG1) / S1
            Q = min(Q, 1)
            Object.AngleFromZ = asin(Q * sin(Theta))
            CosZAngle = cos(Object.AngleFromZ)
            U = (S1 - 1) * (S1 - 1) / ARG1

            CosSquareTheta = CosTheta * CosTheta
            if CosTheta < 0 and CosSquareTheta > U:
                CosZAngle = -1 * CosZAngle
            SinZAngle = sin(Object.AngleFromZ)
            DZCOM = min(DZCOM, 1)
            ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
            if ARGZ == 0:
                DirCosineZ1 = CosZAngle
                DirCosineX1 = CosPhi * SinZAngle
                DirCosineY1 = SinPhi * SinZAngle
            else:
                DirCosineZ1 = DZCOM * CosZAngle + ARGZ * SinZAngle * SinPhi
                DirCosineY1 = DYCOM * CosZAngle + (SinZAngle / ARGZ) * (DXCOM * CosPhi - DYCOM * DZCOM * SinPhi)
                DirCosineX1 = DXCOM * CosZAngle - (SinZAngle / ARGZ) * (DYCOM * CosPhi + DXCOM * DZCOM * SinPhi)

            #TRANSFORM VELOCITY VECTORS TO LAB FRAME
            VelTotal = Sqrt2M * sqrt(EBefore)
            VelXBefore = DirCosineX1 * VelTotal + GasVelX
            VelYBefore = DirCosineY1 * VelTotal + GasVelY
            VelZBefore = DirCosineZ1 * VelTotal + GasVelZ
            # CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
            EBefore = (VelXBefore * VelXBefore + VelYBefore * VelYBefore + VelZBefore * VelZBefore) / TwoM
            VelBeforeM1 = 1 / (Sqrt2M * sqrt(EBefore))
            DirCosineX1 = VelXBefore * VelBeforeM1
            DirCosineY1 = VelYBefore * VelBeforeM1
            DirCosineZ1 = VelZBefore * VelBeforeM1

        Object.VelocityZ *= 1e9
        Object.VelocityY *= 1e9
        Object.VelocityX *= 1e9

        # CALCULATE ROTATED VECTORS AND POSITIONS
        WZR = Object.VelocityZ * RCS - Object.VelocityX * RSN
        WYR = Object.VelocityY
        WXR = Object.VelocityZ * RSN + Object.VelocityX * RCS
        ZR = Object.Z * RCS - Object.X * RSN
        YR = Object.Y
        XR = Object.Z * RSN + Object.X * RCS
        MeanEnergy = 0.0
        for IK in range(4000):
            TotalCollisionFrequencySum = 0.0
            for KI in range(Object.NumberOfGases):
                TotalCollisionFrequencySum += Object.TotalCollisionFrequency[KI][IK]
            MeanEnergy += Object.E[IK] * Object.CollisionEnergies[IK] / TotalCollisionFrequencySum
        Object.MeanElectronEnergy = MeanEnergy / Object.TimeSum
        DriftVelPerSampleZ[iSample] = (ZR - ZR_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        DriftVelPerSampleY[iSample] = (YR - YR_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        WCollX[iSample] = (XR - XR_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        MeanEnergyPerSample[iSample] = (MeanEnergy - MeanEnergy_LastSample) / (Object.TimeSum - ST_LastSample)
        MeanEnergy_LastSample = MeanEnergy

        if iSample >= 2:
            Object.DiffusionX = 5e15 * SumXX / ST1
            Object.DiffusionY = 5e15 * SumYY / ST1
            Object.DiffusionZ = 5e15 * SumZZ / ST1
            Object.DiffusionXY = 5e15 * SumXY / ST1
            Object.DiffusionYZ = 5e15 * SumYZ / ST1
            Object.DiffusionXZ = 5e15 * SumXZ / ST1
            # CALCULATE  ROTATED TENSOR
            DIFXXR = Object.DiffusionX * RCS * RCS + Object.DiffusionZ * RSN * RSN + 2 * RCS * RSN * Object.DiffusionXZ
            DIFYYR = Object.DiffusionY
            DIFZZR = Object.DiffusionX * RSN * RSN + Object.DiffusionZ * RCS * RCS - 2 * RCS * RSN * Object.DiffusionXZ
            DIFXYR = RCS * Object.DiffusionXY + RSN * Object.DiffusionYZ
            DIFYZR = RSN * Object.DiffusionXY - RCS * Object.DiffusionYZ
            DIFXZR = (RCS * RCS - RSN * RSN) * Object.DiffusionXZ - RSN * RCS * (Object.DiffusionX - Object.DiffusionZ)

            SumXXR = SumXX * RCS * RCS + SumZZ * RSN * RSN + 2 * RCS * RSN * SumXZ
            SumYYR = SumYY
            SumZZR = SumXX * RSN * RSN + SumZZ * RCS * RCS - 2 * RCS * RSN * SumXZ
            SumXYR = RCS * SumXY + RSN * SumYZ
            SumYZR = RSN * SumXY - RCS * SumYZ
            SXZR = (RCS * RCS - RSN * RSN) * SumXZ - RSN * RCS * (SumXX - SumZZ)
        DiffZZPerSample[iSample] = 0.0
        DiffXXPerSample[iSample] = 0.0
        DiffYYPerSample[iSample] = 0.0
        DiffYZPerSample[iSample] = 0.0
        DiffXZPerSample[iSample] = 0.0
        DFXCollY[iSample] = 0.0
        if iSample > 1:
            DiffZZPerSample[iSample] = 5e15 * (SumZZR - SumZZ_LastSample) / (ST1 - ST1_LastSample)
            DiffXXPerSample[iSample] = 5e15 * (SumXXR - SumXX_LastSample) / (ST1 - ST1_LastSample)
            DiffYYPerSample[iSample] = 5e15 * (SumYYR - SumYY_LastSample) / (ST1 - ST1_LastSample)
            DiffYZPerSample[iSample] = 5e15 * (SumYZR - SumYZ_LastSample) / (ST1 - ST1_LastSample)
            DiffXZPerSample[iSample] = 5e15 * (SXZR - SXZ_LastSample) / (ST1 - ST1_LastSample)
            DFXCollY[iSample] = 5e15 * (SumXYR - SumXY_LastSample) / (ST1 - ST1_LastSample)
        ZR_LastSample = ZR
        YR_LastSample = YR
        XR_LastSample = XR
        ST_LastSample = Object.TimeSum
        ST1_LastSample = ST1
        SumZZ_LastSample = SumZZR
        SumYY_LastSample = SumYYR
        SumXX_LastSample = SumXXR
        SumXY_LastSample = SumXYR
        SumYZ_LastSample = SumYZR
        SXZ_LastSample = SXZR
        if Object.ConsoleOutputFlag:
            print('{:^12.1f}{:^12.1f}{:^12.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(WZR,WYR,WXR,
                                                                                    Object.MeanElectronEnergy, DIFXXR, DIFYYR,
                                                                                    DIFZZR,DIFYZR,DIFXZR,DIFXYR))

    #CALCULATE ERRORS AND CHECK AVERAGES

    SumV_Samples = 0.0
    TDriftVelPerSampleY = 0.0
    TWCollX = 0.0
    SumE_Samples = 0.0
    SumV2_Samples = 0.0
    T2DriftVelPerSampleY = 0.0
    T2WCollX = 0.0
    SumE2_Samples = 0.0
    SumDZZ_Samples = 0.0
    SumDYY_Samples = 0.0
    SumDXX_Samples = 0.0
    TXCollY = 0.0
    TXCollZ = 0.0
    TYCollZ = 0.0
    SumDZZ2_Samples = 0.0
    SumDYY2_Samples = 0.0
    SumDXX2_Samples = 0.0
    T2XCollY = 0.0
    T2XCollZ = 0.0
    T2YCollZ = 0.0

    for K in range(10):
        SumV_Samples = SumV_Samples + DriftVelPerSampleZ[K]
        TDriftVelPerSampleY = TDriftVelPerSampleY + DriftVelPerSampleY[K]
        TWCollX = TWCollX + WCollX[K]
        SumE_Samples = SumE_Samples + MeanEnergyPerSample[K]
        SumV2_Samples = SumV2_Samples + DriftVelPerSampleZ[K] * DriftVelPerSampleZ[K]
        T2DriftVelPerSampleY = T2DriftVelPerSampleY + DriftVelPerSampleY[K] * DriftVelPerSampleY[K]
        T2WCollX = T2WCollX + WCollX[K] * WCollX[K]
        SumE2_Samples = SumE2_Samples + MeanEnergyPerSample[K] * MeanEnergyPerSample[K]
        if K >= 2:
            SumDZZ_Samples = SumDZZ_Samples + DiffZZPerSample[K]
            SumDYY_Samples = SumDYY_Samples + DiffYYPerSample[K]
            SumDXX_Samples = SumDXX_Samples + DiffXXPerSample[K]
            TYCollZ = TYCollZ + DiffYZPerSample[K]
            TXCollY = TXCollY + DFXCollY[K]
            TXCollZ = TXCollZ + DiffXZPerSample[K]

            SumDZZ2_Samples += DiffZZPerSample[K] ** 2
            SumDXX2_Samples += DiffXXPerSample[K] ** 2
            SumDYY2_Samples += DiffYYPerSample[K] ** 2
            T2YCollZ += DiffYZPerSample[K] ** 2
            T2XCollY += DFXCollY[K] ** 2
            T2XCollZ += DiffXZPerSample[K] ** 2
    Object.VelocityErrorZ = 100 * sqrt((SumV2_Samples - SumV_Samples * SumV_Samples / 10.0) / 9.0) / WZR
    Object.VelocityErrorY = 100 * sqrt((T2DriftVelPerSampleY - TDriftVelPerSampleY * TDriftVelPerSampleY / 10.0) / 9.0) / abs(WYR)
    Object.VelocityErrorX = 100 * sqrt((T2WCollX - TWCollX * TWCollX / 10.0) / 9.0) / abs(WXR)
    Object.MeanElectronEnergyError = 100 * sqrt((SumE2_Samples - SumE_Samples * SumE_Samples / 10.0) / 9.0) / Object.MeanElectronEnergy
    Object.ErrorDiffusionZ = 100 * sqrt((SumDZZ2_Samples - SumDZZ_Samples * SumDZZ_Samples / 8.0) / 7.0) / DIFZZR
    Object.ErrorDiffusionY = 100 * sqrt((SumDYY2_Samples - SumDYY_Samples * SumDYY_Samples / 8.0) / 7.0) / DIFYYR
    Object.ErrorDiffusionX = 100 * sqrt((SumDXX2_Samples - SumDXX_Samples * SumDXX_Samples / 8.0) / 7.0) / DIFXXR
    Object.ErrorDiffusionXY = 100 * sqrt((T2XCollY - TXCollY * TXCollY / 8.0) / 7.0) / abs(DIFXYR)
    Object.ErrorDiffusionXZ = 100 * sqrt((T2XCollZ - TXCollZ * TXCollZ / 8.0) / 7.0) / abs(DIFXZR)
    Object.ErrorDiffusionYZ = 100 * sqrt((T2YCollZ - TYCollZ * TYCollZ / 8.0) / 7.0) / abs(DIFYZR)

    Object.VelocityErrorZ = Object.VelocityErrorZ / sqrt(10)
    Object.VelocityErrorX = Object.VelocityErrorX / sqrt(10)
    Object.VelocityErrorY = Object.VelocityErrorY / sqrt(10)
    Object.MeanElectronEnergyError = Object.MeanElectronEnergyError / sqrt(10)
    Object.ErrorDiffusionX = Object.ErrorDiffusionX / sqrt(8)
    Object.ErrorDiffusionY = Object.ErrorDiffusionY / sqrt(8)
    Object.ErrorDiffusionZ = Object.ErrorDiffusionZ / sqrt(8)
    Object.ErrorDiffusionYZ = Object.ErrorDiffusionYZ / sqrt(8)
    Object.ErrorDiffusionXY = Object.ErrorDiffusionXY / sqrt(8)
    Object.ErrorDiffusionXZ = Object.ErrorDiffusionXZ / sqrt(8)

    #LOAD ROTATED VALUES INTO ARRAYS

    Object.VelocityZ = WZR
    Object.VelocityX = WXR
    Object.VelocityY = WYR
    Object.DiffusionX = DIFXXR
    Object.DiffusionY = DIFYYR
    Object.DiffusionZ = DIFZZR
    Object.DiffusionYZ = DIFYZR
    Object.DiffusionXY = DIFXYR
    Object.DiffusionXZ = DIFXZR

    #CONVERT TO CM/SEC.
    Object.VelocityZ *= 1e5
    Object.VelocityY *= 1e5
    Object.VelocityX *= 1e5

    #CALCULATE TOWNSEND COEFICIENTS AND ERRORS
    Attachment = 0.0
    Ionization = 0.0
    for I in range(Object.NumberOfGases):
        Attachment += Object.CollisionsPerGasPerType[I][2]
        Ionization += Object.CollisionsPerGasPerType[I][1]
    Ionization += IEXTRA
    Object.AttachmentRateError = 0.0

    if Attachment != 0:
        Object.AttachmentRateError = 100 * sqrt(Attachment) / Attachment
    Object.AttachmentRate = Attachment / (Object.TimeSum * Object.VelocityZ) * 1e12
    Object.IonisationRateError = 0.0
    if Ionization != 0:
        Object.IonisationRateError = 100 * sqrt(Ionization) / Ionization
    Object.IonisationRate = Ionization / (Object.TimeSum * Object.VelocityZ) * 1e12
