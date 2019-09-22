from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow
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
cdef void GenerateMaxBoltz(double RandomSeed, double *RandomMaxBoltzArray):
    cdef double RAN1, RAN2, TWOPI
    cdef int J
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RandomSeed)
        RAN2 = random_uniform(RandomSeed)
        TWOPI = 2.0 * np.pi
        RandomMaxBoltzArray[J] = sqrt(-1*log(RAN1)) * cos(RAN2 * TWOPI)
        RandomMaxBoltzArray[J + 1] = sqrt(-1*log(RAN1)) * sin(RAN2 * TWOPI)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at temp =  TemperatureCentigrade.

    This function is used when the magnetic field is perpendicular to the electric field in the z direction.    
    
    The object parameter is the PyBoltz object to have the output results and to be used in the simulation.
    """
    Object.VelocityX = 0.0
    Object.VelocityErrorX = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.DiffusionXZ = 0.0
    Object.DiffusionXY = 0.0
    cdef long long I, NumDecorLengths,  NCOL, IEXTRA, IMBPT, K, J, iCollisionM, iSample, iCollision, GasIndex, IE, IT, CollsToLookBack, IPT, iCorr,NC_LastSampleM
    cdef double ST1, RandomSeed,ST2, SumE2, SumXX, SumYY, SumZZ, SumVX, SumVY, Z_LastSample, ST_LastSample, ST1_LastSample, ST2_LastSample, SZZ_LastSample, SXX_LastSample, SYY_LastSample, SVX_LastSample, SVY_LastSample, SME2_LastSample, TDash
    cdef double ABSFAKEI, DirCosineZ1, DirCosineX1, DirCosineY1, CX1, CY1, CZ1, BP, F1, F2, TwoPi, DirCosineX2, DirCosineY2, DirCosineZ2, CX2, CY2, CZ2, DZCOM, DYCOM, DXCOM, THETA0,
    cdef double  EBefore, Sqrt2M, TwoM, AP, CONST6, RandomNum, VGX, VGY, VGZ, VEX, VEY, VEZ, COMEnergy, Test1, Test2, Test3, CONST11
    cdef double T2, A, B, CONST7, S1, EI, R9, EXTRA, RAN, RandomNum1, F3, EPSI, PHI0, F8, F9, ARG1, D, Q, F6, U, CSQD, F5, VXLAB, VYLAB, VZLAB
    cdef double SumV_Samples, SumE_Samples, SumV2_Samples, SumE2_Samples, SumDXX_Samples, SumDYY_Samples, SumDXX2_Samples, SumDYY2_Samples, SumDZZ_Samples, SumDZZ2_Samples, Attachment, Ionization, E,SumYZ,SumLS,SumTS
    cdef double SYZ_LastSample,SLN_LastSample,STR_LastSample,EBAR_LastSample,EF100, EBAR
    cdef double *STO, *XST, *YST, *ZST, *WZST, *AVEST, *DFZZST, *DFYYST, *DFXXST,*DFYZST,*DFLNST,*DFTRNST,*WYZST, *DFTRST



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

    WYST = <double *> malloc(10 * sizeof(double))
    memset(WYST, 0, 10 * sizeof(double))

    AVEST = <double *> malloc(10 * sizeof(double))
    memset(AVEST, 0, 10 * sizeof(double))

    DFZZST = <double *> malloc(10 * sizeof(double))
    memset(DFZZST, 0, 10 * sizeof(double))

    DFYYST = <double *> malloc(10 * sizeof(double))
    memset(DFYYST, 0, 10 * sizeof(double))

    DFXXST = <double *> malloc(10 * sizeof(double))
    memset(DFXXST, 0, 10 * sizeof(double))

    DFYZST = <double *> malloc(10 * sizeof(double))
    memset(DFYZST, 0, 10 * sizeof(double))

    DFLNST = <double *> malloc(10 * sizeof(double))
    memset(DFLNST, 0, 10 * sizeof(double))

    DFTRNST = <double *> malloc(10 * sizeof(double))
    memset(DFTRNST, 0, 10 * sizeof(double))

    DFTRST  = <double *> malloc(10 * sizeof(double))
    memset(DFTRST, 0, 10 * sizeof(double))

    Object.ErrorDiffusionXZ = 0.0
    Object.ErrorDiffusionXY = 0.0
    Object.TimeSum = 0.0
    ST1 = 0.0
    ST2 = 0.0
    SumXX = 0.0
    SumYY = 0.0
    SumZZ = 0.0
    SumYZ = 0.0
    I=0
    SumLS = 0.0
    SumTS = 0.0
    SumVX = 0.0
    Z_LastSample = 0.0
    Y_LastSample = 0.0
    ST_LastSample = 0.0
    ST1_LastSample = 0.0
    ST2_LastSample = 0.0
    SZZ_LastSample = 0.0
    SXX_LastSample = 0.0
    SYY_LastSample = 0.0
    SYZ_LastSample = 0.0
    SVX_LastSample = 0.0
    SLN_LastSample = 0.0
    STR_LastSample = 0.0
    EBAR_LastSample = 0.0
    EF100 = Object.EField * 100
    EBefore = Object.InitialElectronEnergy
    Sqrt2M = Object.CONST3 * 0.01
    TwoM = Sqrt2M ** 2
    NumDecorLengths = 0
    NCOL = 0
    IEXTRA = 0
    cdef double ** TEMP = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TEMP[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TEMP[K][J] = Object.TotalCollisionFrequency[K][J] + Object.TotalCollisionFrequencyNull[K][J]
    ABSFAKEI = Object.FakeIonizations
    Object.FakeIonizations = 0
    GenerateMaxBoltz(Object.RandomSeed,  Object.RandomMaxBoltzArray)
    IMBPT = 0
    TDash = 0.0
    TwoPi = 2 * acos(-1)
    # INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)
    # INITIAL VELOCITY
    VTOT = Sqrt2M * sqrt(EBefore)
    CX1 = DirCosineX1 * VTOT
    CY1 = DirCosineY1 * VTOT
    CZ1 = DirCosineZ1 * VTOT
    RandomSeed = Object.RandomSeed
    iCollisionM = <long long>(Object.MaxNumberOfCollisions / Object.NumSamples)
    if Object.ConsoleOutputFlag:
        print('{:^12s}{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity Z", "Velocity Y", "Energy",
                                                                       "DIFXX", "DIFYY", "DIFZZ", "DIFYZ","DIFLNG","DIFTRN"))
    for iSample in range(int(Object.NumSamples)):
        for iCollision in range(int(iCollisionM)):
            while True:
                RandomNum = random_uniform(RandomSeed)
                T = -1 * log(RandomNum) / Object.MaxCollisionFreqTotal + TDash
                TDash = T
                WBT = Object.AngularSpeedOfRotation * T
                COSWT = cos(WBT)
                SINWT = sin(WBT)
                DZ = (CZ1 * SINWT + (Object.EFieldOverBField - CY1) * (1 - COSWT)) / Object.AngularSpeedOfRotation

                E = EBefore + DZ * EF100
                # CALCULATE ELECTRON VELOCITY IN LAB FRAME
                CX2 = CX1
                CY2 = (CY1 - Object.EFieldOverBField) * COSWT + CZ1 * SINWT + Object.EFieldOverBField
                CZ2 = CZ1 * COSWT - (CY1 - Object.EFieldOverBField) * SINWT

                # FIND NumDecorLengths ENTITY OF GAS FOR COLLISION
                GasIndex = 0
                RandomNum = random_uniform(RandomSeed)
                if Object.NumberOfGases == 1:
                    GasIndex = 0
                else:
                    while (Object.MaxCollisionFreqTotalG[GasIndex] < RandomNum):
                        GasIndex = GasIndex + 1

                # CALCULATE GAS VELOCITY VECTORS VGX,VGY,VGZ
                IMBPT += 1
                if (IMBPT > 6):
                    GenerateMaxBoltz(Object.RandomSeed,  Object.RandomMaxBoltzArray)
                    IMBPT = 1
                VGX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
                IMBPT += 1
                VGY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
                IMBPT += 1
                VGZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]

                # CALCULATE ENERGY WITH STATIONARY GAS TARGET
                COMEnergy = ((CX2 - VGX) ** 2 + (CY2 - VGY) ** 2 + (CZ2 - VGZ) ** 2) / TwoM
                IE = int(COMEnergy / Object.ElectronEnergyStep)
                IE = min(IE, 3999)

                # Test FOR REAL OR NULL COLLISION
                RandomNum = random_uniform(RandomSeed)
                Test1 = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
                # Test FOR REAL OR NULL COLLISION
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
            Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T

            NCOL += 1
            #CALCULATE DIRECTION COSINES OF ELECTRON IN 0 KELVIN FRAME
            CONST11 = 1 / (Sqrt2M * sqrt(COMEnergy))
            DXCOM = (CX2 - VGX) * CONST11
            DYCOM = (CY2 - VGY) * CONST11
            DZCOM = (CZ2 - VGZ) * CONST11
            #  CALCULATE POSITIONS AT INSTANT BEFORE COLLISION
            #    ALSO UPDATE DIFFUSION  AND ENERGY CALCULATIONS.
            T2 = T ** 2
            TDash = 0.0

            #CALC NEW POSITION
            Object.X += CX1 * T
            Object.Y += Object.EFieldOverBField * T + ((CY1 - Object.EFieldOverBField) * SINWT + CZ1 * (1 - COSWT)) / Object.AngularSpeedOfRotation
            Object.Z += DZ
            Object.TimeSum += T
            IT = int(T)
            IT = min(IT, 299)
            Object.CollisionTimes[IT] += 1

            #ENERGY CollisionEnergiesTRUM FOR O KELVIN FRAME
            Object.CollisionEnergies[IE] += 1
            Object.VelocityZ = Object.Z / Object.TimeSum
            Object.VelocityY = Object.Y / Object.TimeSum
            SumVX += (CX1 ** 2) * T2
            if NumDecorLengths != 0:
                CollsToLookBack = 0
                for J in range(int(Object.Decor_LookBacks)):
                    ST2 = ST2 + T
                    NC_LastSampleM = NCOL + CollsToLookBack
                    if NC_LastSampleM > Object.Decor_Colls:
                        NC_LastSampleM = NC_LastSampleM - Object.Decor_Colls
                    TDiff = Object.TimeSum - STO[NC_LastSampleM-1]
                    SumXX += ((Object.X - XST[NC_LastSampleM-1]) ** 2) * T / TDiff
                    CollsToLookBack += Object.Decor_Step
                    if iSample >= 2:
                        ST1 += T
                        SumZZ += ((Object.Z - ZST[NC_LastSampleM-1] - Object.VelocityZ * TDiff) ** 2) * T / TDiff
                        SumYY += ((Object.Y - YST[NC_LastSampleM-1] - Object.VelocityY * TDiff) ** 2) * T / TDiff
                        SumYZ += (Object.Z - ZST[NC_LastSampleM-1] - Object.VelocityZ * TDiff) * (
                                Object.Y - YST[NC_LastSampleM-1] - Object.VelocityY * TDiff) * T / TDiff
                        A2 = (Object.VelocityZ * TDiff) ** 2 + (Object.VelocityY * TDiff) ** 2
                        B2 = (Object.Z - Object.VelocityZ * TDiff - ZST[NC_LastSampleM-1]) ** 2 + (
                                Object.Y - Object.VelocityY * TDiff - YST[NC_LastSampleM-1]) ** 2
                        C2 = (Object.Z - ZST[NC_LastSampleM-1]) ** 2 + (Object.Y - YST[NC_LastSampleM-1]) ** 2
                        DL2 = (A2 + B2 - C2) ** 2 / (4 * A2)
                        DT2 = B2 - DL2
                        SumLS += DL2 * T / TDiff
                        SumTS += DT2 * T / TDiff
            XST[NCOL-1] = Object.X
            YST[NCOL-1] = Object.Y
            ZST[NCOL-1] = Object.Z
            STO[NCOL-1] = Object.TimeSum
            if NCOL >= Object.Decor_Colls:
                NumDecorLengths += 1
                NCOL = 0
            # ---------------------------------------------------------------------
            #     DETERMINATION OF REAL COLLISION TYPE
            # ---------------------------------------------------------------------
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
                #FIX ENERGY LOSS SMALLER THAN INCNumDecorLengthsENT ENERGY IF ERROR OCCURS
                EI = COMEnergy - 0.0001

            #IF EXCITATION THEN ADD PROBABILITY,PenningFractionC(1,I), OF TRANSFER TO GIVE
            # IONISATION OF THE OTHER GASES IN THE MIXTURE
            if Object.EnablePenning != 0:
                if Object.PenningFraction[GasIndex][0][I] != 0:
                    RAN = random_uniform(RandomSeed)
                    if RAN <= Object.PenningFraction[GasIndex][0][I]:
                        #ADD EXTRA IONISATION COLLISION
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

            # AAnisotropicDetectedTROPIC SCATTERING
            RandomNum = random_uniform(RandomSeed)
            if Object.AngularModel[GasIndex][I] == 1:
                RandomNum1 = random_uniform(RandomSeed)
                F3 = 1.0 - RandomNum * Object.AngleCut[GasIndex][IE][I]
                if RandomNum1 > Object.ScatteringParameter[GasIndex][IE][I]:
                    F3 = -1 * F3
            elif Object.AngularModel[GasIndex][I] == 2:
                EPSI = Object.ScatteringParameter[GasIndex][IE][I]
                F3 = 1 - (2 * RandomNum * (1 - EPSI) / (1 + EPSI * (1 - 2 * RandomNum)))
            else:
                # ISOTROPIC SCATTERING
                F3 = 1 - 2 * RandomNum
            THETA0 = acos(F3)
            RandomNum = random_uniform(RandomSeed)
            PHI0 = TwoPi * RandomNum
            F8 = sin(PHI0)
            F9 = cos(PHI0)
            ARG1 = 1 - S1 * EI / COMEnergy
            ARG1 = max(ARG1, Object.SmallNumber)
            D = 1 - F3 * sqrt(ARG1)
            EBefore = COMEnergy * (1 - EI / (S1 * COMEnergy) - 2 * D / S2)
            EBefore = max(EBefore, Object.SmallNumber)
            Q = sqrt((COMEnergy / EBefore) * ARG1) / S1
            Q = min(Q, 1)
            Object.AngleFromZ = asin(Q * sin(THETA0))
            F6 = cos(Object.AngleFromZ)
            U = (S1 - 1) * (S1 - 1) / ARG1

            CSQD = F3 * F3
            if F3 < 0 and CSQD > U:
                F6 = -1 * F6
            F5 = sin(Object.AngleFromZ)
            DZCOM = min(DZCOM, 1)
            ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
            if ARGZ == 0:
                DirCosineZ1 = F6
                DirCosineX1 = F9 * F5
                DirCosineY1 = F8 * F5
            else:
                DirCosineZ1 = DZCOM * F6 + ARGZ * F5 * F8
                DirCosineY1 = DYCOM * F6 + (F5 / ARGZ) * (DXCOM * F9 - DYCOM * DZCOM * F8)
                DirCosineX1 = DXCOM * F6 - (F5 / ARGZ) * (DYCOM * F9 + DXCOM * DZCOM * F8)
            # TRANSFORM VELOCITY VECTORS TO LAB FRAME
            VTOT = Sqrt2M * sqrt(EBefore)
            CX1 = DirCosineX1 * VTOT + VGX
            CY1 = DirCosineY1 * VTOT + VGY
            CZ1 = DirCosineZ1 * VTOT + VGZ
            # CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
            EBefore = (CX1 * CX1 + CY1 * CY1 + CZ1 * CZ1) / TwoM
            CONST11 = 1 / (Sqrt2M * sqrt(EBefore))
            DirCosineX1 = CX1 * CONST11
            DirCosineY1 = CY1 * CONST11
            DirCosineZ1 = CZ1 * CONST11
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
        EBAR = 0.0
        for IK in range(4000):
            TotalCollisionFrequencySum = 0.0
            for KI in range(Object.NumberOfGases):
                TotalCollisionFrequencySum += Object.TotalCollisionFrequency[KI][IK]
            EBAR += Object.E[IK] * Object.CollisionEnergies[IK] / TotalCollisionFrequencySum
        Object.MeanElectronEnergy = EBAR / Object.TimeSum
        WZST[iSample] = (Object.Z - Z_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        WYST[iSample] = (Object.Y - Y_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        AVEST[iSample] = (EBAR - EBAR_LastSample) / (Object.TimeSum - ST_LastSample)
        EBAR_LastSample = EBAR
        DFZZST[iSample] = 0.0
        DFYYST[iSample] = 0.0
        DFYZST[iSample] = 0.0
        DFLNST[iSample] = 0.0
        DFTRST[iSample] = 0.0
        if iSample > 1:
            DFZZST[iSample] = 5e15 * (SumZZ - SZZ_LastSample) / (ST1 - ST1_LastSample)
            DFYYST[iSample] = 5e15 * (SumYY - SYY_LastSample) / (ST1 - ST1_LastSample)
            DFYZST[iSample] = 5e15 * (SumYZ - SYZ_LastSample) / (ST1 - ST1_LastSample)
            DFLNST[iSample] = 5e15 * (SumLS - SLN_LastSample) / (ST1 - ST1_LastSample)
            DFTRST[iSample] = 5e15 * (SumTS - STR_LastSample) / (ST1 - ST1_LastSample)
        DFXXST[iSample] = 5e15 * (SumXX - SXX_LastSample) / (ST2 - ST2_LastSample)
        if Object.AnisotropicDetected == 0:
            DFXXST[iSample] = 5e15 * (SumVX - SVX_LastSample) / (Object.TimeSum - ST_LastSample)
        Z_LastSample = Object.Z
        Y_LastSample = Object.Y
        ST_LastSample = Object.TimeSum
        ST1_LastSample = ST1
        ST2_LastSample = ST2
        SVX_LastSample = SumVX
        SZZ_LastSample = SumZZ
        SXX_LastSample = SumXX
        SYY_LastSample = SumYY
        SYZ_LastSample = SumYZ
        SLN_LastSample = SumLS
        STR_LastSample = SumTS
        if Object.ConsoleOutputFlag:
            print('{:^12.1f}{:^12.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(Object.VelocityZ,Object.VelocityY,
                                                                                    Object.MeanElectronEnergy, Object.DiffusionX, Object.DiffusionY,
                                                                                    Object.DiffusionZ,Object.DiffusionYZ,Object.LongitudinalDiffusion,Object.TransverseDiffusion))

    # Calculate errors and check averages
    SumV_Samples = 0.0
    TWYST = 0.0
    SumE_Samples = 0.0
    SumV2_Samples = 0.0
    T2WYST = 0.0
    SumE2_Samples = 0.0
    SumDZZ_Samples = 0.0
    SumDYY_Samples = 0.0
    SumDXX_Samples = 0.0
    TYZST = 0.0
    TLNST = 0.0
    TTRST = 0.0
    SumDZZ2_Samples = 0.0
    SumDYY2_Samples = 0.0
    SumDXX2_Samples = 0.0
    T2YZST = 0.0
    T2LNST = 0.0
    T2TRST = 0.0

    for K in range(10):
        SumV_Samples = SumV_Samples + WZST[K]
        TWYST = TWYST + WYST[K]
        SumE_Samples = SumE_Samples + AVEST[K]
        SumV2_Samples = SumV2_Samples + WZST[K] * WZST[K]
        T2WYST = T2WYST + WYST[K] * WYST[K]
        SumE2_Samples = SumE2_Samples + AVEST[K] * AVEST[K]
        SumDXX_Samples += DFXXST[K]
        SumDXX2_Samples += DFXXST[K] ** 2
        if K >= 2:
            SumDZZ_Samples = SumDZZ_Samples + DFZZST[K]
            SumDYY_Samples = SumDYY_Samples + DFYYST[K]
            TYZST = TYZST + DFYZST[K]
            TLNST = TLNST + DFLNST[K]
            TTRST = TTRST + DFTRST[K]
            SumDZZ2_Samples += DFZZST[K] ** 2
            SumDYY2_Samples += DFYYST[K] ** 2
            T2YZST += DFYZST[K] ** 2
            T2LNST += DFLNST[K] ** 2
            T2TRST += DFTRST[K] ** 2
    Object.VelocityErrorZ = 100 * sqrt((SumV2_Samples - SumV_Samples * SumV_Samples / 10.0) / 9.0) / Object.VelocityZ
    Object.VelocityErrorY = 100 * sqrt((T2WYST - TWYST * TWYST / 10.0) / 9.0) / abs(Object.VelocityY)
    Object.MeanElectronEnergyError = 100 * sqrt((SumE2_Samples - SumE_Samples * SumE_Samples / 10.0) / 9.0) / Object.MeanElectronEnergy
    Object.ErrorDiffusionX = 100 * sqrt((SumDXX2_Samples - SumDXX_Samples * SumDXX_Samples / 10.0) / 9.0) / Object.DiffusionX
    Object.ErrorDiffusionY = 100 * sqrt((SumDYY2_Samples - SumDYY_Samples * SumDYY_Samples / 8.0) / 7.0) / Object.DiffusionY
    Object.ErrorDiffusionZ = 100 * sqrt((SumDZZ2_Samples - SumDZZ_Samples * SumDZZ_Samples / 8.0) / 7.0) / Object.DiffusionZ
    Object.ErrorDiffusionYZ = 100 * sqrt((T2YZST - TYZST * TYZST / 8.0) / 7.0) / abs(Object.DiffusionYZ)
    Object.LongitudinalDiffusionError = 100 * sqrt((T2LNST - TLNST * TLNST / 8.0) / 7.0) / Object.LongitudinalDiffusion
    Object.TransverseDiffusionError = 100 * sqrt((T2TRST - TTRST * TTRST / 8.0) / 7.0) / Object.TransverseDiffusion
    Object.VelocityErrorZ = Object.VelocityErrorZ / sqrt(10)
    Object.VelocityErrorY = Object.VelocityErrorY / sqrt(10)
    Object.MeanElectronEnergyError = Object.MeanElectronEnergyError / sqrt(10)
    Object.ErrorDiffusionX = Object.ErrorDiffusionX / sqrt(10)
    Object.ErrorDiffusionY = Object.ErrorDiffusionY / sqrt(8)
    Object.ErrorDiffusionZ = Object.ErrorDiffusionZ / sqrt(8)
    Object.ErrorDiffusionYZ = Object.ErrorDiffusionYZ / sqrt(8)
    Object.LongitudinalDiffusionError = Object.LongitudinalDiffusionError / sqrt(8)
    Object.TransverseDiffusionError = Object.TransverseDiffusionError / sqrt(8)

    # CONVERT CM/SEC

    Object.VelocityZ *= 1e5
    Object.VelocityY *= 1e5

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


