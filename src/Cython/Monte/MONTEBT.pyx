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

#Sample Maxwell Boltzman distribution for gas velocities
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void GenerateMaxBoltz(double RandomSeed, double *RandomMaxBoltzArray):
    cdef double Ran1, Ran2, TwoPi
    cdef int J
    for J in range(0, 5, 2):
        Ran1 = random_uniform(RandomSeed)
        Ran2 = random_uniform(RandomSeed)
        TwoPi = 2.0 * np.pi
        RandomMaxBoltzArray[J] = sqrt(-1*log(Ran1)) * cos(Ran2 * TwoPi)
        RandomMaxBoltzArray[J + 1] = sqrt(-1*log(Ran1)) * sin(Ran2 * TwoPi)


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
    cdef long long I, NumDecorLengths,  NumCollisions, IEXTRA, IMBPT, K, J, iCollisionM, iSample, iCollision, GasIndex, IE, IT, CollsToLookBack, IPT, iCorr,DecorDistance
    cdef double ST1, RandomSeed,ST2, SumE2, SumXX, SumYY, SumZZ, SumVX, SumVY, Z_LastSample, ST_LastSample, ST1_LastSample, ST2_LastSample, SumZZ_LastSample, SumXX_LastSample, SumYY_LastSample, SVX_LastSample, SVY_LastSample, SME2_LastSample, TDash
    cdef double ABSFAKEI, DirCosineZ1, DirCosineX1, DirCosineY1, VelXBefore, VelYBefore, VelZBefore, BP, F1, F2, TwoPi, DirCosineX2, DirCosineY2, DirCosineZ2, VelXAfter, VelYAfter, VelZAfter, DZCOM, DYCOM, DXCOM, Theta,
    cdef double  EBefore, Sqrt2M, TwoM, AP, CONST6, RandomNum, GasVelX, GasVelY, GasVelZ, VEX, VEY, VEZ, COMEnergy, Test1, Test2, Test3, CONST11
    cdef double T2, A, B, CONST7, S1, EI, R9, EXTRA, RAN, RandomNum1, CosTheta, EPSI, Phi, SinPhi, CosPhi, ARG1, D, Q, CosZAngle, U, CosSquareTheta, SinZAngle, VXLAB, VYLAB, VZLAB
    cdef double SumV_Samples, SumE_Samples, SumV2_Samples, SumE2_Samples, SumDXX_Samples, SumDYY_Samples, SumDXX2_Samples, SumDYY2_Samples, SumDZZ_Samples, SumDZZ2_Samples, Attachment, Ionization, E,SumYZ,SumLS,SumTS
    cdef double SumYZ_LastSample,SLN_LastSample,STR_LastSample,EBAR_LastSample,EFieldTimes100, EBAR


    #Initialize variables
    NumDecorLengths = 0
    NumCollisions = 0
    IEXTRA = 0
    IMBPT = 0
    TDash = 0.0
    Object.FakeIonizations = 0
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
    SumZZ_LastSample = 0.0
    SumXX_LastSample = 0.0
    SumYY_LastSample = 0.0
    SumYZ_LastSample = 0.0
    SVX_LastSample = 0.0
    SLN_LastSample = 0.0
    STR_LastSample = 0.0
    EBAR_LastSample = 0.0



    # These arrays store X,Y,Z,T about every real collision

    cdef double *CollT, *CollX, *CollY, *CollZ

    CollT = <double *> malloc(2000000 * sizeof(double))
    memset(CollT, 0, 2000000 * sizeof(double))

    CollX = <double *> malloc(2000000 * sizeof(double))
    memset(CollX, 0, 2000000 * sizeof(double))

    CollY = <double *> malloc(2000000 * sizeof(double))
    memset(CollY, 0, 2000000 * sizeof(double))

    CollZ = <double *> malloc(2000000 * sizeof(double))
    memset(CollZ, 0, 2000000 * sizeof(double))



    # These arrays store estimates of the drift params from each sample
    cdef double *DriftVelPerSampleZ, *MeanEnergyPerSample, *DiffZZPerSample, *DiffYYPerSample, *DiffXXPerSample,*DiffYZPerSample,*DiffLonPerSample,*DFTRNST,*DriftVelPerSampleYZ, *DiffTranPerSample

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

    DFTRNST = <double *> malloc(10 * sizeof(double))
    memset(DFTRNST, 0, 10 * sizeof(double))

    DiffTranPerSample  = <double *> malloc(10 * sizeof(double))
    memset(DiffTranPerSample, 0, 10 * sizeof(double))


    # Here are some constants we will use
    EFieldTimes100 = Object.EField * 100
    EBefore = Object.InitialElectronEnergy
    Sqrt2M = Object.CONST3 * 0.01
    TwoM = Sqrt2M ** 2
    TwoPi = 2 * acos(-1)


    cdef double ** TEMP = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TEMP[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TEMP[K][J] = Object.TotalCollisionFrequency[K][J] + Object.TotalCollisionFrequencyNull[K][J]
    ABSFAKEI = Object.FakeIonizations

    GenerateMaxBoltz(Object.RandomSeed,  Object.RandomMaxBoltzArray)


    # Initial direction cosines and velocities
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    VelTotal = Sqrt2M * sqrt(EBefore)
    VelXBefore = DirCosineX1 * VelTotal
    VelYBefore = DirCosineY1 * VelTotal
    VelZBefore = DirCosineZ1 * VelTotal
    RandomSeed = Object.RandomSeed


    # Optionally write some output header to screen
    if Object.ConsoleOutputFlag:
        print('{:^12s}{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity Z", "Velocity Y", "Energy",
                                                                       "DIFXX", "DIFYY", "DIFZZ", "DIFYZ","DIFLNG","DIFTRN"))

    # We run collisions in NumSamples batches,
    # evenly distributed between its MaxNumberOfCollisions collisions.
    iCollisionM = <long long>(Object.MaxNumberOfCollisions / Object.NumSamples)
    for iSample in range(int(Object.NumSamples)):
        for iCollision in range(int(iCollisionM)):
            while True:
                RandomNum = random_uniform(RandomSeed)
                T = -1 * log(RandomNum) / Object.MaxCollisionFreqTotal + TDash
                TDash = T
                WBT = Object.AngularSpeedOfRotation * T
                CosWT = cos(WBT)
                SinWT = sin(WBT)
                DZ = (VelZBefore * SinWT + (Object.EFieldOverBField - VelYBefore) * (1 - CosWT)) / Object.AngularSpeedOfRotation

                E = EBefore + DZ * EFieldTimes100

                #Update electron velocity in lab frame
                VelXAfter = VelXBefore
                VelYAfter = (VelYBefore - Object.EFieldOverBField) * CosWT + VelZBefore * SinWT + Object.EFieldOverBField
                VelZAfter = VelZBefore * CosWT - (VelYBefore - Object.EFieldOverBField) * SinWT

                # Randomly choose gas to scatter from, based on expected collision freqs.
                GasIndex = 0
                RandomNum = random_uniform(RandomSeed)
                if Object.NumberOfGases == 1:
                    GasIndex = 0
                else:
                    while (Object.MaxCollisionFreqTotalG[GasIndex] < RandomNum):
                        GasIndex = GasIndex + 1

                # Pick random gas molecule velocity for collision
                IMBPT += 1
                if (IMBPT > 6):
                    GenerateMaxBoltz(Object.RandomSeed,  Object.RandomMaxBoltzArray)
                    IMBPT = 1
                GasVelX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
                IMBPT += 1
                GasVelY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
                IMBPT += 1
                GasVelZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]

                # Calculate energy with a stationary gas target
                COMEnergy = ((VelXAfter - GasVelX) ** 2 + (VelYAfter - GasVelY) ** 2 + (VelZAfter - GasVelZ) ** 2) / TwoM
                IE = int(COMEnergy / Object.ElectronEnergyStep)
                IE = min(IE, 3999)

                # Test for real or null collision
                RandomNum = random_uniform(RandomSeed)
                Test1 = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
                # Test FOR REAL OR NULL COLLISION
                if RandomNum > Test1:
                    Test2 = TEMP[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
                    if RandomNum < Test2:
                        if Object.NumMomCrossSectionPointsNull[GasIndex] == 0:
                            continue
                        RandomNum = random_uniform(RandomSeed)
                        I = 0
                        while Object.NullCollisionFreq[GasIndex][IE][I] < RandomNum:
                            # Increment null scatter sum
                            I += 1

                        Object.ICOLNN[GasIndex][I] += 1
                        continue
                    else:
                        Test3 = (TEMP[GasIndex][IE] + ABSFAKEI) / Object.MaxCollisionFreq[GasIndex]
                        if RandomNum < Test3:
                            #  Increment fake ionizaiton counter
                            Object.FakeIonizations += 1
                            continue
                        continue
                else:
                    break
            Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T

            NumCollisions += 1

            #Calculate direction cosines in COM frame
            CONST11 = 1 / (Sqrt2M * sqrt(COMEnergy))
            DXCOM = (VelXAfter - GasVelX) * CONST11
            DYCOM = (VelYAfter - GasVelY) * CONST11
            DZCOM = (VelZAfter - GasVelZ) * CONST11
            #  CALCULATE POSITIONS AT INSTANT BEFORE COLLISION
            #    ALSO UPDATE DIFFUSION  AND ENERGY CALCULATIONS.
            T2 = T ** 2
            TDash = 0.0

            # Update electron position
            Object.X += VelXBefore * T
            Object.Y += Object.EFieldOverBField * T + ((VelYBefore - Object.EFieldOverBField) * SinWT + VelZBefore * (1 - CosWT)) / Object.AngularSpeedOfRotation
            Object.Z += DZ
            Object.TimeSum += T
            IT = int(T)
            IT = min(IT, 299)
            Object.CollisionTimes[IT] += 1

            #Update collision counter and global velocity
            Object.CollisionEnergies[IE] += 1
            Object.VelocityZ = Object.Z / Object.TimeSum
            Object.VelocityY = Object.Y / Object.TimeSum
            SumVX += (VelXBefore ** 2) * T2


            # Decor_Colls specifies how far we must have drifted within this Sample
            #   before we start measuring the ensemble.  We only do anything if
            #   Decor_Colls collisions deep (equivalently, NumDecorLengths>1).
            #
            # Once we are far enough, we do the following. We'll be looking backward
            #   by Decor_Colls-N*Decor_Step.  N is ideally Decor_LookBacks.
            #
            # In the case where that would put us too far back (Decor_Colls or less)
            #   then we settle for a smaller N. This tells us the decorrelation distance.
            #
            # Reminder: NumCollisions is the number of collisions we are into *this*
            #   decorrelation length - it resets to zero each time we enter a
            #   new one.
            #
            # This formalism is based on Eqs 8, Frasier and Mathieson.
            #
            # The entries are weighted by time between this collision and the last T.
            # The TDiff on the denominator

            if NumDecorLengths != 0:
                CollsToLookBack = 0
                for J in range(int(Object.Decor_LookBacks)):
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

            # Record collision positions
            CollX[NumCollisions-1] = Object.X
            CollY[NumCollisions-1] = Object.Y
            CollZ[NumCollisions-1] = Object.Z
            CollT[NumCollisions-1] = Object.TimeSum
            if NumCollisions >= Object.Decor_Colls:
                NumDecorLengths += 1
                NumCollisions = 0

            # Randomly pick the type of collision we will have
            RandomNum = random_uniform(RandomSeed)

            # Find location within 4 units in collision array
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

            # Generate scattering angles and update laboratory cosines after collision also update energy of electron
            IPT = <long long>Object.InteractionType[GasIndex][I]
            Object.CollisionsPerGasPerType[GasIndex][int(IPT)] += 1
            Object.ICOLN[GasIndex][I] += 1
            if COMEnergy < EI:
                #FIX ENERGY LOSS SMALLER THAN INCNumDecorLengthsENT ENERGY IF ERROR OCCURS
                EI = COMEnergy - 0.0001

           # If Penning is enabled and the energy transfer let to excitation,
            #  add probabilities to transfer energy to surrounding molecules
            if Object.EnablePenning != 0:
                if Object.PenningFraction[GasIndex][0][I] != 0:
                    RAN = random_uniform(RandomSeed)
                    if RAN <= Object.PenningFraction[GasIndex][0][I]:
                        #ADD EXTRA IONISATION COLLISION
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

            # Anisotropic scattering - pick the scattering angle theta depending on scatter type
            RandomNum = random_uniform(RandomSeed)
            if Object.AngularModel[GasIndex][I] == 1:
	            # Use method of Capitelli et al
                RandomNum1 = random_uniform(RandomSeed)
                CosTheta = 1.0 - RandomNum * Object.AngleCut[GasIndex][IE][I]
                if RandomNum1 > Object.ScatteringParameter[GasIndex][IE][I]:
                    CosTheta = -1 * CosTheta
            elif Object.AngularModel[GasIndex][I] == 2:
                # Use method of Okhrimovskyy et al
                EPSI = Object.ScatteringParameter[GasIndex][IE][I]
                CosTheta = 1 - (2 * RandomNum * (1 - EPSI) / (1 + EPSI * (1 - 2 * RandomNum)))
            else:
                # Isotropic scattering
                CosTheta = 1 - 2 * RandomNum
            Theta = acos(CosTheta)

            # Pick a random Phi - must be uniform by symmetry of the gas
            RandomNum = random_uniform(RandomSeed)
            Phi = TwoPi * RandomNum
            SinPhi = sin(Phi)
            CosPhi = cos(Phi)
            ARG1 = 1 - S1 * EI / COMEnergy
            ARG1 = max(ARG1, Object.SmallNumber)
            D = 1 - CosTheta * sqrt(ARG1)

            # Update the energy to start drifing for the next round.
            #  If its zero, make it small but nonzero.
            EBefore = COMEnergy * (1 - EI / (S1 * COMEnergy) - 2 * D / S2)
            EBefore = max(EBefore, Object.SmallNumber)


            Q = sqrt((COMEnergy / EBefore) * ARG1) / S1
            Q = min(Q, 1)
            Object.AngleFromZ = asin(Q * sin(Theta))
            CosZAngle = cos(Object.AngleFromZ)
            U = (S1 - 1) * (S1 - 1) / ARG1

            # Find new directons after scatter
            CosSquareTheta = CosTheta * CosTheta
            if CosTheta < 0 and CosSquareTheta > U:
                CosZAngle = -1 * CosZAngle
            SinZAngle = sin(Object.AngleFromZ)
            DZCOM = min(DZCOM, 1)
            ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
            if ARGZ == 0:
                # If scattering frame is same as lab frame, do this;
                DirCosineZ1 = CosZAngle
                DirCosineX1 = CosPhi * SinZAngle
                DirCosineY1 = SinPhi * SinZAngle
            else:
                # otherwise do this.
                DirCosineZ1 = DZCOM * CosZAngle + ARGZ * SinZAngle * SinPhi
                DirCosineY1 = DYCOM * CosZAngle + (SinZAngle / ARGZ) * (DXCOM * CosPhi - DYCOM * DZCOM * SinPhi)
                DirCosineX1 = DXCOM * CosZAngle - (SinZAngle / ARGZ) * (DYCOM * CosPhi + DXCOM * DZCOM * SinPhi)

            # Transform velocity vectors to lab frame
            VelTotal = Sqrt2M * sqrt(EBefore)
            VelXBefore = DirCosineX1 * VelTotal + GasVelX
            VelYBefore = DirCosineY1 * VelTotal + GasVelY
            VelZBefore = DirCosineZ1 * VelTotal + GasVelZ

            # Calculate energy and direction cosines in lab frame
            EBefore = (VelXBefore * VelXBefore + VelYBefore * VelYBefore + VelZBefore * VelZBefore) / TwoM
            CONST11 = 1 / (Sqrt2M * sqrt(EBefore))
            DirCosineX1 = VelXBefore * CONST11
            DirCosineY1 = VelYBefore * CONST11
            DirCosineZ1 = VelZBefore * CONST11

        #Put stuff into the right units and normalize
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
        DriftVelPerSampleZ[iSample] = (Object.Z - Z_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        DriftVelPerSampleY[iSample] = (Object.Y - Y_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        MeanEnergyPerSample[iSample] = (EBAR - EBAR_LastSample) / (Object.TimeSum - ST_LastSample)
        EBAR_LastSample = EBAR
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
        if Object.ConsoleOutputFlag:
            print('{:^12.1f}{:^12.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(Object.VelocityZ,Object.VelocityY,
                                                                                    Object.MeanElectronEnergy, Object.DiffusionX, Object.DiffusionY,
                                                                                    Object.DiffusionZ,Object.DiffusionYZ,Object.LongitudinalDiffusion,Object.TransverseDiffusion))

    # Calculate errors and check averages
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
    Object.VelocityErrorZ = 100 * sqrt((SumV2_Samples - SumV_Samples * SumV_Samples / 10.0) / 9.0) / Object.VelocityZ
    Object.VelocityErrorY = 100 * sqrt((T2DriftVelPerSampleY - TDriftVelPerSampleY * TDriftVelPerSampleY / 10.0) / 9.0) / abs(Object.VelocityY)
    Object.MeanElectronEnergyError = 100 * sqrt((SumE2_Samples - SumE_Samples * SumE_Samples / 10.0) / 9.0) / Object.MeanElectronEnergy
    Object.ErrorDiffusionX = 100 * sqrt((SumDXX2_Samples - SumDXX_Samples * SumDXX_Samples / 10.0) / 9.0) / Object.DiffusionX
    Object.ErrorDiffusionY = 100 * sqrt((SumDYY2_Samples - SumDYY_Samples * SumDYY_Samples / 8.0) / 7.0) / Object.DiffusionY
    Object.ErrorDiffusionZ = 100 * sqrt((SumDZZ2_Samples - SumDZZ_Samples * SumDZZ_Samples / 8.0) / 7.0) / Object.DiffusionZ
    Object.ErrorDiffusionYZ = 100 * sqrt((T2YCollZ - TYCollZ * TYCollZ / 8.0) / 7.0) / abs(Object.DiffusionYZ)
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

    # Convert to cm/sec
    Object.VelocityZ *= 1e5
    Object.VelocityY *= 1e5

    # Calculate Townsend coeffs and errors. Error here is purely poissonian.
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


