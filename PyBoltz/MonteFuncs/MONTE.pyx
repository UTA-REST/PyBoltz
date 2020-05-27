from PyBoltz.Boltz cimport Boltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow
from libc.string cimport memset
from PyBoltz.Boltz cimport drand48
from PyBoltz.MBSorts cimport MBSort
from libc.stdlib cimport malloc, free
import cython
import numpy as np
cimport numpy as np
import sys

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = drand48(dummy)
    return r


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(Boltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at temp =  TemperatureCentigrade.

    This function is used when there is no magnetic Field.     
    
    Electric Field in z direction.

    The object parameter is the Boltz object to have the output results and to be used in the simulation.
    """
    Object.VelocityX = 0.0
    Object.VelocityY = 0.0
    Object.VelocityErrorX = 0.0
    Object.VelocityErrorY = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.TimeSum = 0.0
    cdef long long I, NumDecorLengths, NumCollisions, IEXTRA, IMBPT, K, J, CollisionsPerSample, iSample, iCollision, GasIndex, iEnergyBinnergyBin, iTimeBin, CollsBackToLook, IPT, iCorr, DecorDistance
    cdef double ST1, RandomSeed, ST2, SumE2, SumXX, SumYY, SumZZ, SumVX, SumVY, Z_LastSample, ST_LastSample, ST1_LastSample, ST2_LastSample, SumZZ_LastSample, SumXX_LastSample, SumYY_LastSample, SumVX_LastSample, SumVY_LastSample, SME2_LastSample, TDash,TDiff
    cdef double ABSFAKEI, DirCosineZ1, DirCosineX1, DirCosineY1, VelXBefore, VelYBefore, BP, F1, F2, TwoPi, DirCosineX2, DirCosineY2, DirCosineZ2, Theta,
    cdef double  EBefore, Sqrt2M, TwoM, AP,  VEX, VEY, VEZ, Test1, Test2, Test3,VelocityRatio,CosWT,SinWT,WBT,VelAfter, VelBefore
    cdef double T2, A, B, VelocityBefore,  S1, EI,  EXTRA, RandomNum, RandomNum2, CosTheta, EPSI,  Phi, SinPhi, CosPhi, ARG1, D, Q, CosZAngle, U,  SinZAngle
    cdef double SumV2_Samples, SumV_Samples, SumE2_Samples, SumE_Samples, SumDXX_Samples, SumDYY_Samples, SumDZZ_Samples, SumDXX2_Samples, SumDYY2_Samples, SumDZZ2_Samples, Attachment, Ionization, EAfter
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

    cdef double *DriftVelPerSample, *MeanEnergyPerSample, *DiffZZPerSample, *DiffYYPerSample, *DiffXXPerSample

    DriftVelPerSample = <double *> malloc(Object.Num_Samples * sizeof(double))
    memset(DriftVelPerSample, 0, Object.Num_Samples * sizeof(double))

    MeanEnergyPerSample = <double *> malloc(Object.Num_Samples * sizeof(double))
    memset(MeanEnergyPerSample, 0, Object.Num_Samples * sizeof(double))

    DiffZZPerSample = <double *> malloc(Object.Num_Samples * sizeof(double))
    memset(DiffZZPerSample, 0, Object.Num_Samples * sizeof(double))

    DiffYYPerSample = <double *> malloc(Object.Num_Samples * sizeof(double))
    memset(DiffYYPerSample, 0, Object.Num_Samples * sizeof(double))

    DiffXXPerSample = <double *> malloc(Object.Num_Samples * sizeof(double))
    memset(DiffXXPerSample, 0, Object.Num_Samples * sizeof(double))

    RandomSeed = Object.Random_Seed
    EBefore = Object.InitialElectronEnergy
    NumDecorLengths = 0
    NumCollisions = 0
    IEXTRA = 0
    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))
    for J in range(4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]
    ABSFAKEI = abs(Object.FakeIonizations)
    Object.FakeIonizations = 0


    # Here are some constants we will use
    BP = pow(Object.EField, 2) * Object.CONST1  # This should be: 1/2 m e^2 EField^2
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3          # This should be: sqrt( m / 2) e EField
    Sqrt2M = Object.CONST3 * 0.01               # This should be: sqrt(2m)
    TwoM   =  pow(Sqrt2M, 2)                    # This should be: 2m
    TwoPi = 2.0 * np.pi                         # This should be: 2 Pi


    # Initial direction cosines
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    VelBefore = Sqrt2M * sqrt(EBefore)
    VelXBefore = DirCosineX1 * VelBefore
    VelYBefore = DirCosineY1 * VelBefore
    VelZBefore = DirCosineZ1 * VelBefore


    # Optionally write some output header to screen
    if Object.Console_Output_Flag:
        print('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity", "Position", "Time", "Energy",
                                                                         "DIFXX", "DIFYY", "DIFZZ"))

    # We run collisions in NumSamples batches,
    # evenly distributed between its MaxNumberOfCollisions collisions.
    CollisionsPerSample = <long long> (Object.MaxNumberOfCollisions / Object.Num_Samples)

    INTEM = 8
    DeltaE = Object.Max_Electron_Energy / float(INTEM)

    for iSample in range(int(Object.Num_Samples)):
        for iCollision in range(int(CollisionsPerSample)):
            while True:
                RandomNum = random_uniform(RandomSeed)
                I = int(EBefore / DeltaE) + 1
                I = min(I, INTEM) - 1
                TLIM = Object.MaxCollisionFreqNT[I]

                # Random sampling formula from Skullerud
                T = -1 * log(RandomNum) / TLIM + TDash
                Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T
                TDash = T
                AP = DirCosineZ1 * F2 * sqrt(EBefore)
                EAfter = EBefore + (AP + BP * T) * T
                iEnergyBin = int(EAfter / Object.ElectronEnergyStep)

                iEnergyBin = min(iEnergyBin, 3999)
                if TEMP[iEnergyBin] > TLIM:
                    TDash += log(RandomNum) / TLIM
                    Object.MaxCollisionFreqNT[I] *= 1.05
                    continue

                # Now the Skullerud null collision method
                RandomNum = random_uniform(RandomSeed)

                # If we draw below this number, we will null-scatter (no mom xfer)
                Test1 = Object.TotalCollisionFrequencyNT[iEnergyBin] / TLIM

                if RandomNum > Test1:
                    Test2 = TEMP[iEnergyBin] / TLIM
                    if RandomNum < Test2:
                        # Test for null levels
                        if Object.NumMomCrossSectionPointsNullNT == 0:
                            continue
                        RandomNum = random_uniform(RandomSeed)
                        I = 0
                        while Object.NullCollisionFreqNT[iEnergyBin][I] < RandomNum:
                            I += 1
                        # Increment null scatter sum
                        Object.ICOLNNNT[I] += 1
                        continue
                    else:
                        Test3 = (TEMP[iEnergyBin] + ABSFAKEI) / TLIM
                        if RandomNum < Test3:
                            # FAKE IONISATION INCREMENT COUNTER
                            Object.FakeIonizations += 1
                            continue
                        continue
                else:
                    break

             # If we got this far, we have a collision.
            NumCollisions += 1
            # Keep a running average of mean time between real collisions
            Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * TDash

            # Reset time-to-next-real-collision clock
            TDash = 0.0


            # From above, A = m VBefore a T_total
            #             B = 1/2 m a^2 T_total^2
            # which is accurate, because only null collisions happened,
            # so we had simple uniform acceleration.
            #
            T2 = T * T
            A = AP * T
            B = BP * T2

            # Add Integral(E^2,dt) to running total.
            SumE2 = SumE2 + T * (EBefore + A / 2.0 + B / 3.0)

            # These are the X and Y velocities before we started accelerating
            VelocityBefore =  Sqrt2M * sqrt(EBefore)
            VelXBefore = DirCosineX1 * VelocityBefore
            VelYBefore = DirCosineY1 * VelocityBefore
            VelZBefore = DirCosineZ1 * VelocityBefore


            if(Object.BFieldMode==1):
                VelocityRatio = sqrt(EBefore / EAfter)
                DirCosineX2 = DirCosineX1 * VelocityRatio
                DirCosineY2 = DirCosineY1 * VelocityRatio
                DirCosineZ2 = DirCosineZ1 * VelocityRatio + T * F2 / (2.0*sqrt(EAfter))

                # Update position following acceleration
                A = T * VelocityBefore
                Object.X = Object.X + DirCosineX1 * A
                Object.Y = Object.Y + DirCosineY1 * A
                Object.Z = Object.Z + DirCosineZ1 * A + T2 * F1

            elif(Object.BFieldMode==2):
                 VelocityRatio = sqrt(EBefore / EAfter)
                 WBT = Object.AngularSpeedOfRotation * T
                 CosWT = cos(WBT)
                 SinWT = sin(WBT)
                 VelAfter = Sqrt2M * sqrt(EAfter)
                 A = T * VelocityBefore
                 DirCosineX2 = (VelXBefore * CosWT - VelYBefore * SinWT) / VelAfter
                 DirCosineY2 = (VelYBefore * CosWT + VelXBefore * SinWT) / VelAfter
                 DirCosineZ2 = DirCosineZ1 * VelocityRatio + T * F2 / (2.0*sqrt(EAfter))
                 Object.X +=  (VelXBefore * SinWT - VelYBefore * (1 - CosWT)) / Object.AngularSpeedOfRotation
                 Object.Y += (VelYBefore * SinWT + VelXBefore * (1 - CosWT)) / Object.AngularSpeedOfRotation
                 Object.Z += DirCosineZ1 * A + T2 * F1
            Object.TimeSum = Object.TimeSum + T



            # Figure out which time bin we're in, 299 is overflow; record collision
            #  at that time
            iTimeBin = int(T)
            iTimeBin = min(iTimeBin, 299)
            Object.CollisionTimes[iTimeBin] += 1
            Object.CollisionEnergies[iEnergyBin] += 1
            Object.VelocityZ = Object.Z / Object.TimeSum
            # Keep running total of Sum(V^2 T^2).
            SumVX = SumVX + VelXBefore * VelXBefore * T2
            SumVY = SumVY + VelYBefore * VelYBefore * T2

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
                for iCorr in range(int(Object.Decor_Lookbacks)):
                    ST2 += T
                    DecorDistance = NumCollisions + CollsToLookBack
                    if DecorDistance > Object.Decor_Colls:
                        DecorDistance = DecorDistance - Object.Decor_Colls
                    TDiff = Object.TimeSum - CollT[DecorDistance - 1]

                    # These are used to calculate diffusion constants, per Frasier and Mathieson Eq 8.
                    SumXX = SumXX + ((Object.X - CollX[DecorDistance - 1]) ** 2) * T / TDiff
                    SumYY = SumYY + ((Object.Y - CollY[DecorDistance - 1]) ** 2) * T / TDiff

                    # Becayse Z term includes drift velocity, must have calc'd it somewhat to usefully accumulate
                    #  in the ensemble, so only start accumulating after 2nd sample
                    if iSample >= 2:
                        ST1 += T
                        SumZZ = SumZZ + ((Object.Z - CollZ[DecorDistance - 1] - Object.VelocityZ * TDiff) ** 2) * T / TDiff
                    CollsToLookBack += Object.Decor_Step

            # Record collision positions
            CollX[NumCollisions - 1] = Object.X
            CollY[NumCollisions - 1] = Object.Y
            CollZ[NumCollisions - 1] = Object.Z

            CollT[NumCollisions - 1] = Object.TimeSum
            if NumCollisions >= Object.Decor_Colls:
                NumDecorLengths += 1
                NumCollisions = 0

            # Randomly pick the type of collision we will have
            RandomNum = random_uniform(RandomSeed)


            # Find location within 4 units in collision array
            I = MBSort(I, RandomNum, iEnergyBin, Object)
            while Object.CollisionFrequencyNT[iEnergyBin][I] < RandomNum:
                I = I + 1
            S1 = Object.RGasNT[I]
            EI = Object.EnergyLevelsNT[I]
            if Object.ElectronNumChangeNT[I] > 0:
                # Use flat distributioon of electron energy between E-IonizationEnergy and 0.0 EV, same as in Boltzmann
                RandomNum = random_uniform(RandomSeed)
                EXTRA = RandomNum * (EAfter - EI)
                EI = EXTRA + EI
                # Add extra ionisation collision
                IEXTRA += <long long>Object.NC0NT[I]

            # Generate scattering angles and update laboratory cosines after collision also update energy of electron
            IPT = <long long>Object.InteractionTypeNT[I]

            Object.CollisionsPerGasPerTypeNT[int(IPT) - 1] += 1
            Object.ICOLNNT[I] += 1
            if EAfter < EI:
                EI = EAfter - 0.0001

            # If Penning is enabled and the energy transfer let to excitation,
            #  add probabilities to transfer energy to surrounding molecules
            if Object.Enable_Penning != 0:
                if Object.PenningFractionNT[0][I] != 0:
                    RAN = random_uniform(RandomSeed)
                    if RAN <= Object.PenningFractionNT[0][I]:
                        # add extra ionisation collision
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

            # Anisotropic scattering - pick the scattering angle theta depending on scatter type
            RandomNum = random_uniform(RandomSeed)
            if Object.AngularModelNT[I] == 1:
                RandomNum1 = random_uniform(RandomSeed)
                CosTheta = 1.0 - RandomNum * Object.AngleCutNT[iEnergyBin][I]
                if RandomNum1 > Object.ScatteringParameterNT[iEnergyBin][I]:
                    CosTheta = -1 * CosTheta
            elif Object.AngularModelNT[I] == 2:
                EPSI = Object.ScatteringParameterNT[iEnergyBin][I]
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

            #TODO: Understand what Arg1, D, U and Q are
            ARG1 = max(1 - S1 * EI / EAfter, Object.SmallNumber)
            D = 1.0 - CosTheta * sqrt(ARG1)
            U = (S1 - 1) * (S1 - 1) / ARG1

            # Update the energy to start drifting for the next round.
            #  If its zero, make it small but nonzero.
            EBefore = max(EAfter * (1 - EI / (S1 * EAfter) - 2 * D / S2), Object.SmallNumber)
            Q = min(sqrt((EAfter / EBefore) * ARG1) / S1,1.0)

            # Calculate angle of scattering from Z direction
            Object.AngleFromZ = asin(Q * sin(Theta))
            CosZAngle = cos(Object.AngleFromZ)

            # Find new directons after scatter
            if CosTheta < 0 and CosTheta * CosTheta > U:
                CosZAngle = -1 * CosZAngle
            SinZAngle = sin(Object.AngleFromZ)
            DirCosineZ2 = min(DirCosineZ2, 1)

            ARGZ = sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
            if ARGZ == 0:
                # If scattering frame is same as lab frame, do this;
                DirCosineZ1 = CosZAngle
                DirCosineX1 = CosPhi * SinZAngle
                DirCosineY1 = SinPhi * SinZAngle
            else:
                # otherwise do this.
                DirCosineZ1 = DirCosineZ2 * CosZAngle + ARGZ * SinZAngle * SinPhi
                DirCosineY1 = DirCosineY2 * CosZAngle + (SinZAngle / ARGZ) * (DirCosineX2 * CosPhi - DirCosineY2 * DirCosineZ2 * SinPhi)
                DirCosineX1 = DirCosineX2 * CosZAngle - (SinZAngle / ARGZ) * (DirCosineY2 * CosPhi + DirCosineX2 * DirCosineZ2 * SinPhi)

            #And go around again to the next collision!
        #Put stuff into the right units and normalize
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
                print(SumXX, SumYY,  SumXX_LastSample, SumYY_LastSample, ST2, ST2_LastSample)
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
            print('{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(Object.VelocityZ, Object.Z, Object.TimeSum,
                                                                                         Object.MeanElectronEnergy, Object.DiffusionX,
                                                                                         Object.DiffusionY,
                                                                                         Object.DiffusionZ))
        if Object.CollisionEnergies[3999] > (1000 * float(iSample + 1)):
            raise ValueError("WARNING ENERGY OUT OF RANGE, INCREASE ELECTRON ENERGY INTEGRATION RANGE")
        #print("tmp2")

    # Calculate errors and check averages.  Means and errors are calculated statistically
    #  from the NumSamples data points.

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
    for K in range(Object.Num_Samples):
        SumV_Samples = SumV_Samples + DriftVelPerSample[K]
        SumE_Samples = SumE_Samples + MeanEnergyPerSample[K]
        SumV2_Samples = SumV2_Samples + DriftVelPerSample[K] * DriftVelPerSample[K]
        SumE2_Samples = SumE2_Samples + MeanEnergyPerSample[K] * MeanEnergyPerSample[K]
        SumDXX_Samples = SumDXX_Samples + DiffXXPerSample[K]
        SumDYY_Samples = SumDYY_Samples + DiffYYPerSample[K]
        SumDYY2_Samples = SumDYY2_Samples + pow(DiffYYPerSample[K], 2)
        SumDXX2_Samples = SumDXX2_Samples + pow(DiffXXPerSample[K], 2)
        if K >= 2:
            SumDZZ_Samples = SumDZZ_Samples + DiffZZPerSample[K]
            SumDZZ2_Samples += pow(DiffZZPerSample[K], 2)

    # (remember, we only started counting ZZ after 2 samples, as needed to estimate drift velocity first)
    Object.VelocityErrorZ = 100 * sqrt((SumV2_Samples - SumV_Samples ** 2 / Object.Num_Samples) / (Object.Num_Samples - 1)) / Object.VelocityZ
    Object.MeanElectronEnergyError = 100 * sqrt((SumE2_Samples - SumE_Samples ** 2 / Object.Num_Samples) / (Object.Num_Samples - 1)) / Object.MeanElectronEnergy
    Object.ErrorDiffusionX = 100 * sqrt((SumDXX2_Samples - SumDXX_Samples ** 2 / Object.Num_Samples) / (Object.Num_Samples - 1)) / Object.DiffusionX
    Object.ErrorDiffusionY = 100 * sqrt((SumDYY2_Samples - SumDYY_Samples ** 2 / Object.Num_Samples) / (Object.Num_Samples - 1)) / Object.DiffusionY
    Object.ErrorDiffusionZ = 100 * sqrt((SumDZZ2_Samples - SumDZZ_Samples ** 2 / Object.Num_Samples - 2) / (Object.Num_Samples - 2 - 1)) / Object.DiffusionZ
    Object.VelocityErrorZ = Object.VelocityErrorZ / sqrt(Object.Num_Samples)
    Object.MeanElectronEnergyError = Object.MeanElectronEnergyError / sqrt(Object.Num_Samples)
    Object.ErrorDiffusionX = Object.ErrorDiffusionX / sqrt(Object.Num_Samples)
    Object.ErrorDiffusionY = Object.ErrorDiffusionY / sqrt(Object.Num_Samples)
    Object.ErrorDiffusionZ = Object.ErrorDiffusionZ / sqrt(Object.Num_Samples - 2)
    Object.LongitudinalDiffusion = Object.DiffusionZ
    Object.TransverseDiffusion = (Object.DiffusionX + Object.DiffusionY) / 2

    # Convert to cm/sec
    Object.VelocityZ *= 1.0e5
    Object.LongitudinalDiffusionError = Object.ErrorDiffusionZ
    Object.TransverseDiffusionError = (Object.ErrorDiffusionX + Object.ErrorDiffusionY) / 2.0

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

    # Free up the memory and close down gracefully
    free(CollT)
    free(CollX)
    free(CollY)
    free(CollZ)
    free(DriftVelPerSample)
    free(MeanEnergyPerSample)
    free(DiffXXPerSample)
    free(DiffYYPerSample)
    free(DiffZZPerSample)
    free(TEMP)


    return

