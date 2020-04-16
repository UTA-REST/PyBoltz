from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
from libc.string cimport memset
from PyBoltz cimport drand48
from MBSorts cimport MBSortT
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
import cython
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

#Sample Maxwell Boltzman distribution for gas velocities
cdef void GenerateMaxBoltz(double RandomSeed, double *RandomMaxBoltzArray):
    cdef double Ran1, Ran2, TwoPi
    cdef int J
    for J in range(0, 5, 2):
        Ran1 = random_uniform(RandomSeed)
        Ran2 = random_uniform(RandomSeed)
        TwoPi = 2.0 * np.pi
        RandomMaxBoltzArray[J] = sqrt(-1 * log(Ran1)) * cos(Ran2 * TwoPi)
        RandomMaxBoltzArray[J + 1] = sqrt(-1 * log(Ran1)) * sin(Ran2 * TwoPi)

# Austins version of a viaiable MB distrobution
cdef RandomDirection(double length, double RandomSeed):
    cdef double TwoPi, phi, ctheta, stheta
    cdef double dx, dy, dz

    TwoPi = 2.0 * np.pi
    phi = TwoPi * random_uniform(RandomSeed)
    ctheta  = 2 * random_uniform(RandomSeed) - 1.
    stheta = sqrt(1. - ctheta * ctheta)
    dx = length * cos(phi) * stheta
    dy = length * sin(phi) * stheta
    dz = length * ctheta
    return dx,dy,dz


cdef GausianBoxMuller(double mu, double sigma, double RandomSeed):
    cdef double r_1, r_2, x, y
    
    r_1 = random_uniform(RandomSeed)
    r_2 = random_uniform(RandomSeed)
    
    x = sigma * sqrt(-2 * log(r_1)) \
      * cos(2 * np.pi * r_2) + mu
    y = sigma * sqrt(-2 * log(r_1)) \
      * sin(2 * np.pi * r_2) + mu
    return [x, y]

cdef AustinGasVel(double mu, double sigma, double RandomSeed):
    cdef double GasVel, GasVel1
    cdef GasX, GasY, GasZ 

    while True:
        GasVel, GasVel1 = GausianBoxMuller(mu, sigma, RandomSeed)
        if GasVel>0.01:
            break
            
    GasX,GasY,GasZ = RandomDirection(GasVel * 1e-10, RandomSeed)
    return GasX,GasY,GasZ

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at TotalCollFreqIncludingNull =  TotalCollFreqIncludingNulleratureCentigrade.

    This function is used when there is no magnetic field.     
    
    Electric field in z direction.

    The object parameter is the PyBoltz object to have the output results and to be used in the simulation.
    """

    Object.VelocityX = 0.0
    Object.VelocityY = 0.0
    Object.VelocityErrorX = 0.0
    Object.VelocityErrorY = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.TimeSum = 0.0
    cdef long long I, NumDecorLengths, NumCollisions, IEXTRA, MaxBoltzNumsUsed, K, J, CollisionsPerSample, iSample, iCollision, GasIndex, iEnergyBin, iTimeBin, CollsBackToLook, IPT, iCorr, DecorDistance
    cdef double ST1, RandomSeed, ST2, SumE2, SumXX, SumYY, SumZZ, SumVX, SumVY, Z_LastSample, ST_LastSample, ST1_LastSample, ST2_LastSample, SumZZ_LastSample, SumXX_LastSample, SumYY_LastSample, SVX_LastSample, SVY_LastSample, SME2_LastSample, TDash,TDiff
    cdef double AbsFakeIoniz, DirCosineZ1, DirCosineX1, DirCosineY1,  BP, F1, F2, TwoPi, DZCOM, DYCOM, DXCOM, Theta, VelXBefore, VelYBefore, BelZBefore, SinWT, CosWT, WBT
    cdef double  EBefore, Sqrt2M, TwoM, AP,  GasVelX, GasVelY, GasVelZ, VelXAfter, VelYAfter, VelZAfter, VelAfter, COMEnergy, Test1, Test2, Test3, VelocityInCOM, VelBeforeM1
    cdef double T2, A, B, VelocityBefore,  S1, EI,  EXTRA, RandomNum, RandomNum2, CosTheta, EpsilonOkhr,  Phi, SinPhi, CosPhi, ARG1, D, Q, CosZAngle, U,  SinZAngle, VXLab, VYLab, VZLab
    cdef double SumV2_Samples, SumV_Samples, SumE2_Samples, SumE_Samples, SumDXX_Samples, SumDYY_Samples, SumDZZ_Samples, SumDXX2_Samples, SumDYY2_Samples, SumDZZ2_Samples, Attachment, Ionization, EAfter
    cdef int Swarm_Index = 0
    cdef double Total_Coll = 0.0
    I = 0
    ST1 = 0.0
    ST2 = 0.0
    SumE2 = 0.0
    SumXX = 0.0
    SumYY = 0.0
    SumZZ = 0.0
    SumVX = 0.0
    SumVY = 0.0
    TDiff=0.0
    Z_LastSample = 0.0
    ST_LastSample = 0.0
    ST1_LastSample = 0.0
    ST2_LastSample = 0.0
    SumZZ_LastSample = 0.0
    SumXX_LastSample = 0.0
    SumYY_LastSample = 0.0
    SVX_LastSample = 0.0
    SVY_LastSample = 0.0
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

    # Generate initial random maxwell boltzman numbers
    GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)


    MaxBoltzNumsUsed = 0
    TDash = 0.0
    cdef int i = 0
    cdef double ** TotalCollFreqIncludingNull = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TotalCollFreqIncludingNull[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TotalCollFreqIncludingNull[K][J] = Object.TotalCollisionFrequency[K][J] + Object.TotalCollisionFrequencyNull[K][J]


    AbsFakeIoniz = 0.0
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


    #Optionally write some output header to screen
    if Object.Console_Output_Flag:
        print('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity", "Position", "Time", "Energy",
                                                                         "DIFXX", "DIFYY", "DIFZZ"))


    # We run collisions in NumSamples batches,
    # evenly distributed between its MaxNumberOfCollisions collisions.
    CollisionsPerSample = <long long> (Object.MaxNumberOfCollisions / Object.Num_Samples)
    SwarmCollision = <long long> CollisionsPerSample*6/100
    for iSample in range(int(Object.Num_Samples)):
        for iCollision in range(int(CollisionsPerSample)):
            while True:

                # Sample random time to next collision. T is global total time.
                RandomNum = random_uniform(RandomSeed)

                # This is the formula from Skullerud
                T = -log(RandomNum) / Object.MaxCollisionFreqTotal + TDash
                TDash = T

                # Apply acceleration.
                #
                #     VBefore = VBefore + a t
                #      EAfter = 1/2 m VAfter^2
                #             = 1/2 m(VAfterX^2 + VAfterY^2 + VAfterZ^2)
                #             = 1/2 m((VBeforeZ + at)^2 + VBeforeX^2 + VBeforeY^2)
                #             = EBefore + (dir_z)(AP + BP * T) * T
                #
                #  w/      AP = m VBefore a
                #          BP = 1/2 m a^2
                #  
                # So here, F2 = sqrt(m / 2) EField e
                #          BP = 1/2 m EField^2 e^2 
                #

                AP = DirCosineZ1 * F2 * sqrt(EBefore)
                EAfter = EBefore + (AP + BP * T) * T
                VelocityRatio = sqrt(EBefore / EAfter)



                # Randomly choose gas to scatter from, based on expected collision freqs.
                GasIndex = 0
                if Object.NumberOfGases == 1:
                    RandomNum = random_uniform(RandomSeed)
                    GasIndex = 0
                else:
                    RandomNum = random_uniform(RandomSeed)
                    while (Object.MaxCollisionFreqTotalG[GasIndex] < RandomNum):
                        GasIndex = GasIndex + 1

                # Pick random gas molecule velocity for collision
                MaxBoltzNumsUsed += 1
                if (MaxBoltzNumsUsed > 6):
                    GenerateMaxBoltz(Object.Random_Seed, Object.RandomMaxBoltzArray)
                    MaxBoltzNumsUsed = 1
                GasVelX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]
                MaxBoltzNumsUsed += 1
                GasVelY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]
                MaxBoltzNumsUsed += 1
                GasVelZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]

                if PyBoltz.Gas_Vel_Sigma != 0:
                    #print("it works")
                    GasVelX, GasVelY, GasVelZ = AustinGasVel(2, PyBoltz.Gas_Vel_Sigma, RandomSeed)

                #print(np.sqrt(pow(GasVelX,2)+pow(GasVelY,2)+pow(GasVelZ,2)))

                #Update velocity vectors following field acceleration
                if(Object.BFieldMode==1):
                    # No applied B Field
                    VelXAfter = DirCosineX1 * VelocityRatio * Sqrt2M * sqrt(EAfter)
                    VelYAfter = DirCosineY1 * VelocityRatio * Sqrt2M * sqrt(EAfter)
                    VelZAfter = (DirCosineZ1 * VelocityRatio +  T * F2 / (2.0*sqrt(EAfter))) * Sqrt2M * sqrt(EAfter)

                elif(Object.BFieldMode==2):
                    # B Field parallel to E field
                    # Calculate cyclotron motion
                    WBT = Object.AngularSpeedOfRotation * T
                    CosWT = cos(WBT)
                    SinWT = sin(WBT)

                    #Update velocity vectors following acceln
                    VelAfter = Sqrt2M * sqrt(EAfter)
                    VelXAfter = VelXBefore * CosWT - VelYBefore * SinWT
                    VelYAfter = VelYBefore * CosWT + VelXBefore * SinWT
                    VelZAfter = VelAfter * (DirCosineZ1 * VelocityRatio + Object.EField * T * (Object.CONST3/2.0) / sqrt(EAfter))

                # Calculate energy in center of mass frame
                #   E = 1/2 m dx^2 + dvy^2 + dvz^2
                #   works if TwoM = 2m
                COMEnergy = (pow((VelXAfter - GasVelX), 2) + pow((VelYAfter - GasVelY), 2) + pow((VelZAfter - GasVelZ), 2)) / TwoM

                # Which collision energy bin are we in? If we are too high, pin to 3999 (4000 is top)

                iEnergyBin = int(COMEnergy / Object.ElectronEnergyStep)

                iEnergyBin = min(iEnergyBin, 3999)

                # Now the Skullerud null collision method
                RandomNum = random_uniform(RandomSeed)

                # If we draw below this number, we will null-scatter (no mom xfer)
                Test1 = Object.TotalCollisionFrequency[GasIndex][iEnergyBin] / Object.MaxCollisionFreq[GasIndex]

                if RandomNum > Test1:
                    Test2 = TotalCollFreqIncludingNull[GasIndex][iEnergyBin] / Object.MaxCollisionFreq[GasIndex]
                    if RandomNum < Test2:
                        if Object.NumMomCrossSectionPointsNull[GasIndex] == 0:
                            continue
                        RandomNum = random_uniform(RandomSeed)
                        I = 0
                        while Object.NullCollisionFreq[GasIndex][iEnergyBin][I] < RandomNum:
                            # Add a null scatter
                            I += 1

                        Object.ICOLNN[GasIndex][I] += 1
                        continue
                    else:
                        Test3 = (TotalCollFreqIncludingNull[GasIndex][iEnergyBin] + AbsFakeIoniz) / Object.MaxCollisionFreq[GasIndex]
                        if RandomNum < Test3:
                            # Increment fake ionization counter
                            Object.FakeIonizations += 1
                            continue
                        continue
                else:
                    break
            # [end of while(True)]
            
            # If we got this far, we have a collision.
            NumCollisions += 1

            #  sqrt(2m E_com) = |v_com|
            VelocityInCOM  =  (Sqrt2M * sqrt(COMEnergy))
            
            # Calculate direction cosines of electron in 0 kelvin frame
            DXCOM = (VelXAfter - GasVelX) / VelocityInCOM
            DYCOM = (VelYAfter - GasVelY) / VelocityInCOM
            DZCOM = (VelZAfter - GasVelZ) / VelocityInCOM

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

            # Update position following acceleration
            VelocityBefore =  Sqrt2M * sqrt(EBefore)

            A = T * VelocityBefore
            if(Object.BFieldMode==1):
                # No applied B Field
                Object.X = Object.X + DirCosineX1 * A
                Object.Y = Object.Y + DirCosineY1 * A
                Object.Z = Object.Z + DirCosineZ1 * A + T2 * F1
                # Keep running total of Sum(V^2 T^2).
                VelX = DirCosineX1 * VelocityBefore
                VelY = DirCosineY1 * VelocityBefore

                SumVX = SumVX + VelX * VelX * T2
                SumVY = SumVY + VelY * VelY * T2

            elif(Object.BFieldMode==2):
                # B Feild parallel to E (cyclotron motion)
                DX = (VelXBefore * SinWT - VelYBefore * (1 - CosWT)) / Object.AngularSpeedOfRotation
                Object.X += DX
                DY = (VelYBefore * SinWT + VelXBefore * (1 - CosWT)) / Object.AngularSpeedOfRotation
                Object.Y += DY
                Object.Z += DirCosineZ1 * A + T2 * F1
                SumVX += DX**2
                SumVY += DY**2
            Object.TimeSum = Object.TimeSum + T

            Total_Coll += 1
            if ((Total_Coll)%(SwarmCollision))==0 and iSample>3 and  Object.Swarm==1:
                Object.SwarmX[Swarm_Index] = Object.X
                Object.SwarmY[Swarm_Index] = Object.Y
                Object.SwarmZ[Swarm_Index] = Object.Z
                Object.SwarmT[Swarm_Index] = Object.TimeSum
                Object.SwarmE[Swarm_Index] = EBefore
                Swarm_Index +=1

            # Figure out which time bin we're in, 299 is overflow; record collision
            #  at that time
            iTimeBin = int(T)
            iTimeBin = min(iTimeBin, 299)
            Object.CollisionTimes[iTimeBin] += 1

            # Record collision center of mass energy
            Object.CollisionEnergies[iEnergyBin] += 1

            # Record instantaneous integrated drift velocity
            Object.VelocityZ = Object.Z / Object.TimeSum



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
                CollsBackToLook = 0
                for iCorr in range(int(Object.Decor_Lookbacks)):
                    ST2 += T
                    DecorDistance = NumCollisions + CollsBackToLook
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
                    CollsBackToLook += Object.Decor_Step


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
            I = MBSortT(GasIndex, I, RandomNum, iEnergyBin, Object)
            while Object.CollisionFrequency[GasIndex][iEnergyBin][I] < RandomNum:
                I += 1
            S1 = Object.RGas[GasIndex][I]
            EI = Object.EnergyLevels[GasIndex][I]

            if Object.ElectronNumChange[GasIndex][I] > 0:
                # Use flat distribution of electron energy between E-IonizationEnergy and 0.0 EV, same as in Boltzmann
                RandomNum = random_uniform(RandomSeed)
                EXTRA = RandomNum * (COMEnergy - EI)
                EI = EXTRA + EI
                # If Auger fluorescence add extra ionisation collisions
                IEXTRA += <long long> Object.NC0[GasIndex][I]

            # Generate scattering angles and update laboratory cosines after collision also update energy of electron
            IPT = <long long> Object.InteractionType[GasIndex][I]
            Object.CollisionsPerGasPerType[GasIndex][<int> IPT - 1] += 1
            Object.ICOLN[GasIndex][I] += 1
            if COMEnergy < EI:
                EI = COMEnergy - 0.0001

            # If Penning is enabled and the energy transfer let to excitation,
            #  add probabilities to transfer energy to surrounding molecules
            if Object.Enable_Penning != 0:
                if Object.PenningFraction[GasIndex][0][I] != 0:
                    RandomNum = random_uniform(RandomSeed)
                    if RandomNum <= Object.PenningFraction[GasIndex][0][I]:
                        IEXTRA += 1
            S2 = pow(S1, 2) / (S1 - 1.0)

            # Anisotropic scattering - pick the scattering angle theta depending on scatter type
            RandomNum = random_uniform(RandomSeed)
            if Object.AngularModel[GasIndex][I] == 1:
	            # Use method of Capitelli et al
                RandomNum2 = random_uniform(RandomSeed)
                CosTheta = 1.0 - RandomNum * Object.AngleCut[GasIndex][iEnergyBin][I]
                if RandomNum2 > Object.ScatteringParameter[GasIndex][iEnergyBin][I]:
                    CosTheta = -1.0 * CosTheta
            elif Object.AngularModel[GasIndex][I] == 2:
                # Use method of Okhrimovskyy et al
                EpsilonOkhr = Object.ScatteringParameter[GasIndex][iEnergyBin][I]
                CosTheta = 1.0 - (2.0 * RandomNum * (1.0 - EpsilonOkhr) / (1.0 + EpsilonOkhr * (1.0 - 2.0 * RandomNum)))
            else:
                # Isotropic scattering
                CosTheta = 1.0 - 2.0 * RandomNum
            Theta = acos(CosTheta)

            # Pick a random Phi - must be uniform by symmetry of the gas
            RandomNum = random_uniform(RandomSeed)
            Phi = TwoPi * RandomNum
            SinPhi = sin(Phi)
            CosPhi = cos(Phi)

            #TODO: Understand what Arg1, D, U and Q are
            
            ARG1 = max(1.0 - S1 * EI / COMEnergy, Object.SmallNumber)

            D = 1.0 - CosTheta * sqrt(ARG1)
            U = (S1 - 1) * (S1 - 1) / ARG1

            # Update the energy to start drifing for the next round.
            #  If its zero, make it small but nonzero.
            EBefore = max(COMEnergy * (1.0 - EI / (S1 * COMEnergy) - 2.0 * D / S2), Object.SmallNumber)
                       
            Q = min(sqrt((COMEnergy / EBefore) * ARG1) / S1,1.0)

            # Calculate angle of scattering from Z direction
            Object.AngleFromZ = asin(Q * sin(Theta))
            CosZAngle = cos(Object.AngleFromZ)

            # Find new directons after scatter
            if CosTheta < 0 and CosTheta * CosTheta > U:
                CosZAngle = -1 * CosZAngle
            SinZAngle = sin(Object.AngleFromZ)
            DZCOM = min(DZCOM, 1.0)
            ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
            if ARGZ < 1e-6:
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
            VelBefore = Sqrt2M * sqrt(EBefore)
            VelXBefore = DirCosineX1 * VelBefore + GasVelX
            VelYBefore = DirCosineY1 * VelBefore + GasVelY
            VelZBefore = DirCosineZ1 * VelBefore + GasVelZ

            # Calculate energy and direction cosines in lab frame
            EBefore = (VelXBefore * VelXBefore + VelYBefore * VelYBefore + VelZBefore * VelZBefore) / TwoM
            VelBeforeM1 = 1/(Sqrt2M * sqrt(EBefore))
            DirCosineX1 = VelXBefore * VelBeforeM1
            DirCosineY1 = VelYBefore * VelBeforeM1
            DirCosineZ1 = VelZBefore * VelBeforeM1

            #And go around again to the next collision!
            
        #Put stuff into the right units and normalize
        Object.VelocityZ *= 1.0e9
        Object.MeanElectronEnergy = SumE2 / Object.TimeSum
        Object.LongitudinalDiffusion = 0.0
        if Object.AnisotropicDetected == 0:
            Object.DiffusionX = 5.0e15 * SumVX / Object.TimeSum
            Object.DiffusionY = 5.0e15 * SumVY / Object.TimeSum
            DiffXXPerSample[iSample] = 5.0e15 * (SumVX - SVX_LastSample) / (Object.TimeSum - ST_LastSample)
            DiffYYPerSample[iSample] = 5.0e15 * (SumVY - SVY_LastSample) / (Object.TimeSum - ST_LastSample)
        else:

            if ST2 != 0.0:
                Object.DiffusionY = 5.0e15 * SumYY / ST2
                Object.DiffusionX = 5.0e15 * SumXX / ST2
                DiffXXPerSample[iSample] = 5.0e15 * (SumXX - SumXX_LastSample) / (ST2 - ST2_LastSample)
                DiffYYPerSample[iSample] = 5.0e15 * (SumYY - SumYY_LastSample) / (ST2 - ST2_LastSample)

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
            print(
                '{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(Object.VelocityZ, Object.Z, Object.TimeSum,
                                                                                         Object.MeanElectronEnergy, Object.DiffusionX,
                                                                                         Object.DiffusionY,
                                                                                         Object.DiffusionZ))
        if Object.CollisionEnergies[3999] > (1000 * float(iSample + 1)):
            raise ValueError("WARNING ENERGY OUT OF RANGE, INCREASE ELECTRON ENERGY INTEGRATION RANGE")


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
    Object.ErrorDiffusionZ = 100 * sqrt((SumDZZ2_Samples - SumDZZ_Samples ** 2 / (Object.Num_Samples - 2)) / (Object.Num_Samples - 2 - 1)) / Object.DiffusionZ
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
    for i in range(6):
        free(TotalCollFreqIncludingNull[i])
    free(TotalCollFreqIncludingNull)
    return Object
