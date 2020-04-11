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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object):
    print("THIS IS SWARM!!")
    
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
    

    SwarmCollision = <long long> Object.MaxNumberOfCollisions / 100


    #for iSample in range(int(Object.Num_Samples)):
    for iCollision in range(int(Object.MaxNumberOfCollisions)):
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

        #Add to the swarm
        if ((iCollision)%(SwarmCollision))==0:
            if ((Swarm_Index)%10) == 0:
                print("SWARM")
            Object.SwarmX[Swarm_Index] = Object.X
            Object.SwarmY[Swarm_Index] = Object.Y
            Object.SwarmZ[Swarm_Index] = Object.Z
            Object.SwarmT[Swarm_Index] = Object.TimeSum
            Object.SwarmE[Swarm_Index] = EBefore
            Swarm_Index +=1



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


        # Figure out which time bin we're in, 299 is overflow; record collision
        #  at that time
        iTimeBin = int(T)
        iTimeBin = min(iTimeBin, 299)
        Object.CollisionTimes[iTimeBin] += 1

        # Record collision center of mass energy
        Object.CollisionEnergies[iEnergyBin] += 1



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
            


    # Free up the memory and close down gracefully
    for i in range(6):
        free(TotalCollFreqIncludingNull[i])
    free(TotalCollFreqIncludingNull)
    return Object
