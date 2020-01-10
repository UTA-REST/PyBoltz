from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan
from libc.string cimport memset
from PyBoltz cimport drand48
from MBSorts cimport MBSort
from libc.stdlib cimport malloc, free
import MonteVars
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

cdef int NewPrimary(PyBoltz Object, MonteVars*MV):
    MV.IPrimary += 1
    if MV.IPrimary > 1:
        if MV.Iterator > MV.NumberOfMaxColli:
            MV.IPrimary -= 1
            return 0
        else:
            Object.X = 0.0
            Object.Y = 0.0
            Object.Z = 0.0
            MV.DirCosineX1 = MV.DirCosineX100
            MV.DirCosineY1 = MV.DirCosineY100
            MV.DirCosineZ1 = MV.DirCosineZ100
            MV.StartingEnergy = MV.Energy100
            MV.NCLUS += 1
            Object.ST = 0.0
            MV.TimeSumStart = 0.0
            MV.SpaceZStart = 0.0
            MV.IPlane = 0
    if MV.IPrimary > 10000000:
        print("Too many primaries program stopped!")
        return 0

    return 1

cdef void NewElectron(PyBoltz Object, MonteVars*MV):
    MV.TDash = 0.0
    MV.NumberOfElectron += 1
    MV.TimeStop = MV.IPlane * Object.TimeStep

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object, int ConsoleOuput):
    # All the simulation variables are put into the MonteVars struct
    cdef MonteVars MV
    cdef double RandomSeed = 0.3, RandomNum, S1, EI, ESEC, EISTR, F3
    cdef double XS[2001], YS[2001], ZS[2001], TS[2001], ES[2001], DirCosineX[2001], DirCosineY[2001], DirCosineZ[2001], EAuger
    cdef double GasVelX, GasVelY, GasVelZ, VelocityRatio, VelXAfter, VelYAfter, VelZAfter, COMEnergy, Test1, A, VelocityInCOM, T2
    cdef int IPlaneS[2001], Flag, GasIndex, MaxBoltzNumsUsed, NumCollisions = 0, I, IPT, NCLTMP, IAuger, J

    if ConsoleOuput != 0:
        MV.NumberOfMaxColli = Object.MaxNumberOfCollisions

    MV.StartingEnergy = Object.InitialElectronEnergy

    MV.Sqrt2M = Object.CONST3 * 0.01  # This should be: sqrt(2m)
    MV.TwoM = pow(MV.Sqrt2M, 2)  # This should be: 2m
    MV.TwoPi = 2.0 * np.pi  # This should be: 2 Pi

    Object.reset()

    cdef int i = 0, K, J
    cdef double ** TotalCollFreqIncludingNull = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TotalCollFreqIncludingNull[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TotalCollFreqIncludingNull[K][J] = Object.TotalCollisionFrequency[K][J] + \
                                               Object.TotalCollisionFrequencyNull[K][J]
    AbsFakeIoniz = Object.FakeIonisationsEstimate

    for J in range(8):
        MV.FakeIonisationsTime[J] = 0.0
        MV.FakeIonisationsSpace[J] = 0.0

    # Initial direction cosines
    MV.DirCosineZ1 = cos(Object.AngleFromZ)
    MV.DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    MV.DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    MV.Energy100 = MV.StartingEnergy

    MV.DirCosineZ100 = MV.DirCosineZ1
    MV.DirCosineX100 = MV.DirCosineX1
    MV.DirCosineY100 = MV.DirCosineY1

    # Generate initial random maxwell boltzman numbers
    GenerateMaxBoltz(Object.RandomSeed, Object.RandomMaxBoltzArray)
    MaxBoltzNumsUsed = 0

    # Here are some constants we will use
    MV.BP = pow(Object.EField, 2) * Object.CONST1  # This should be: 1/2 m e^2 EField^2
    MV.F1 = Object.EField * Object.CONST2
    MV.F2 = Object.EField * Object.CONST3  # This should be: sqrt( m / 2) e EField

    MV.PrintN = MV.NumberOfMaxColli / 10
    # Create primary electron
    NewPrimary(Object, &MV)
    # register this electron
    NewElectron(Object, &MV)
    while (1):
        # Sample random time to next collision. T is global total time.
        RandomNum = random_uniform(RandomSeed)

        # This is the formula from Skullerud
        MV.T = -log(RandomNum) / Object.MaxCollisionFreqTotal + MV.TDash
        MV.TDash = MV.T
        MV.AP = MV.DirCosineZ1 * MV.F2 * sqrt(MV.StartingEnergy)

        '''
        call TPLANET
        '''

        # check if the total time is bigger than the time for the current plane(TimeStop)
        if MV.T + Object.TimeSum >= MV.TimeStop:

            # while the total time is bigger than the time for the current plane
            while MV.T + Object.TimeSum >= MV.TimeStop:

                # move to the next plane and update the time for the current plane
                MV.IPlane += 1
                MV.TimeStop += Object.TimeStep
                #TPLANET(object,T, E1, DCX1, DCY1, DCZ1, AP, BP, IPLANE - 1)

                if MV.T + Object.TimeSum >= MV.TimeStop and MV.TimeStop <= Object.MaxTime:
                    continue
                else:
                    break
            if MV.T + Object.TimeSum >= Object.MaxTime:
                # total time is bigger than the maximum time set for the simulation
                # update the space and time for the two electrons
                Object.TotalSpaceZPrimary += Object.Z
                Object.TotalTimePrimary += Object.TimeSum
                Object.TotalSpaceZSecondary += Object.Z - MV.SpaceZStart
                Object.TotalTimeSecondary += Object.TimeSum - MV.TimeSumStart
                MV.TimeStop = Object.TimeStep
                if MV.NumberOfElectron == MV.NCLUS + 1:
                    # Create primary electron
                    Flag = NewPrimary(Object, &MV)
                    if Flag == 0:
                        break
                    continue
                Object.X = XS[MV.NPONT]
                Object.Y = YS[MV.NPONT]
                Object.Z = ZS[MV.NPONT]
                Object.TimeSum = TS[MV.NPONT]
                MV.StartingEnergy = ES[MV.NPONT]
                MV.DirCosineX1 = DirCosineX[MV.NPONT]
                MV.DirCosineY1 = DirCosineY[MV.NPONT]
                MV.DirCosineZ1 = DirCosineZ[MV.NPONT]
                MV.IPlane = IPlaneS[MV.NPONT]
                MV.NPONT -= 1
                MV.SpaceZStart = Object.Z
                MV.TimeSumStart = Object.TimeSum

            # register this electron
            NewElectron(Object, &MV)
            continue
        # similar to MONTET, calculate the energy of the electron after the collision
        MV.Energy = MV.StartingEnergy + (MV.AP + MV.BP * MV.T)

        if MV.Energy < 0.0:
            MV.Energy = 0.001

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
            GenerateMaxBoltz(Object.RandomSeed, Object.RandomMaxBoltzArray)
            MaxBoltzNumsUsed = 1
        GasVelX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]
        MaxBoltzNumsUsed += 1
        GasVelY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]
        MaxBoltzNumsUsed += 1
        GasVelZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]

        #Update velocity vectors following field acceleration
        VelocityRatio = sqrt(MV.StartingEnergy / MV.Energy)
        VelXAfter = MV.DirCosineX1 * VelocityRatio * MV.Sqrt2M * sqrt(MV.Energy)
        VelYAfter = MV.DirCosineY1 * VelocityRatio * MV.Sqrt2M * sqrt(MV.Energy)
        VelZAfter = (MV.DirCosineZ1 * VelocityRatio + MV.T * MV.F2 / (2.0 * sqrt(MV.Energy))) * MV.Sqrt2M * sqrt(
            MV.Energy)

        # Calculate energy in center of mass frame
        #   E = 1/2 m dx^2 + dvy^2 + dvz^2
        #   works if TwoM = 2m
        COMEnergy = (pow((VelXAfter - GasVelX), 2) + pow((VelYAfter - GasVelY), 2) + pow((VelZAfter - GasVelZ),
                                                                                         2)) / MV.TwoM

        # Now the Skullerud null collision method
        RandomNum = random_uniform(RandomSeed)
        MV.iEnergyBin = COMEnergy / Object.ElectronEnergyStep
        MV.iEnergyBin = min(3999, MV.iEnergyBin)

        # If we draw below this number, we will null-scatter (no mom xfer)
        Test1 = Object.TotalCollisionFrequency[GasIndex][MV.iEnergyBin] / Object.MaxCollisionFreq[GasIndex]

        if RandomNum > Test1:
            Test2 = TotalCollFreqIncludingNull[GasIndex][MV.iEnergyBin] / Object.MaxCollisionFreq[GasIndex]
            if RandomNum < Test2:
                if Object.NumMomCrossSectionPointsNull[GasIndex] == 0:
                    continue
                RandomNum = random_uniform(RandomSeed)
                I = 0
                while Object.NullCollisionFreq[GasIndex][MV.iEnergyBin][I] < RandomNum:
                    # Add a null scatter
                    I += 1

                Object.ICOLNN[GasIndex][I] += 1
                continue
            else:
                Test3 = (TotalCollFreqIncludingNull[GasIndex][MV.iEnergyBin] + AbsFakeIoniz) / Object.MaxCollisionFreq[
                    GasIndex]
                if RandomNum < Test3:
                    # Increment fake ionization counter
                    Object.FakeIonizations += 1
                    MV.FakeIonisationsTime[MV.IPlane] += 1
                    if Object.FakeIonizations < 0.0:
                        MV.NumberOfElectronIon += 1

                        # Fake attachment start a new electron
                        if MV.NumberOfElectron == MV.NCLUS + 1:
                            # Create primary electron if the number of cascaded electrons is reached
                            Flag = NewPrimary(Object, &MV)
                            if Flag == 0:
                                break
                            continue

                        # get a new electron
                        Object.X = XS[MV.NPONT]
                        Object.Y = YS[MV.NPONT]
                        Object.Z = ZS[MV.NPONT]
                        Object.TimeSum = TS[MV.NPONT]
                        MV.StartingEnergy = ES[MV.NPONT]
                        MV.DirCosineX1 = DirCosineX[MV.NPONT]
                        MV.DirCosineY1 = DirCosineY[MV.NPONT]
                        MV.DirCosineZ1 = DirCosineZ[MV.NPONT]
                        MV.IPlane = IPlaneS[MV.NPONT]
                        MV.NPONT -= 1
                        MV.SpaceZStart = Object.Z
                        MV.TimeSumStart = Object.TimeSum

                        # register this electron
                        NewElectron(Object, &MV)
                        continue

                    # fake Ionisation add electron to the store (S)

                    MV.NCLUS += 1
                    MV.NPONT += 1
                    MV.NMXADD = max(MV.NMXADD, MV.NPONT)

                    if MV.NPONT > 2000:
                        raise ValueError("More than 2000 stored electrons")
                    A = MV.T * MV.Sqrt2M * sqrt(MV.StartingEnergy)
                    XS[MV.NPONT] = Object.X + MV.DirCosineX1 * A
                    YS[MV.NPONT] = Object.Y + MV.DirCosineY1 * A
                    ZS[MV.NPONT] = Object.Z + MV.DirCosineZ1 * A
                    TS[MV.NPONT] = Object.TimeSum + MV.T
                    ES[MV.NPONT] = MV.Energy
                    IPlaneS[MV.NPONT] = MV.IPlane
                    DirCosineX[MV.NPONT] = MV.DirCosineX1 * VelocityRatio
                    DirCosineY[MV.NPONT] = MV.DirCosineY1 * VelocityRatio
                    DirCosineZ[MV.NPONT] = (MV.DirCosineZ1 * VelocityRatio + MV.T * MV.F2 / (2.0 * sqrt(MV.Energy)))

                continue
            continue

        # If we got this far, we have a collision.
        NumCollisions += 1

        #  sqrt(2m E_com) = |v_com|
        VelocityInCOM = (Sqrt2M * sqrt(COMEnergy))

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
        # Calculate the positions before collision and update diffusion and energy calculation
        T2 = MV.T * MV.T
        A = MV.T * MV.Sqrt2M * sqrt(MV.StartingEnergy)
        Object.X += MV.DirCosineX1 * A
        Object.Y += MV.DirCosineY1 * A
        Object.Z += MV.DirCosineZ1 * A + T2 * MV.F1

        # update energy and time bins 
        Object.TimeSum += MV.T
        Object.CollisionTimes[min(299, int(MV.T + 1))] += 1
        Object.CollisionEnergies[MV.iEnergyBin] += 1

        # Randomly pick the type of collision we will have
        RandomNum = random_uniform(RandomSeed)

        # Find location within 4 units in collision array
        I = MBSortT(GasIndex, I, RandomNum, iEnergyBin, Object)
        while Object.CollisionFrequency[GasIndex][iEnergyBin][I] < RandomNum:
            I += 1

        S1 = Object.RGas[GasIndex][I]
        EI = Object.EnergyLevels[GasIndex][I]

        if COMEnergy < EI:
            EI = COMEnergy - 0.0001

        if Object.ElectronNumChange[GasIndex][I] == -1:
            # An attachment happened
            MV.NumberOfElectronIon += 1
            IPT = <long long> Object.InteractionType[GasIndex][I]
            MV.ID += 1
            MV.Iterator += 1
            MV.IPrint += 1
            Object.CollisionsPerGasPerType[GasIndex][IPT] += 1
            Object.ICOLN[GasIndex][I] += 1
            Object.CollisionTimes[min(299, int(MV.T + 1))] += 1

            Object.TotalSpaceZPrimary += Object.Z
            Object.TotalTimePrimary += Object.TimeSum
            Object.TotalSpaceZSecondary += Object.Z - MV.SpaceZStart
            Object.TotalTimeSecondary += Object.TimeSum - MV.TimeSumStart

            if MV.NumberOfElectron == MV.NCLUS + 1:
                # Electron captured start a new primary
                Flag = NewPrimary(Object, &MV)
                if Flag == 0:
                    break
                continue
            # get a new electron
            Object.X = XS[MV.NPONT]
            Object.Y = YS[MV.NPONT]
            Object.Z = ZS[MV.NPONT]
            Object.TimeSum = TS[MV.NPONT]
            MV.StartingEnergy = ES[MV.NPONT]
            MV.DirCosineX1 = DirCosineX[MV.NPONT]
            MV.DirCosineY1 = DirCosineY[MV.NPONT]
            MV.DirCosineZ1 = DirCosineZ[MV.NPONT]
            MV.IPlane = IPlaneS[MV.NPONT]
            MV.NPONT -= 1
            MV.SpaceZStart = Object.Z
            MV.TimeSumStart = Object.TimeSum

            # register this electron
            NewElectron(Object, &MV)
            continue
        elif Object.ElectronNumChange[GasIndex][I] == 1:
            # An ionisation happened
            EISTR = EI
            RandomNum = random_uniform(RandomSeed)

            # Use Opal Peterson and Beaty splitting factor
            ESEC = Object.WPL[GasIndex][I] * tan(RandomNum * atan((COMEnergy - EI) / (2 * Object.WPL[GasIndex][I])))
            ESEC = Object.WPL[GasIndex][I] * (ESEC / Object.WPL[GasIndex][I]) ** 0.9524

            EI = ESEC + EI

            # Store position, energy, direction, and time of generation of ionisation electron
            MV.NCLUS += 1
            MV.NPONT += 1
            MV.NMXADD = max(MV.NMXADD, MV.NPONT)
            if MV.NPONT > 2000:
                raise ValueError("More than 2000 stored electrons")
            XS[MV.NPONT] = Object.X
            YS[MV.NPONT] = Object.Y
            ZS[MV.NPONT] = Object.Z
            TS[MV.NPONT] = Object.TimeSum
            ES[MV.NPONT] = ESEC
            IPlaneS[MV.NPONT] = MV.IPlane

            MV.NTPMFlag = 1
            # Store possible shell emissions auger or fluorescence, update the angles and cosines
            if EISTR > 30.0:
                # Auger Emission without fluorescence
                NAuger = Object.NC0[GasIndex][I]
                EAuger = Object.EC0[GasIndex][I] / NAuger
                for J in range(NAuger):
                    MV.NCLUS += 1
                    MV.NPONT += 1
                    XS[MV.NPONT] = Object.X
                    YS[MV.NPONT] = Object.Y
                    ZS[MV.NPONT] = Object.Z
                    TS[MV.NPONT] = Object.TimeSum
                    ES[MV.NPONT] = EAuger
                    RandomNum = random_uniform(RandomSeed)

                    F3 = 1.0 - 2 * RandomNum
