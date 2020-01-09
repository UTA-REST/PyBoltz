from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow
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


cdef int NewPrimary(PyBoltz Object, MonteVars* MV):
    MV.IPrimary+=1
    if MV.IPrimary>1:
        if MV.Iterator>MV.NumberOfMaxColli:
            MV.IPrimary-=1
            return 0
        else:
            Object.X = 0.0
            Object.Y =0.0
            Object.Z = 0.0
            MV.DirCosineX1 = MV.DirCosineX100
            MV.DirCosineY1 = MV.DirCosineY100
            MV.DirCosineZ1 = MV.DirCosineZ100
            MV.StartingEnergy = MV.Energy100
            MV.NCLUS +=1
            Object.ST = 0.0
            MV.TimeSumStart= 0.0
            MV.SpaceZStart=0.0
            MV.IPlane = 0
    if MV.IPrimary>10000000:
        print("Too many primaries program stopped!")
        return 0

    return 1

cdef void NewElectron(PyBoltz Object, MonteVars* MV):
    MV.TDash = 0.0
    MV.NumberOfElectron +=1
    MV.TimeStop = MV.IPlane* Object.TimeStep


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object,int ConsoleOuput):

    # All the simulation variables are put into the MonteVars struct
    cdef MonteVars MV
    cdef double RandomSeed = 0.3,RandomNum
    cdef double XS[2000],YS[2000],ZS[2000],TS[2000],ES[2000],DirCosineX[2000],DirCosineY[2000],DirCosineZ[2000]
    cdef double GasVelX,GasVelY,GasVelZ,VelocityRatio,VelXAfter,VelYAfter,VelZAfter,COMEnergy,Test1,
    cdef int IPlaneS[2000],Flag,GasIndex,MaxBoltzNumsUsed

    if ConsoleOuput !=0:
        MV.NumberOfMaxColli = Object.MaxNumberOfCollisions

    MV.StartingEnergy = Object.InitialElectronEnergy

    MV.Sqrt2M = Object.CONST3 * 0.01               # This should be: sqrt(2m)
    MV.TwoM   =  pow(MV.Sqrt2M, 2)                    # This should be: 2m
    MV.TwoPi = 2.0 * np.pi                         # This should be: 2 Pi

    Object.reset()

    cdef int i = 0,K,J
    cdef double ** TotalCollFreqIncludingNull = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TotalCollFreqIncludingNull[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TotalCollFreqIncludingNull[K][J] = Object.TotalCollisionFrequency[K][J] + Object.TotalCollisionFrequencyNull[K][J]
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
    GenerateMaxBoltz(Object.RandomSeed,  Object.RandomMaxBoltzArray)
    MaxBoltzNumsUsed = 0

    # Here are some constants we will use
    MV.BP = pow(Object.EField, 2) * Object.CONST1  # This should be: 1/2 m e^2 EField^2
    MV.F1 = Object.EField * Object.CONST2
    MV.F2 = Object.EField * Object.CONST3          # This should be: sqrt( m / 2) e EField

    MV.PrintN = MV.NumberOfMaxColli/10
    # Create primary electron
    NewPrimary(Object, &MV)
    # register this electron
    NewElectron(Object, &MV)
    while(1):
        # Sample random time to next collision. T is global total time.
        RandomNum = random_uniform(RandomSeed)

        # This is the formula from Skullerud
        MV.T = -log(RandomNum) / Object.MaxCollisionFreqTotal + MV.TDash
        MV.TDash = MV.T
        MV.AP = MV.DirCosineZ1 * MV.F2 * sqrt(MV.StartingEnergy)

        '''
        call TPLANET
        '''
        if MV.T + Object.TimeSum >= MV.TimeStop:
            while MV.T + Object.TimeSum >= MV.TimeStop:
                MV.IPlane += 1
                MV.TimeStop +=  Object.TimeStep
                #TPLANET(object,T, E1, DCX1, DCY1, DCZ1, AP, BP, IPLANE - 1)
                if MV.T + Object.TimeSum >= MV.TimeStop and MV.TimeStop <= Object.MaxTime:
                    continue
                else:
                    break
            if MV.T + Object.TimeSum  >= Object.MaxTime:
                Object.TotalSpaceZPrimary += Object.Z
                Object.TotalTimePrimary += Object.TimeSum
                Object.TotalSpaceZSecondary += Object.Z - MV.SpaceZStart
                Object.TotalTimeSecondary += Object.TimeSum - MV.TimeSumStart
                MV.TimeStop = Object.TimeStep
                if MV.NumberOfElectron == MV.NCLUS + 1:
                    # Create primary electron
                    Flag = NewPrimary(Object, &MV)
                    if Flag==0:
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
        MV.Energy = MV.StartingEnergy +(MV.AP + MV.BP  * MV.T)

        if MV.Energy:
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
            GenerateMaxBoltz(Object.RandomSeed,  Object.RandomMaxBoltzArray)
            MaxBoltzNumsUsed = 1
        GasVelX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]
        MaxBoltzNumsUsed += 1
        GasVelY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]
        MaxBoltzNumsUsed += 1
        GasVelZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(MaxBoltzNumsUsed - 1)]

        #Update velocity vectors following field acceleration
        VelocityRatio = sqrt(MV.StartingEnergy/MV.Energy)
        VelXAfter = MV.DirCosineX1 * VelocityRatio * MV.Sqrt2M * sqrt(MV.Energy)
        VelYAfter = MV.DirCosineY1 * VelocityRatio * MV.Sqrt2M * sqrt(MV.Energy)
        VelZAfter = (MV.DirCosineZ1 * VelocityRatio +  MV.T * MV.F2 / (2.0*sqrt(MV.Energy))) * MV.Sqrt2M * sqrt(MV.Energy)

        # Calculate energy in center of mass frame
        #   E = 1/2 m dx^2 + dvy^2 + dvz^2
        #   works if TwoM = 2m
        COMEnergy = (pow((VelXAfter - GasVelX), 2) + pow((VelYAfter - GasVelY), 2) + pow((VelZAfter - GasVelZ), 2)) / MV.TwoM

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




