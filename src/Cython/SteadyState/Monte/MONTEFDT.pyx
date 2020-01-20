from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan
from libc.string cimport memset
from PyBoltz cimport drand48
from MBSorts cimport MBSortT
from libc.stdlib cimport malloc, free
from MonteVars cimport MonteVars
import cython
import numpy as np
cimport numpy as np
import sys
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
    Object.IPrimary += 1
    MV.IPlane = 0
    MV.TimeStop = 1000.0
    if Object.IPrimary > 1:
        if MV.Iterator > MV.NumberOfMaxColli:
            Object.IPrimary -= 1
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
            Object.TimeSum = 0.0
            MV.TimeSumStart = 0.0
            MV.SpaceZStart = 0.0
            MV.IPlane = 0
    if Object.IPrimary > 10000000:
        print("Too many primaries program stopped!")
        return 0

    return 1

cdef void NewElectron(PyBoltz Object, MonteVars*MV):
    MV.TDash = 0.0
    MV.NumberOfElectron += 1
cdef void SpacePlaneUpdate(PyBoltz Object, MonteVars*MV):
    cdef double CurrentTime, TimeLeft, A, B, EPlane, VelocityRatio, TimeLeft2, DirCosineZ2
    cdef double XPlane, YPlane, ZPlane, VZPlane, WGHT, RPlane
    if MV.IPlane > 8:
        return
    TimeLeft = MV.TimeStop
    TimeLeft2 = TimeLeft * TimeLeft
    A = MV.AP * TimeLeft
    B = MV.BP * TimeLeft2
    EPlane = MV.StartingEnergy + A + B
    VelocityRatio = sqrt(MV.StartingEnergy / EPlane)
    DirCosineZ2 = (MV.DirCosineZ1 * VelocityRatio + TimeLeft * MV.F2 / (2.0 * sqrt(EPlane)))

    XPlane = Object.X + MV.DirCosineX1 * TimeLeft * sqrt(MV.StartingEnergy) * Object.CONST3 * 0.01
    YPlane = Object.Y + MV.DirCosineY1 * TimeLeft * sqrt(MV.StartingEnergy) * Object.CONST3 * 0.01
    ZPlane = Object.Z + MV.DirCosineZ1 * TimeLeft * sqrt(
        MV.StartingEnergy) * Object.CONST3 * 0.01 + TimeLeft2 * Object.EField * Object.CONST2

    VZPlane = DirCosineZ2 * sqrt(EPlane) * Object.CONST3 * 0.01

    WGHT = abs(1 / VZPlane)
    RPlane = sqrt(XPlane ** 2 + YPlane ** 2)

    Object.SXPlanes[MV.IPlane] += XPlane * WGHT
    Object.SYPlanes[MV.IPlane] += YPlane * WGHT
    Object.SZPlanes[MV.IPlane] += ZPlane * WGHT
    Object.RSPL[MV.IPlane] += RPlane * WGHT

    Object.TMSPL[MV.IPlane]+=(Object.TimeSum+TimeLeft)*WGHT
    Object.TTMSPL[MV.IPlane]+=    (Object.TimeSum+TimeLeft)*(Object.TimeSum+TimeLeft)*WGHT

    Object.SX2Planes[MV.IPlane]+= XPlane*XPlane*WGHT
    Object.SY2Planes[MV.IPlane]+= YPlane*YPlane*WGHT
    Object.SZ2Planes[MV.IPlane]+= ZPlane*ZPlane*WGHT
    Object.RRSPM[MV.IPlane]+= RPlane*RPlane*WGHT

    Object.SEPlanes[MV.IPlane]+= EPlane*WGHT
    Object.STPlanes[MV.IPlane]+= WGHT/(Object.TimeSum+TimeLeft)

    Object.SVZPlanes[MV.IPlane]+= VZPlane*WGHT

    Object.STSPlanes[MV.IPlane] +=WGHT
    Object.STS2Planes[MV.IPlane]+=WGHT*WGHT


cdef int GetRightElectron(PyBoltz Object,MonteVars  * MV):
    cdef double EPOT
    while(1):
        EPOT = Object.EField  *(Object.Z - Object.MaxSpaceZ)*100
        if MV.StartingEnergy< EPOT:
            MV.NumberOfElectron+=1
            MV.ISolution = 1
            Object.TotalSpaceZPrimary+=Object.Z
            Object.TotalTimePrimary +=Object.TimeSum
            Object.TotalTimeSecondary+=Object.TimeSum - MV.TimeSumStart
            Object.TotalSpaceZSecondary+=Object.Z - MV.SpaceZStart
            if MV.NumberOfElectron == MV.NCLUS + 1:
                # Create primary electron
                Flag = NewPrimary(Object, &MV)
                NewElectron(Object, &MV)
                if Flag == 0:
                    return 0
                return 1
            # Take an electron from the store
            Object.X = MV.XS[MV.NPONT]
            Object.Y = MV.YS[MV.NPONT]
            Object.Z = MV.ZS[MV.NPONT]
            Object.TimeSum = MV.TS[MV.NPONT]
            MV.StartingEnergy = MV.ES[MV.NPONT]
            MV.DirCosineX1 = MV.DirCosineX[MV.NPONT]
            MV.DirCosineY1 = MV.DirCosineY[MV.NPONT]
            MV.DirCosineZ1 = MV.DirCosineZ[MV.NPONT]
            MV.IPlane = MV.IPlaneS[MV.NPONT]
            MV.NPONT -= 1
            MV.SpaceZStart = Object.Z
            MV.TimeSumStart = Object.TimeSum
        else:
            break
    return 2

cdef int GetRightElectronn(PyBoltz Object,MonteVars  * MV):
    cdef int Flag = 0
    while(1):
        # call Tclac
        if MV.TimeStop==-99:
            # catch runaway electrons at high field
            MV.NumberOfElectron+=1
            MV.ISolution = 1
            Object.TotalSpaceZPrimary+=Object.Z
            Object.TotalTimePrimary +=Object.TimeSum
            Object.TotalTimeSecondary+=Object.TimeSum - MV.TimeSumStart
            Object.TotalSpaceZSecondary+=Object.Z - MV.SpaceZStart
            if MV.NumberOfElectron == MV.NCLUS + 1:
                # Create primary electron
                Flag = NewPrimary(Object, &MV)
                NewElectron(Object, &MV)
                if Flag == 0:
                    return 0
                return 1
            # Take an electron from the store
            Object.X = MV.XS[MV.NPONT]
            Object.Y = MV.YS[MV.NPONT]
            Object.Z = MV.ZS[MV.NPONT]
            Object.TimeSum = MV.TS[MV.NPONT]
            MV.StartingEnergy = MV.ES[MV.NPONT]
            MV.DirCosineX1 = MV.DirCosineX[MV.NPONT]
            MV.DirCosineY1 = MV.DirCosineY[MV.NPONT]
            MV.DirCosineZ1 = MV.DirCosineZ[MV.NPONT]
            MV.IPlane = MV.IPlaneS[MV.NPONT]
            MV.NPONT -= 1
            MV.SpaceZStart = Object.Z
            MV.TimeSumStart = Object.TimeSum
            Flag = GetRightElectron()
            if Flag !=2:
                return Flag
        else:
            break
    return 2



cdef int TimeCalculations(PyBoltz Object, MonteVars*MV):
    cdef Flag
    if (MV.T >= MV.TimeStop and MV.TOld < MV.TimeStop):
        SpacePlaneUpdate(Object,&MV)
        #TODO: change to NumberOfSpaceSteps-1 for anode termination
        if MV.IPlane>= Object.NumberOfSpaceSteps+1:
            Object.TotalSpaceZPrimary+=Object.Z
            Object.TotalTimePrimary +=Object.TimeSum
            Object.TotalTimeSecondary+=Object.TimeSum - MV.TimeSumStart
            Object.TotalSpaceZSecondary+=Object.Z - MV.SpaceZStart
            if MV.NumberOfElectron == MV.NCLUS + 1:
                # Create primary electron
                Flag = NewPrimary(Object, &MV)
                NewElectron(Object, &MV)
                if Flag == 0:
                    return 0
                return 1
            # Take an electron from the store
            Object.X = MV.XS[MV.NPONT]
            Object.Y = MV.YS[MV.NPONT]
            Object.Z = MV.ZS[MV.NPONT]
            Object.TimeSum = MV.TS[MV.NPONT]
            MV.StartingEnergy = MV.ES[MV.NPONT]
            MV.DirCosineX1 = MV.DirCosineX[MV.NPONT]
            MV.DirCosineY1 = MV.DirCosineY[MV.NPONT]
            MV.DirCosineZ1 = MV.DirCosineZ[MV.NPONT]
            MV.IPlane = MV.IPlaneS[MV.NPONT]
            MV.NPONT -= 1
            MV.SpaceZStart = Object.Z
            MV.TimeSumStart = Object.TimeSum
            Flag = GetRightElectron(Object,MV)
            if Flag !=2:
                return Flag

            Flag = GetRightElectronn(Object,MV)
            if Flag !=2:
                return Flag
            NewElectron(Object, &MV)
            return 1
        else:
            if MV.ISolution==2:
                Object.MaxSpaceZ = MV.MaxSpaceZ1
                MV.ISolution =1
                return 3
    return 2



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object, int ConsoleOuput):
    cdef MonteVars MV
    cdef double VelBefore, VelXBefore, VelYBefore, VelZBefore, VelBeforeM1, DZCOM, DYCOM, DXCOM
    cdef double PenningTransferTime, TempTime, S2, RandomNum2, ARG1, D, U, ARGZ, TempSinZ, TempCosZ, TempPhi, TempSinPhi, TempCosPhi
    cdef double RandomSeed = 0.3, RandomNum, S1, EI, ESEC, EISTR, CosTheta, SinTheta, Phi, SinPhi, CosPhi, Sign, RandomNum1
    cdef double  EAuger,AIS,DirCosineZ2,DirCosineX2,DirCosineY2
    cdef double GasVelX, GasVelY, GasVelZ, VelocityRatio, VelXAfter, VelYAfter, VelZAfter, COMEnergy, Test1, A, VelocityInCOM, T2
    cdef int Flag=1, GasIndex, MaxBoltzNumsUsed, NumCollisions = 0, I, IPT, NCLTMP, IAuger, J, NAuger
    cdef int TempPlane, JPrint, J1 = 1,FFlag = 0
    Object.TimeSum = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.TotalSpaceZPrimary = 0.0
    Object.TotalSpaceZSecondary = 0.0
    Object.TotalTimePrimary = 0.0
    Object.TotalTimeSecondary = 0.0
    MV.StartingEnergy = 0.0

    MV.Sqrt2M = Object.CONST3 * 0.01  # This should be: sqrt(2m)
    MV.TwoM = pow(MV.Sqrt2M, 2)  # This should be: 2m
    MV.TwoPi = 2.0 * np.pi  # This should be: 2 Pi

    Object.reset()

    for J in range(9):
        Object.SXPlanes[J] = 0.0
        Object.SYPlanes[J] = 0.0
        Object.SZPlanes[J] = 0.0
        Object.STPlanes[J] = 0.0
        Object.SEPlanes[J] = 0.0
        Object.SX2Planes[J] = 0.0
        Object.SY2Planes[J] = 0.0
        Object.SZ2Planes[J] = 0.0
        Object.SVZPlanes[J] = 0.0
        Object.STSPlanes[J] = 0.0
        Object.STS2Planes[J] = 0.0
        Object.TMSPL[J] = 0.0
        Object.TTMSPL[J] = 0.0
        Object.RSPL[J] = 0.0
        Object.RRSPL[J] = 0.0
        Object.RRSPM[J] = 0.0
        Object.NESST[J] = 0.0

    Object.NESST[9] = 0.0
    MV.ID = 0
    MV.I100 = 0
    MV.NCLUS = 0
    MV.NPONT = 0
    MV.NMXADD = 0
    MV.NTPMFlag = 0
    MV.NumberOfElectronIon = 0
    MV.NumberOfElectron = 0
    MV.NumberOfCollision = 0
    MV.NumberOfNullCollision = 0
    Object.TotalSpaceZPrimary = 0
    Object.TotalTimePrimary = 0
    Object.TotalSpaceZSecondary = 0
    Object.TotalTimeSecondary = 0

    cdef int i = 0, K
    cdef double ** TotalCollFreqIncludingNull = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TotalCollFreqIncludingNull[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TotalCollFreqIncludingNull[K][J] = Object.TotalCollisionFrequency[K][J] + \
                                               Object.TotalCollisionFrequencyNull[K][J]
    AbsFakeIoniz = abs(Object.FakeIonisationsEstimate)

    for J in range(9):
        MV.FakeIonisationsTime[J] = 0.0
        MV.FakeIonisationsSpace[J] = 0.0

    # Generate initial random maxwell boltzman numbers
    GenerateMaxBoltz(Object.RandomSeed, Object.RandomMaxBoltzArray)
    MaxBoltzNumsUsed = 0

    # Initial direction cosines
    MV.DirCosineZ1 = cos(Object.AngleFromZ)
    MV.DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    MV.DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    MV.Energy100 = MV.StartingEnergy

    MV.DirCosineZ100 = MV.DirCosineZ1
    MV.DirCosineX100 = MV.DirCosineX1
    MV.DirCosineY100 = MV.DirCosineY1
    MV.NumberOfMaxColli = Object.MaxNumberOfCollisions
    # Here are some constants we will use
    MV.BP = pow(Object.EField, 2) * Object.CONST1  # This should be: 1/2 m e^2 EField^2
    MV.F1 = Object.EField * Object.CONST2
    MV.F2 = Object.EField * Object.CONST3  # This should be: sqrt( m / 2) e EField

    MV.PrintN = Object.MaxNumberOfCollisions / 10
    # Create primary electron
    NewPrimary(Object, &MV)
    # register this electron
    NewElectron(Object, &MV)
    while (1):
        # Sample random time to next collision. T is global total time.
        RandomNum = random_uniform(RandomSeed)
        # This is the formula from Skullerud
        MV.T = -log(RandomNum) / Object.MaxCollisionFreqTotal + MV.TDash
        MV.TOld = MV.TDash
        MV.TDash = MV.T
        MV.AP = MV.DirCosineZ1 * MV.F2 * sqrt(MV.StartingEnergy)
        Flag = TimeCalculations()
        while(Flag == 3):
            Flag = TimeCalculations()

        if Flag ==1:
            continue
        elif Flag ==0:
            break

        # similar to MONTET, calculate the energy of the electron after the collision
        MV.Energy = MV.StartingEnergy + (MV.AP + MV.BP * MV.T) * MV.T

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

        AIS = 1.0
        DirCosineZ2 = (MV.DirCosineZ1 * VelocityRatio + MV.T * MV.F2 / (2.0 * sqrt(MV.Energy)))
        DirCosineX2 =  MV.DirCosineX1 * VelocityRatio
        DirCosineY2 =  MV.DirCosineY1 * VelocityRatio
        if DirCosineZ2<0.0:
            AIS = -1.0
        DirCosineZ2 = AIS * sqrt(1.0 - (DirCosineX2**2) - (DirCosineY2**2))
        # Calculate electron velocity after
        VelXAfter = DirCosineX2 *  MV.Sqrt2M * sqrt(MV.Energy)
        VelYAfter = DirCosineY2 *  MV.Sqrt2M * sqrt(MV.Energy)
        VelZAfter = DirCosineZ2 * MV.Sqrt2M * sqrt(MV.Energy)

        # Calculate energy in center of mass frame
        #   E = 1/2 m dx^2 + dvy^2 + dvz^2
        #   works if TwoM = 2m
        COMEnergy = (pow((VelXAfter - GasVelX), 2) + pow((VelYAfter - GasVelY), 2) + pow((VelZAfter - GasVelZ),
                                                                                         2)) / MV.TwoM
         # Now the Skullerud null collision method
        RandomNum = random_uniform(RandomSeed)
        MV.iEnergyBin = <int> (COMEnergy / Object.ElectronEnergyStep)
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
                    MV.FakeIonisationsTime[MV.IPlane + 1] += 1
                    if Object.FakeIonisationsEstimate < 0.0:
                        MV.NumberOfElectronIon += 1

                        # Fake attachment start a new electron
                        if MV.NumberOfElectron == MV.NCLUS + 1:
                            # Create primary electron if the number of max cascaded electrons is reached
                            Flag = NewPrimary(Object, &MV)
                            NewElectron(Object, &MV)
                            if Flag == 0:
                                break
                            continue

                        # get a new electron from store
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

                        FFlag =1

                    # fake Ionisation add electron to the store (S)

                    MV.NCLUS += 1
                    MV.NPONT += 1
                    MV.NMXADD = max(MV.NMXADD, MV.NPONT)

                    if MV.NPONT > 2000:
                        raise ValueError("More than 2000 stored electrons")
                    A = MV.T * MV.Sqrt2M * sqrt(MV.StartingEnergy)
                    XS[MV.NPONT] = Object.X + MV.DirCosineX1 * A
                    YS[MV.NPONT] = Object.Y + MV.DirCosineY1 * A
                    ZS[MV.NPONT] = Object.Z + MV.DirCosineZ1 * A + MV.T * MV.T * MV.F1
                    TS[MV.NPONT] = Object.TimeSum + MV.T
                    ES[MV.NPONT] = MV.Energy
                    IPlaneS[MV.NPONT] = MV.IPlane
                    DirCosineX[MV.NPONT] = MV.DirCosineX1 * VelocityRatio
                    DirCosineY[MV.NPONT] = MV.DirCosineY1 * VelocityRatio
                    DirCosineZ[MV.NPONT] = (MV.DirCosineZ1 * VelocityRatio + MV.T * MV.F2 / (2.0 * sqrt(MV.Energy)))
                    continue
                continue
            continue






