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
cdef void TCALCT(PyBoltz Object, MonteVars*MV):
    '''
    
    :param Object: 
    :param MV: 
    :return: 
    Calculate elapsed time, timestop1, until arrival at next plane, IPlane. 
    If two positive solutions exist, set ISolution to 2, and calculate the second solution, TimeStop2.   
    '''
    cdef double A, B, B2, C1 = 0, C2 = 0, FAC, TimeStop1, TimeStop2,Flag = 0
    MV.ISolution = 1
    A = Object.EField * Object.CONST2
    B = sqrt(MV.StartingEnergy) * Object.CONST3 * 0.01 * MV.DirCosineZ1
    B2 = B * B
    Flag = 0
    if Object.Z < MV.ZPlanes[1]:
        MV.IPlane = 1
        C1 = Object.Z - MV.ZPlanes[1]
        C2 = 0
    else:
        for J in range(2, Object.NumberOfSpaceSteps + 1):
            if Object.Z < MV.ZPlanes[J]:
                MV.IPlane = J
                C1 = Object.Z - MV.ZPlanes[J]
                C2 = Object.Z - MV.ZPlanes[J - 1]
                Flag = 1
                break
        if Flag==0:
            MV.IPlane = 9
            C1 = Object.Z - MV.ZPlanes[8] - 10 * Object.SpaceStepZ
            C2 = Object.Z - MV.ZPlanes[8]
    Flag = 0
    # check plane in drift direction( only one positive solution)
    FAC = B2 - 4 * A * C1

    #print(C1,C2,MV.IPlane,FAC,"HEREEEEEE")
    #sys.exit()
    if FAC < 0.0:
        # passed final plane (runaway electron)
        MV.TimeStop = -99
        return
    TimeStop1 = (-1 * B + sqrt(B2 - 4 * A * C1)) / (2 * A)
    TimeStop2 = (-1 * B - sqrt(B2 - 4 * A * C1)) / (2 * A)

    if TimeStop1 < TimeStop2:
        if TimeStop1 < 0.0:
            MV.TimeStop = TimeStop2
        else:
            MV.TimeStop = TimeStop1

        if MV.IPlane == 1:
            return
    else:
        if TimeStop2 >= 0.0:
            MV.TimeStop = TimeStop2
        else:
            MV.TimeStop = TimeStop1

        if MV.IPlane == 1:
            return
    # check plane in backward direction( only one positive solution)
    FAC = B2 - 4 * A * C2
    if FAC < 0.0:
        return

    TimeStop1 = (-1 * B + sqrt(FAC)) / (2 * A)
    TimeStop2 = (-1 * B - sqrt(FAC)) / (2 * A)
    if TimeStop1 < 0.0:
        return
    # Found backward solution
    MV.ISolution = 2
    MV.IPlane -= 1
    if TimeStop1 >= TimeStop2:
        MV.TimeStop = TimeStop2
        MV.TimeStop1 = TimeStop1
    else:
        MV.TimeStop = TimeStop1
        MV.TimeStop1 = TimeStop2

    return

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

    Object.TMSPL[MV.IPlane] += (Object.TimeSum + TimeLeft) * WGHT
    Object.TTMSPL[MV.IPlane] += (Object.TimeSum + TimeLeft) * (Object.TimeSum + TimeLeft) * WGHT

    Object.SX2Planes[MV.IPlane] += XPlane * XPlane * WGHT
    Object.SY2Planes[MV.IPlane] += YPlane * YPlane * WGHT
    Object.SZ2Planes[MV.IPlane] += ZPlane * ZPlane * WGHT
    Object.RRSPM[MV.IPlane] += RPlane * RPlane * WGHT

    Object.SEPlanes[MV.IPlane] += EPlane * WGHT
    Object.STPlanes[MV.IPlane] += WGHT / (Object.TimeSum + TimeLeft)

    Object.SVZPlanes[MV.IPlane] += VZPlane * WGHT

    Object.STSPlanes[MV.IPlane] += WGHT
    Object.STS2Planes[MV.IPlane] += WGHT * WGHT

cdef int TimeCalculations(PyBoltz Object, MonteVars*MV):
    cdef int Flag
    cdef double EPOT
    while (1):
        #TODO: change to NumberOfSpaceSteps-1 for anode termination
        if MV.IPlane >= Object.NumberOfSpaceSteps + 1 or Object.Z > Object.MaxSpaceZ or MV.TimeStop == -99 or MV.FFlag == 1:
            Object.TotalSpaceZPrimary += Object.Z
            Object.TotalTimePrimary += Object.TimeSum
            Object.TotalTimeSecondary += Object.TimeSum - MV.TimeSumStart
            Object.TotalSpaceZSecondary += Object.Z - MV.SpaceZStart
            if MV.NumberOfElectron == MV.NCLUS + 1 and MV.FFlag !=1:
                # Create primary electron
                Flag = NewPrimary(Object, MV)
                NewElectron(Object, MV)
                if Flag == 0:
                    return 0
                MV.FFFlag = 0
                return 1
            MV.FFlag = 0
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
            if Object.Z > Object.MaxSpaceZ:
                EPOT = Object.EField * (Object.Z - Object.MaxSpaceZ) * 100
                if MV.StartingEnergy < EPOT:
                    MV.NumberOfElectron += 1
                    MV.ISolution = 1
                    continue
            TCALCT(Object, MV)

            if MV.TimeStop == -99:
                MV.NumberOfElectron += 1
                MV.ISolution = 1
                continue
            NewElectron(Object, MV)
            MV.FFFlag = 0
            return 1
        if MV.ISolution == 2:
            MV.TimeStop = MV.TimeStop1
            MV.ISolution = 1
            if MV.T >= MV.TimeStop and MV.TOld < MV.TimeStop:
                SpacePlaneUpdate(Object, MV)
                continue
            else:
                return 2
        else:
            return 2

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object, int ConsoleOuput):
    cdef MonteVars MV
    cdef double VelBefore, VelXBefore, VelYBefore, VelZBefore, VelBeforeM1, DZCOM, DYCOM, DXCOM
    cdef double PenningTransferTime, TempTime, S2, RandomNum2, ARG1, D, U, ARGZ, TempSinZ, TempCosZ, TempPhi, TempSinPhi, TempCosPhi
    cdef double RandomSeed = 0.3, RandomNum, S1, EI, ESEC, EISTR, CosTheta, SinTheta, Phi, SinPhi, CosPhi, Sign, RandomNum1
    cdef double  EAuger, AIS, DirCosineZ2, DirCosineX2, DirCosineY2, EPOT
    cdef double GasVelX, GasVelY, GasVelZ, VelocityRatio, VelXAfter, VelYAfter, VelZAfter, COMEnergy, Test1, A, VelocityInCOM, T2
    cdef int Flag = 1, GasIndex, MaxBoltzNumsUsed, NumCollisions = 0, I, IPT, NCLTMP, IAuger, J, NAuger
    cdef int TempPlane, JPrint, J1 = 1, FFlag = 0, IDM1

    MV.FFlag = 0
    MV.FFFlag = 0
    Object.TimeSum = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.TotalSpaceZPrimary = 0.0
    Object.TotalSpaceZSecondary = 0.0
    Object.TotalTimePrimary = 0.0
    Object.TotalTimeSecondary = 0.0
    MV.StartingEnergy = Object.InitialElectronEnergy
    for J in range(Object.NumberOfSpaceSteps + 1):
        MV.ZPlanes[J] = Object.SpaceStepZ * J

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
    MV.NCLUS = 0
    MV.NumberOfNullCollision = 0
    Object.TotalSpaceZPrimary = 0
    Object.TotalTimePrimary = 0
    Object.TotalSpaceZSecondary = 0
    Object.TotalTimeSecondary = 0
    MV.TimeSumStart = 0
    MV.SpaceZStart = 0
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
    MV.NumberOfMaxColli = <int> Object.MaxNumberOfCollisions
    # Here are some constants we will use
    MV.BP = pow(Object.EField, 2) * Object.CONST1  # This should be: 1/2 m e^2 EField^2
    MV.F1 = Object.EField * Object.CONST2
    MV.F2 = Object.EField * Object.CONST3  # This should be: sqrt( m / 2) e EField

    MV.PrintN = <int> Object.MaxNumberOfCollisions / 10
    # Create primary electron
    NewPrimary(Object, &MV)
    # register this electron
    NewElectron(Object, &MV)
    while (1):
        if MV.FFFlag == 0:
            # Sample random time to next collision. T is global total time.
            RandomNum = random_uniform(RandomSeed)
            # This is the formula from Skullerud
            MV.T = -log(RandomNum) / Object.MaxCollisionFreqTotal + MV.TDash
            MV.TOld = MV.TDash
            MV.TDash = MV.T
            MV.AP = MV.DirCosineZ1 * MV.F2 * sqrt(MV.StartingEnergy)

            if MV.T >= MV.TimeStop and MV.TOld < MV.TimeStop:
                SpacePlaneUpdate(Object, &MV)
                Flag = TimeCalculations(Object, &MV)
                if Flag == 1:
                    continue
                elif Flag == 0:
                    break
        MV.FFFlag = 0
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
        VelocityRatio = sqrt(MV.StartingEnergy / MV.Energy)
        AIS = 1.0
        DirCosineZ2 = (MV.DirCosineZ1 * VelocityRatio + MV.T * MV.F2 / (2.0 * sqrt(MV.Energy)))
        DirCosineX2 = MV.DirCosineX1 * VelocityRatio
        DirCosineY2 = MV.DirCosineY1 * VelocityRatio
        if DirCosineZ2 < 0.0:
            AIS = -1.0
        DirCosineZ2 = AIS * sqrt(1.0 - (DirCosineX2 ** 2) - (DirCosineY2 ** 2))
        # Calculate electron velocity after
        VelXAfter = DirCosineX2 * MV.Sqrt2M * sqrt(MV.Energy)
        VelYAfter = DirCosineY2 * MV.Sqrt2M * sqrt(MV.Energy)
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
                    MV.FakeIonisationsSpace[MV.IPlane + 1] += 1
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
                        MV.FFlag = 1
                        Object.TotalSpaceZPrimary -= Object.Z
                        Object.TotalTimePrimary -= Object.TimeSum
                        Object.TotalTimeSecondary -= Object.TimeSum - MV.TimeSumStart
                        Object.TotalSpaceZSecondary -= Object.Z - MV.SpaceZStart
                        Flag = TimeCalculations(Object, &MV)
                        if Flag == 1:
                            continue
                        elif Flag == 0:
                            break
                        elif Flag == 2:
                            MV.FFFlag = 1
                            continue
                    # fake Ionisation add electron to the store (S)

                    MV.NCLUS += 1
                    MV.NPONT += 1
                    MV.NMXADD = max(MV.NMXADD, MV.NPONT)

                    if MV.NPONT > 2000:
                        raise ValueError("More than 2000 stored electrons")
                    A = MV.T * MV.Sqrt2M * sqrt(MV.StartingEnergy)
                    MV.XS[MV.NPONT] = Object.X + MV.DirCosineX1 * A
                    MV.YS[MV.NPONT] = Object.Y + MV.DirCosineY1 * A
                    MV.ZS[MV.NPONT] = Object.Z + MV.DirCosineZ1 * A + MV.T * MV.T * MV.F1
                    MV.TS[MV.NPONT] = Object.TimeSum + MV.T
                    MV.ES[MV.NPONT] = MV.Energy
                    AIS = 1.0
                    if DirCosineZ2 < 0.0:
                        AIS = -1.0
                    MV.DirCosineX[MV.NPONT] = DirCosineX2
                    MV.DirCosineY[MV.NPONT] = DirCosineY2
                    MV.DirCosineZ[MV.NPONT] = AIS * sqrt(1.0 - (DirCosineX2 ** 2) - (DirCosineY2 ** 2))
                    IDM1 = 1 + int(MV.ZS[MV.NPONT] / Object.SpaceStepZ)
                    if IDM1 < 1: IDM1 = 1
                    if IDM1 > 9: IDM1 = 9
                    MV.IPlaneS[MV.NPONT] = IDM1
                    Object.NESST[MV.IPlaneS[MV.NPONT]] += 1
                    continue
                continue
            continue

        # If we got this far, we have a collision.
        NumCollisions += 1

        #  sqrt(2m E_com) = |v_com|
        VelocityInCOM = (MV.Sqrt2M * sqrt(COMEnergy))

        # Calculate direction cosines of electron in 0 kelvin frame
        DXCOM = (VelXAfter - GasVelX) / VelocityInCOM
        DYCOM = (VelYAfter - GasVelY) / VelocityInCOM
        DZCOM = (VelZAfter - GasVelZ) / VelocityInCOM

        # Keep a running average of mean time between real collisions
        Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * MV.TDash

        # Reset time-to-next-real-collision clock
        MV.TDash = 0.0

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
        I = MBSortT(GasIndex, I, RandomNum, MV.iEnergyBin, Object)
        while Object.CollisionFrequency[GasIndex][MV.iEnergyBin][I] < RandomNum:
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
            Object.CollisionsPerGasPerType[GasIndex][<int> IPT - 1] += 1
            Object.ICOLN[GasIndex][I] += 1
            Object.CollisionTimes[min(299, int(MV.T + 1))] += 1

            Object.TotalSpaceZPrimary += Object.Z
            Object.TotalTimePrimary += Object.TimeSum
            Object.TotalSpaceZSecondary += Object.Z - MV.SpaceZStart
            Object.TotalTimeSecondary += Object.TimeSum - MV.TimeSumStart
            IDM1 = 1 + int(Object.Z / Object.SpaceStepZ)
            if IDM1 < 1: IDM1 = 1
            if IDM1 > 9: IDM1 = 9
            Object.NESST[IDM1] -= 1
            if MV.NumberOfElectron == MV.NCLUS + 1:
                # Electron captured start a new primary
                Flag = NewPrimary(Object, &MV)
                NewElectron(Object, &MV)
                if Flag == 0:
                    break
                continue
            MV.FFlag = 1
            Object.TotalSpaceZPrimary -= Object.Z
            Object.TotalTimePrimary -= Object.TimeSum
            Object.TotalTimeSecondary -= Object.TimeSum - MV.TimeSumStart
            Object.TotalSpaceZSecondary -= Object.Z - MV.SpaceZStart
            Flag = TimeCalculations(Object, &MV)
            if Flag == 1:
                continue
            elif Flag == 0:
                break
            elif Flag == 2:
                MV.FFFlag = 1
                continue
        elif Object.ElectronNumChange[GasIndex][I] != 0:
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
            MV.XS[MV.NPONT] = Object.X
            MV.YS[MV.NPONT] = Object.Y
            MV.ZS[MV.NPONT] = Object.Z
            MV.TS[MV.NPONT] = Object.TimeSum
            MV.ES[MV.NPONT] = ESEC
            MV.NTPMFlag = 1
            NCLTMP = MV.NPONT
            IDM1 = 1 + int(Object.Z / Object.SpaceStepZ)
            if IDM1 < 1: IDM1 = 1
            if IDM1 > 9: IDM1 = 9
            MV.IPlaneS[MV.NPONT] = IDM1
            Object.NESST[MV.IPlaneS[MV.NPONT]] += 1
            # Store possible shell emissions auger or fluorescence, update the angles and cosines
            if EISTR > 30.0:
                # Auger Emission without fluorescence
                NAuger = <int> Object.NC0[GasIndex][I]
                EAuger = Object.EC0[GasIndex][I] / NAuger
                for J in range(NAuger):
                    MV.NCLUS += 1
                    MV.NPONT += 1
                    MV.XS[MV.NPONT] = Object.X
                    MV.YS[MV.NPONT] = Object.Y
                    MV.ZS[MV.NPONT] = Object.Z
                    MV.TS[MV.NPONT] = Object.TimeSum
                    MV.ES[MV.NPONT] = EAuger
                    RandomNum = random_uniform(RandomSeed)

                    # Angular distribution (isotropic scattering)
                    CosTheta = 1.0 - 2 * RandomNum
                    Theta = acos(CosTheta)
                    CosTheta = cos(Theta)
                    SinTheta = sin(Theta)

                    RandomNum = random_uniform(RandomSeed)
                    Phi = MV.TwoPi * RandomNum
                    SinPhi = sin(Phi)
                    CosPhi = cos(Phi)
                    MV.DirCosineZ[MV.NPONT] = CosTheta
                    MV.DirCosineX[MV.NPONT] = CosPhi * SinTheta
                    MV.DirCosineY[MV.NPONT] = SinPhi * SinTheta
                    IDM1 = 1 + int(MV.ZS[MV.NPONT] / Object.SpaceStepZ)
                    if IDM1 < 1: IDM1 = 1
                    if IDM1 > 9: IDM1 = 9
                    MV.IPlaneS[MV.NPONT] = IDM1
                    Object.NESST[MV.IPlaneS[MV.NPONT]] += 1
        # Generate scattering angles and update laboratory cosines after collision also update energy of electron
        IPT = <long long> Object.InteractionType[GasIndex][I]
        MV.ID += 1
        MV.Iterator += 1
        MV.IPrint += 1
        Object.CollisionsPerGasPerType[GasIndex][<int> IPT - 1] += 1
        Object.ICOLN[GasIndex][I] += 1

        # If it is an excitation then add the probability
        # of transfer to give ionisation of the other gases in the mixture

        if Object.EnablePenning != 0:
            if Object.PenningFraction[GasIndex][0][I] != 0:
                RandomNum = random_uniform(RandomSeed)
                if RandomNum <= Object.PenningFraction[GasIndex][0][I]:
                    MV.NCLUS += 1
                    MV.NPONT += 1
                    if MV.NPONT > 2000:
                        raise ValueError("More than 2000 stored electrons")
                    # Possible delocalisation length for penning transfer
                    if Object.PenningFraction[GasIndex][1][I] == 0.0:
                        MV.XS[MV.NPONT] = Object.X
                        MV.YS[MV.NPONT] = Object.Y
                        MV.ZS[MV.NPONT] = Object.Z
                    else:
                        Sign = 1.0
                        RandomNum = random_uniform(RandomSeed)
                        if RandomNum < 0.5:
                            Sign = -1 * Sign
                        RandomNum = random_uniform(RandomSeed)
                        MV.XS[MV.NPONT] = Object.X - log(RandomNum) * Object.PenningFraction[GasIndex][1][I] * Sign

                        RandomNum = random_uniform(RandomSeed)
                        if RandomNum < 0.5:
                            Sign = -1 * Sign
                        RandomNum = random_uniform(RandomSeed)
                        MV.YS[MV.NPONT] = Object.Y - log(RandomNum) * Object.PenningFraction[GasIndex][1][I] * Sign

                        RandomNum = random_uniform(RandomSeed)
                        if RandomNum < 0.5:
                            Sign = -1 * Sign
                        RandomNum = random_uniform(RandomSeed)
                        MV.ZS[MV.NPONT] = Object.Z - log(RandomNum) * Object.PenningFraction[GasIndex][1][I] * Sign
                    if MV.ZS[MV.NPONT] < 0.0 or MV.ZS[MV.NPONT] > Object.MaxSpaceZ:
                        # Penning happens after final time plane, dont store the electron
                        MV.NPONT -= 1
                        MV.NCLUS -= 1
                    else:
                        # Possible penning transfer time
                        PenningTransferTime = Object.TimeSum
                        if Object.PenningFraction[GasIndex][2][I] != 0.0:
                            RandomNum = random_uniform(RandomSeed)
                            PenningTransferTime = Object.TimeSum - log(RandomNum) * Object.PenningFraction[GasIndex][2][I]
                        MV.TS[MV.NPONT] = PenningTransferTime
                        MV.ES[MV.NPONT] = 1.0
                        MV.DirCosineX[MV.NPONT] = MV.DirCosineX1
                        MV.DirCosineY[MV.NPONT] = MV.DirCosineY1
                        MV.DirCosineZ[MV.NPONT] = MV.DirCosineZ1
                        IDM1 = 1 + int(MV.ZS[MV.NPONT] / Object.SpaceStepZ)
                        if IDM1 < 1: IDM1 = 1
                        if IDM1 > 9: IDM1 = 9
                        MV.IPlaneS[MV.NPONT] = IDM1
                        Object.NESST[MV.IPlaneS[MV.NPONT]] += 1

        S2 = (S1 * S1) / (S1 - 1.0)

        # Anisotropic scattering - pick the scattering angle theta depending on scatter type
        RandomNum = random_uniform(RandomSeed)
        if Object.AngularModel[GasIndex][I] == 1:
            # Use method of Capitelli et al
            RandomNum2 = random_uniform(RandomSeed)
            CosTheta = 1.0 - RandomNum * Object.AngleCut[GasIndex][MV.iEnergyBin][I]
            if RandomNum2 > Object.ScatteringParameter[GasIndex][MV.iEnergyBin][I]:
                CosTheta = -1.0 * CosTheta
        elif Object.AngularModel[GasIndex][I] == 2:
            # Use method of Okhrimovskyy et al
            EpsilonOkhr = Object.ScatteringParameter[GasIndex][MV.iEnergyBin][I]
            CosTheta = 1.0 - (2.0 * RandomNum * (1.0 - EpsilonOkhr) / (1.0 + EpsilonOkhr * (1.0 - 2.0 * RandomNum)))
        else:
            # Isotropic scattering
            CosTheta = 1.0 - 2.0 * RandomNum

        Theta = acos(CosTheta)

        # Pick a random Phi - must be uniform by symmetry of the gas
        RandomNum = random_uniform(RandomSeed)
        Phi = MV.TwoPi * RandomNum
        SinPhi = sin(Phi)
        CosPhi = cos(Phi)

        if COMEnergy < EI:
            EI = 0.0

        ARG1 = max(1.0 - S1 * EI / COMEnergy, Object.SmallNumber)

        D = 1.0 - CosTheta * sqrt(ARG1)
        U = (S1 - 1) * (S1 - 1) / ARG1

        # Update the energy to start drifing for the next round.
        #  If its zero, make it small but nonzero.
        MV.StartingEnergy = max(COMEnergy * (1.0 - EI / (S1 * COMEnergy) - 2.0 * D / S2), Object.SmallNumber)

        Q = min(sqrt((COMEnergy / MV.StartingEnergy) * ARG1) / S1, 1.0)

        # Calculate angle of scattering from Z direction
        Object.AngleFromZ = asin(Q * sin(Theta))
        CosZAngle = cos(Object.AngleFromZ)

        # Find new directons after scatter
        if CosTheta < 0 and CosTheta * CosTheta > U:
            CosZAngle = -1 * CosZAngle
        SinZAngle = sin(Object.AngleFromZ)
        DZCOM = min(DZCOM, 1.0)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)

        if ARGZ == 0.0:
            # If scattering frame is same as lab frame, do this;
            MV.DirCosineZ1 = CosZAngle
            MV.DirCosineX1 = CosPhi * SinZAngle
            MV.DirCosineY1 = SinPhi * SinZAngle
            if MV.NTPMFlag == 1:
                # Use free kinematics for ionisation secondary angle
                TempSinZ = SinZAngle * sqrt(MV.StartingEnergy / MV.ES[NCLTMP])
                if TempSinZ > 1:
                    TempSinZ = 1.0

                TempSinZ = sin(asin(TempSinZ))
                TempCosZ = cos(asin(TempSinZ))

                if TempCosZ < 0.0:
                    TempCosZ = -1 * TempCosZ

                TempPhi = Phi + np.pi

                if TempPhi > MV.TwoPi:
                    TempPhi = Phi - MV.TwoPi

                TempSinPhi = sin(TempPhi)
                TempCosPhi = cos(TempPhi)

                MV.DirCosineZ[NCLTMP] = TempCosZ
                MV.DirCosineX[NCLTMP] = TempCosPhi * TempSinZ
                MV.DirCosineY[NCLTMP] = TempSinPhi * TempSinZ
                MV.NTPMFlag = 0
        else:
            # Otherwise do this.
            MV.DirCosineZ1 = DZCOM * CosZAngle + ARGZ * SinZAngle * SinPhi
            MV.DirCosineY1 = DYCOM * CosZAngle + (SinZAngle / ARGZ) * (DXCOM * CosPhi - DYCOM * DZCOM * SinPhi)
            MV.DirCosineX1 = DXCOM * CosZAngle - (SinZAngle / ARGZ) * (DYCOM * CosPhi + DXCOM * DZCOM * SinPhi)
            if MV.NTPMFlag == 1:
                # Use free kinematics for ionisation secondary angle
                TempSinZ = SinZAngle * sqrt(MV.StartingEnergy / MV.ES[NCLTMP])
                if TempSinZ > 1:
                    TempSinZ = 1.0

                TempSinZ = sin(asin(TempSinZ))
                TempCosZ = cos(asin(TempSinZ))

                if TempCosZ < 0.0:
                    TempCosZ = -1 * TempCosZ

                TempPhi = Phi + np.pi

                if TempPhi > MV.TwoPi:
                    TempPhi = Phi - MV.TwoPi

                TempSinPhi = sin(TempPhi)
                TempCosPhi = cos(TempPhi)

                MV.DirCosineZ[NCLTMP] = DZCOM * TempCosZ + ARGZ * TempSinZ * TempSinPhi

                MV.DirCosineX[NCLTMP] = DXCOM * TempCosZ - (TempSinZ / ARGZ) * (
                        DYCOM * TempCosPhi + DXCOM * DZCOM * TempSinPhi)

                MV.DirCosineY[NCLTMP] = DYCOM * TempCosZ + (TempSinZ / ARGZ) * (
                        DXCOM * TempCosPhi - DYCOM * DZCOM * TempSinPhi)
                MV.NTPMFlag = 0

        # Transform velocity vectors to lab frame
        VelBefore = MV.Sqrt2M * sqrt(MV.StartingEnergy)
        VelXBefore = MV.DirCosineX1 * VelBefore + GasVelX
        VelYBefore = MV.DirCosineY1 * VelBefore + GasVelY
        VelZBefore = MV.DirCosineZ1 * VelBefore + GasVelZ

        # Calculate energy and direction cosines in lab frame
        MV.StartingEnergy = (VelXBefore * VelXBefore + VelYBefore * VelYBefore + VelZBefore * VelZBefore) / MV.TwoM
        VelBeforeM1 = 1 / (MV.Sqrt2M * sqrt(MV.StartingEnergy))
        MV.DirCosineX1 = VelXBefore * VelBeforeM1
        MV.DirCosineY1 = VelYBefore * VelBeforeM1
        MV.DirCosineZ1 = VelZBefore * VelBeforeM1
        #And go around again to the next collision!
        if MV.NumberOfElectron == 6828:
            print(MV.NumberOfElectron)
            print (MV.T, MV.TimeStop, MV.StartingEnergy,MV.DirCosineZ1)
            sys.exit()
        MV.I100 += 1
        if MV.I100 == 200:
            MV.DirCosineZ100 = MV.DirCosineZ1
            MV.DirCosineX100 = MV.DirCosineX1
            MV.DirCosineY100 = MV.DirCosineY1
            MV.Energy100 = MV.StartingEnergy
            MV.I100 = 0

        if Object.Z > Object.MaxSpaceZ:
            EPOT = Object.EField * (Object.Z - Object.MaxSpaceZ) * 100
            if MV.StartingEnergy < EPOT:
                Flag = TimeCalculations(Object, &MV)
                if Flag == 1:
                    MV.FFFlag = 0
                    continue
                elif Flag == 0:
                    break
                elif Flag == 2:
                    MV.FFFlag = 1
                    continue
        TCALCT(Object, &MV)
        if MV.TimeStop == -99:
            Flag = TimeCalculations(Object, &MV)
            if Flag == 1:
                MV.FFFlag = 0
                continue
            elif Flag == 0:
                break
            elif Flag == 2:
                MV.FFFlag = 1
                continue

    # ATTOINT,ATTERT,AIOERT
    if MV.NumberOfElectron > Object.IPrimary:
        Object.ATTOION = MV.NumberOfElectronIon / (MV.NumberOfElectron - Object.IPrimary)
        Object.ATTATER = sqrt(MV.NumberOfElectronIon) / MV.NumberOfElectronIon
        Object.ATTIOER = sqrt(MV.NumberOfElectron - Object.IPrimary) / (MV.NumberOfElectron - Object.IPrimary)
    else:
        Object.ATTOIN = -1
        Object.ATTATER = sqrt(MV.NumberOfElectronIon) / MV.NumberOfElectronIon

    if ConsoleOuput:
        print(
            '\nSimulation through {} Space planes:\n Total number of Electrons: {:10.1f}\n Number of Negative Ions: {:10.1f}\n Number of primaries: {:10.1f}\n'.format(
                Object.NumberOfSpaceSteps,
                MV.NumberOfElectron,
                MV.NumberOfElectronIon, Object.IPrimary))

    #TODO: EPRM

    for i in range(6):
        free(TotalCollFreqIncludingNull[i])
    free(TotalCollFreqIncludingNull)
