from libcpp.stack cimport stack
from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan
from libc.string cimport memset
from PyBoltz cimport drand48
from MBSorts cimport MBSortT
from libc.stdlib cimport malloc, free
from MonteVars cimport MonteVars
import cython
import numpy as np
from Electron cimport Electron
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


cdef void NewPrimary(PyBoltz Object, MonteVars*MV, stack[Electron]* ElectronStore):
    Object.IPrimary += 1
    cdef Electron tempElectron
    if Object.IPrimary > 1 and Object.IPrimary < 10000000:
        if MV.Iterator > MV.NumberOfMaxColli:
            Object.IPrimary -= 1
        else:
            tempElectron.X = 0.0
            tempElectron.Y = 0.0
            tempElectron.Z = 0.0
            tempElectron.DirCosineX = MV.DirCosineX100
            tempElectron.DirCosineY = MV.DirCosineY100
            tempElectron.DirCosineZ = MV.DirCosineZ100
            tempElectron.Energy = MV.Energy100
            MV.NCLUS += 1
            tempElectron.TimeSum = 0.0
            MV.TimeSumStart = 0.0
            MV.SpaceZStart = 0.0
            tempElectron.IPlane = 0
            ElectronStore.push(tempElectron)
    if Object.IPrimary > 10000000:
        print("Too many primaries program stopped!")


cdef void NewElectron(PyBoltz Object, MonteVars*MV,stack[Electron]* ElectronStore):
    MV.TDash = 0.0
    MV.NumberOfElectron += 1


cdef void SpacePlaneUpdate(PyBoltz Object, MonteVars*MV, Electron* electron):
    cdef double CurrentTime, TimeLeft, A, B, EPlane, VelocityRatio, TimeLeft2, DirCosineZ2
    cdef double XPlane, YPlane, ZPlane, VZPlane, WGHT, RPlane
    if electron.IPlane > 8:
        return
    TimeLeft = MV.TimeStop
    TimeLeft2 = TimeLeft * TimeLeft
    A = MV.AP * TimeLeft
    B = MV.BP * TimeLeft2
    EPlane = electron.Energy + A + B
    VelocityRatio = sqrt(electron.Energy / EPlane)
    DirCosineZ2 = (electron.DirCosineZ * VelocityRatio + TimeLeft * MV.F2 / (2.0 * sqrt(EPlane)))

    XPlane = Object.X + electron.DirCosineX * TimeLeft * sqrt(electron.StartingEnergy) * Object.CONST3 * 0.01
    YPlane = Object.Y + electron.DirCosineY * TimeLeft * sqrt(electron.StartingEnergy) * Object.CONST3 * 0.01
    ZPlane = Object.Z + electron.DirCosineZ * TimeLeft * sqrt(
        electron.Energy) * Object.CONST3 * 0.01 + TimeLeft2 * Object.EField * Object.CONST2

    VZPlane = DirCosineZ2 * sqrt(EPlane) * Object.CONST3 * 0.01

    WGHT = abs(1 / VZPlane)
    RPlane = sqrt(XPlane ** 2 + YPlane ** 2)

    Object.SXPlanes[electron.IPlane] += XPlane * WGHT
    Object.SYPlanes[electron.IPlane] += YPlane * WGHT
    Object.SZPlanes[electron.IPlane] += ZPlane * WGHT
    Object.RSPL[electron.IPlane] += RPlane * WGHT

    Object.TMSPL[electron.IPlane]+=(Object.TimeSum+TimeLeft)*WGHT
    Object.TTMSPL[electron.IPlane]+=    (Object.TimeSum+TimeLeft)*(Object.TimeSum+TimeLeft)*WGHT

    Object.SX2Planes[electron.IPlane]+= XPlane*XPlane*WGHT
    Object.SY2Planes[electron.IPlane]+= YPlane*YPlane*WGHT
    Object.SZ2Planes[electron.IPlane]+= ZPlane*ZPlane*WGHT
    Object.RRSPM[electron.IPlane]+= RPlane*RPlane*WGHT

    Object.SEPlanes[electron.IPlane]+= EPlane*WGHT
    Object.STPlanes[electron.IPlane]+= WGHT/(Object.TimeSum+TimeLeft)

    Object.SVZPlanes[electron.IPlane]+= VZPlane*WGHT

    Object.STSPlanes[electron.IPlane] +=WGHT
    Object.STS2Planes[electron.IPlane]+=WGHT*WGHT

cdef int TimeCalculations(PyBoltz Object, MonteVars*MV,stack[Electron]* ElectronStore,Electron* electron ):
     while (MV.T >= MV.TimeStop and MV.TOld < MV.TimeStop):
            SpacePlaneUpdate(Object,&MV,Electron)
            #TODO: change to NumberOfSpaceSteps-1 for anode termination
            if electron.IPlane>= Object.NumberOfSpaceSteps+1:
                Object.TotalSpaceZPrimary+=Object.Z
                Object.TotalTimePrimary +=Object.TimeSum
                Object.TotalTimeSecondary+=Object.TimeSum - MV.TimeSumStart
                Object.TotalSpaceZSecondary+=Object.Z - MV.SpaceZStart
                if ElectronStore.size()==0:
                    # Create primary electron
                    NewPrimary(Object, &MV, &ElectronStore)
                    # register this electron
                    NewElectron(Object, &MV,&ElectronStore)
                    return 0

                # Take an electron from the store




cpdef run(PyBoltz Object, int ConsoleOuput):
    cdef MonteVars MV
    cdef stack[Electron] ElectronStore
    cdef Electron CurrentElectron
    cdef double VelBefore, VelXBefore, VelYBefore, VelZBefore, VelBeforeM1, DZCOM, DYCOM, DXCOM
    cdef double PenningTransferTime, TempTime, S2, RandomNum2, ARG1, D, U, ARGZ, TempSinZ, TempCosZ, TempPhi, TempSinPhi, TempCosPhi
    cdef double RandomSeed = 0.3, RandomNum, S1, EI, ESEC, EISTR, CosTheta, SinTheta, Phi, SinPhi, CosPhi, Sign, RandomNum1
    cdef double XS[2001], YS[2001], ZS[2001], TS[2001], ES[2001], DirCosineX[2001], DirCosineY[2001], DirCosineZ[2001], EAuger
    cdef double GasVelX, GasVelY, GasVelZ, VelocityRatio, VelXAfter, VelYAfter, VelZAfter, COMEnergy, Test1, A, VelocityInCOM, T2
    cdef int IPlaneS[2001], Flag=1, GasIndex, MaxBoltzNumsUsed, NumCollisions = 0, I, IPT, NCLTMP, IAuger, J, NAuger
    cdef int TempPlane, JPrint, J1 = 1,ISolution,FFlag = 0
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
    NewPrimary(Object, &MV, &ElectronStore)
    # register this electron
    NewElectron(Object, &MV,&ElectronStore)


    while(not(ElectronStore.empty())):
        CurrentElectron = ElectronStore.top()
        ElectronStore.pop()
        # Sample random time to next collision. T is global total time.
        RandomNum = random_uniform(RandomSeed)
        # This is the formula from Skullerud
        MV.T = -log(RandomNum) / Object.MaxCollisionFreqTotal + MV.TDash
        MV.TOld = MV.TDash
        MV.TDash = MV.T
        MV.AP = CurrentElectron.DirCosineZ * MV.F2 * sqrt(CurrentElectron.Energy)
        Flag = TimeCalculations(Object,&MV, &ElectronStore,&CurrentElectron)

        if Flag ==0:
            # New primary move on
            continue
        elif Flag ==1:


