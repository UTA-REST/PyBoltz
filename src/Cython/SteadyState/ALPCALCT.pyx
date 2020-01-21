from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
from libc.string cimport memset
import PulsedTownsend
import TimeOfFlight
from PyBoltz cimport drand48
from MBSorts cimport MBSort
import Monte
import PulsedTownsend
import TimeOfFlight
from Monte import *
from PulsedTownsend import *
from TimeOfFlight import *
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
cpdef run(PyBoltz   Object):
    Object.VelocityZ *= 1e5
    cdef double NetReducedRate, NetRate, TimeCutHigh, TimeCutLow, SpaceCutHighZ, SpaceCutLowZ, Alpha, NewAlpha
    cdef double StartingAlpha, ErrStartingAlpha, StartingNetAttachment, ErrStartingNetAttachment,StartingNetReducedRate
    # Ensure enough collisions are simulated
    if Object.MaxNumberOfCollisions < 5e7:
        Object.MaxNumberOfCollisions = 5e7

    # Recalculate the reduced ionisation and attachment rates
    Object.PresTempCor = 760.0 * (Object.TemperatureKelvin) / (293.15 * Object.PressureTorr)
    Object.ReducedIonization = Object.IonisationRate / Object.PresTempCor
    Object.ReducedAttachment = Object.AttachmentRate / Object.PresTempCor

    # Calculate the diffrence rates (reduced and non-reduced)
    NetReducedRate = Object.ReducedIonization - Object.ReducedAttachment
    NetRate = Object.IonisationRate - Object.AttachmentRate

    # Calculate Space and Time cut values (later used to limit the estimates of time and space steps)
    TimeCutHigh = 1.2e-10 * Object.PresTempCor
    TimeCutLow = 1e-13 * Object.PresTempCor
    SpaceCutHighZ = 1.2e-2 * Object.PresTempCor
    SpaceCutLowZ = 1e-5 * Object.PresTempCor

    # If the net reduced rate is greater than 30, then we have a weak attachment
    if NetReducedRate <= 30.0:
        if abs(NetReducedRate) < 100:
            # Some attachment
            Alpha = abs(Object.AttachmentRate) * 0.8
        elif abs(NetReducedRate) >= 100 and abs(NetReducedRate) < 1000:
            # Small net attachment
            Alpha = abs(NetReducedRate) * 0.65
        elif abs(NetReducedRate) >= 1000 and abs(NetReducedRate) < 10000:
            # Larger net attachment
            Alpha = abs(NetReducedRate) * 0.6
        elif abs(NetReducedRate) >= 10000 and abs(NetReducedRate) < 100000:
            Alpha = abs(NetReducedRate) * 0.5
        elif abs(NetReducedRate) >= 100000 and abs(NetReducedRate) < 2000000:
            Alpha = abs(NetReducedRate) * 0.2
        else:
            raise ValueError("Attachment is too large")

        # Calculate time and space step values
        Object.FakeIonisationsEstimate = Alpha * Object.VelocityZ * 1e-12
        NewAlpha = 0.85 * abs(Alpha + NetReducedRate)
        # Use the new alpha to estimate the time step
        Object.TimeStep = log(3) / (NewAlpha * Object.VelocityZ)

        # Limit the estimation to the cut values
        if Object.TimeStep > TimeCutHigh:
            Object.TimeStep = TimeCutHigh
        if Object.TimeStep < TimeCutLow:
            Object.TimeStep = TimeCutLow

        # Update null collision frequency limit
        for J in range(Object.NumberOfGases):
            Object.MaxCollisionFreq[J] = Object.MaxCollisionFreq[J] + abs(
                Object.FakeIonisationsEstimate) / Object.NumberOfGases

        # Print the alphas
        if Object.ConsoleOutputFlag:
            print("NewAlpha = ", NewAlpha, " NetReducedRate = ", NetReducedRate, "Alpha = ", Alpha, "TimeStep = ",
                  Object.TimeStep)
        # Convert to picoseconds
        Object.TimeStep *= 1e12
        Object.MaxTime = 7 * Object.TimeStep
        Object.NumberOfTimeSteps = 7

        # Calculate good starting values for ionisation and attachment rates
        Monte.MONTEFTT.run(Object, 0)
        PulsedTownsend.PT.PT(Object, 0)
        TimeOfFlight.TOF.TOF(Object, 0)

        StartingAlpha = (Object.RALPHA / Object.TOFWR) * 1e7
        ErrStartingAlpha = (Object.RALPER * StartingAlpha / 100)
        StartingNetAttachment = (Object.RATTOF / Object.TOFWR) * 1e7
        ErrStartingNetAttachment = Object.RATOFER * StartingNetAttachment / 100

        if Object.ConsoleOutputFlag:
            print("\nGood starting values for calculation:")
            print("Alpha = {} Error: {}".format(StartingAlpha, ErrStartingAlpha))
            print("NetAttachment = {} Error: {}".format(StartingNetAttachment, ErrStartingNetAttachment))

        for J in range(Object.NumberOfGases):
            Object.MaxCollisionFreq[J] = Object.MaxCollisionFreq[J] - abs(
                Object.FakeIonisationsEstimate) / Object.NumberOfGases

        # Calculate fake ionisation rate scaling by 1.2
        Alpha = StartingNetAttachment * 1.2

        if StartingAlpha - StartingNetAttachment > 30 * Object.PresTempCor: Alpha = abs(StartingNetAttachment * 0.4)

        if abs(StartingAlpha - StartingNetAttachment) < (StartingAlpha / 10) or abs(
                StartingAlpha - StartingNetAttachment) < (StartingNetAttachment / 10):
            Alpha = abs(StartingNetAttachment) * 0.3

        if (StartingAlpha - StartingNetAttachment) > 100 * Object.PresTempCor:
            Alpha = 0.0
        Object.VelocityZ = Object.TOFWR * 1e5
    else:
        Alpha = 0.0
        StartingAlpha = Object.IonisationRate
        StartingNetAttachment = Object.AttachmentRate

    Object.FakeIonisationsEstimate = Alpha * Object.VelocityZ * 1e-12
    NewAlpha = 0.85 * abs(Alpha + StartingAlpha - StartingNetAttachment)

    if StartingAlpha + Alpha > 10 * NewAlpha or StartingNetAttachment > (10 * NewAlpha):
        if StartingAlpha + Alpha > 100 * Object.PresTempCor:
            NewAlpha = NewAlpha * 15
        elif StartingAlpha + Alpha > 50 * Object.PresTempCor:
            NewAlpha = NewAlpha * 12.0
        else:
            NewAlpha = NewAlpha * 8.0

    Object.TimeStep = log(3) / (NewAlpha * Object.VelocityZ)
    Object.SpaceStepZ = log(3) / NewAlpha
    if Object.TimeStep > TimeCutHigh: Object.TimeStep = TimeCutHigh
    if Object.SpaceStepZ > SpaceCutHighZ: Object.SpaceStepZ = SpaceCutHighZ

    for J in range(Object.NumberOfGases):
        Object.MaxCollisionFreq[J] = Object.MaxCollisionFreq[J] - abs(
            Object.FakeIonisationsEstimate) / Object.NumberOfGases

    StartingNetReducedRate = StartingAlpha - StartingNetAttachment

            # Print the alphas
    if Object.ConsoleOutputFlag:
        print("NewAlpha = ", NewAlpha, " NetReducedRate = ", StartingNetReducedRate, "Alpha = ", Alpha, "TimeStep = ",
                  Object.TimeStep," SpaceStepZ = ", Object.SpaceStepZ)

    Object.TimeStep*= 1e12
    Object.SpaceStepZ *=0.01
    Object.MaxTime = 7* Object.TimeStep
    Object.NumberOfTimeSteps = 7
    Object.MaxSpaceZ = 8 *  Object.SpaceStepZ
    Object.NumberOfSpaceSteps = 8
    print("Solution for Steady State Townsend parameters")
    print("Space step between sampling planes = {} Microns".format(Object.SpaceStepZ*1e6))

    # call MONTEFDT
    # call SST
