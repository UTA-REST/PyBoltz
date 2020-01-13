from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow
from libc.string cimport memset
import PulsedTownsend
import TimeOfFlight
from PyBoltz cimport drand48
from MBSorts cimport MBSort
import Monte
import PulsedTownsend
import TimeOfFlight
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
cpdef run(PyBoltz Object):
    cdef double NetReducedRate, NetRate, TimeCutHigh,TimeCutLow,SpaceCutHighZ,SpaceCutLowZ,Alpha,NewAlpha


    # Ensure enough collisions are simulated
    if Object.MaxNumberOfCollisions< 5e7:
        Object.MaxNumberOfCollisions = 5e7

    # Recalculate the reduced ionisation and attachment rates
    Object.PresTempCor = 760.0 *(Object.TemperatureKelvin)/(293.15 * Object.PressureTorr)
    Object.ReducedIonization = Object.IonisationRate / Object.PresTempCor
    Object.ReducedAttachment = Object.AttachmentRate / Object.PresTempCor

    # Calculate the diffrence rates (reduced and non-reduced)
    NetReducedRate =  Object.ReducedIonization - Object.ReducedAttachment
    NetRate = Object.IonisationRate - Object.AttachmentRate

    # Calculate Space and Time cut values (later used to limit the estimates of time and space steps)
    TimeCutHigh = 1.2e-10 * Object.PresTempCor
    TimeCutLow = 1e-13*Object.PresTempCor
    SpaceCutHighZ = 1.2e-2 * Object.PresTempCor
    SpaceCutLowZ = 1e-5 *Object.PresTempCor

    # If the net reduced rate is greater than 30, then we have a weak attachment
    if NetReducedRate<=30.0:
        if abs(NetReducedRate)<100:
            # Some attachment
            Alpha = abs(Object.AttachmentRate) * 0.8
        elif abs(NetReducedRate)>=100 and abs(NetReducedRate)<1000:
            # Small net attachment
            Alpha = abs(NetReducedRate)*0.65
        elif abs(NetReducedRate)>=1000 and abs(NetReducedRate)<10000:
            # Larger net attachment
            Alpha = abs(NetReducedRate)*0.6
        elif abs(NetReducedRate)>=10000 and abs(NetReducedRate)<100000:
            Alpha = abs(NetReducedRate)*0.5
        elif abs(NetReducedRate)>=100000 and abs(NetReducedRate)<1000000:
            Alpha = abs(NetReducedRate)*0.2
        else:
            raise ValueError("Attachment is too large")

        # Calculate time and space step values
        Object.FakeIonisationsEstimate = Alpha * Object.VelocityZ * 1e-12
        NewAlpha = 0.85 * abs(Alpha+NetReducedRate)

        # Use the new alpha to estimate the time step
        Object.TimeStep = log(3)/(NewAlpha*Object.VelocityZ*1e5)

        # Limit the estimation to the cut values
        if Object.TimeStep > TimeCutHigh:
            Object.TimeStep = TimeCutHigh
        if Object.TimeStep <TimeCutLow:
            Object.TimeStep = TimeCutLow

        # Update null collision frequency limit
        for J in range(Object.NumberOfGases):
            Object.MaxCollisionFreq[J] = Object.MaxCollisionFreq[J]+ abs(Object.FakeIonisationsEstimate)/Object.NumberOfGases

        # Convert to picoseconds
        Object.TimeStep *= 1e12
        Object.MaxTime = 7 * Object.TimeStep
        Object.NumberOfTimeSteps = 7

        # Calculate good starting values for ionisation and attachment rates
        Monte.MONTEFTT.run(Object,0)
        PulsedTownsend.PT.PT(Object,0)
        TimeOfFlight.TOF.TOF(Object,0)

