from Boltz cimport Boltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan

cpdef run(Boltz Object, int ConsoleOuput):
    # Pulsed Townsend calculations
    cdef double NumberOfElecPlanes[9]
    # At plane 0 the # of Ionisations is equal to the number of primaries.
    NumberOfElecPlanes[0] = Object.NumberOfElectronsPlanes[1]
    Object.RealIonisation[0] = ((log(NumberOfElecPlanes[0])) - log(float(Object.IPrimary))) / Object.TimeStep
    Object.RealIonisation[0] -= Object.FakeIonisationsEstimate
    Object.EnergyPT[0] = Object.TEPlanes[1] / NumberOfElecPlanes[0]
    Object.AverageTimePT[0] = Object.TTPlanes[1] / NumberOfElecPlanes[0]
    Object.VelocityZPT[0] = 1e9 * Object.TVZPlanes[1] / NumberOfElecPlanes[0]


    for I in range(2,Object.NumberOfTimeSteps+1):
        if Object.NumberOfElectronsPlanes[I]==0:
            Object.NumberOfTimeSteps-=1
            break
        # For each other plane, the number of ionisations is equal to the number of electrons at each plane.
        NumberOfElecPlanes[I - 1] = Object.NumberOfElectronsPlanes[I]
        Object.RealIonisation[I - 1] = ((log(NumberOfElecPlanes[I - 1])) - log(float(NumberOfElecPlanes[I - 2]))) / Object.TimeStep
        Object.RealIonisation[I - 1] -= Object.FakeIonisationsEstimate
        Object.EnergyPT[I - 1] = Object.TEPlanes[I] / NumberOfElecPlanes[I - 1]
        Object.AverageTimePT[I - 1] = Object.TTPlanes[I] / NumberOfElecPlanes[I - 1]
        Object.VelocityZPT[I - 1] = 1e9 * Object.TVZPlanes[I] / NumberOfElecPlanes[I - 1]

    if ConsoleOuput:
        print("Pulsed Towensend results at " + str(Object.NumberOfTimeSteps) + " sequential time planes")
        print('{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}'.format("Plane #", "(Ion-Att) Freq", "Energy", "Velocity",
                                                                         "# of Elec."))
        for J in range(Object.NumberOfTimeSteps):
            print(
                '{:^15.5f}{:^15.5f}{:^15.5f}{:^15.5f}{:^15.5f}'.format(J + 1, Object.RealIonisation[J], Object.EnergyPT[J],
                                                                       Object.VelocityZPT[J], Object.NumberOfElectronsPlanes[J + 1]))