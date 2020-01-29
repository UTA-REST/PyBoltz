from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan

cpdef run(PyBoltz Object, int ConsoleOuput):
    cdef double ANTPL[9]
    ANTPL[0] = Object.NETPL[1]
    Object.RI[0] = ((log(ANTPL[0]))-log(float(Object.IPrimary)))/ Object.TimeStep
    Object.RI[0] -= Object.FakeIonisationsEstimate
    Object.EPT[0] = Object.TEPlanes[1]/ANTPL[0]
    Object.TTEST[0] = Object.TTPlanes[1]/ANTPL[0]
    Object.VZPT[0] = 1e9*Object.TVZPlanes[1]/ANTPL[0]
    for I in range(2,Object.NumberOfTimeSteps+1):
        if Object.NETPL[I]==0:
            Object.NumberOfTimeSteps-=1
            break

        ANTPL[I-1] = Object.NETPL[I]
        Object.RI[I-1] = ((log(ANTPL[I-1]))-log(float(ANTPL[I-2])))/ Object.TimeStep
        Object.RI[I-1] -= Object.FakeIonisationsEstimate
        Object.EPT[I-1] = Object.TEPlanes[I]/ANTPL[I-1]
        Object.TTEST[I-1] = Object.TTPlanes[I]/ANTPL[I-1]
        Object.VZPT[I-1] = 1e9*Object.TVZPlanes[I]/ANTPL[I-1]

    if ConsoleOuput:
        print("Pulsed Towensend results at " + str(Object.NumberOfTimeSteps) + " sequential time planes")
        print('{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}'.format("Plane #", "(Ion-Att) Freq", "Energy", "Velocity",
                                                                         "# of Elec."))
        for J in range(Object.NumberOfTimeSteps):
            print(
                '{:^15.5f}{:^15.5f}{:^15.5f}{:^15.5f}{:^15.5f}'.format(J+1, Object.RI[J], Object.EPT[J],
                                                                                         Object.VZPT[J], Object.NETPL[J+1]))