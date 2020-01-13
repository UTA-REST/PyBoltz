from COLF cimport COLF
from COLFT cimport COLFT
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan
from PyBoltz cimport PyBoltz

cdef void TOF(PyBoltz Object, int ConsoleOuput):
    cdef double FREQ, FREEL, FRION, FRATT, FREIN, NTOTAL, CORERR, ANTPL[8], WR[8],ANST[8]
    cdef double DLTF[8], DXTF[8], DYTF[8],DT2,DT3,ATER
    cdef int J,I1,I2

    # Get ionisation and attachment frequencies
    if Object.EnableThermalMotion:
        COLFT(&FREQ, &FREEL, &FRION, &FRATT, &FREIN, &NTOTAL, Object)
    else:
        COLF(&FREQ, &FREEL, &FRION, &FRATT, &FREIN, &NTOTAL, Object)

    Object.ATTOINT = FRATT / FRION
    CORERR = abs((Object.FakeIonisationsEstimate + FRION - FRATT) / (FRION - FRATT))
    ANTPL[0] = Object.NETPL[1]
    WR[0] = Object.TZPlanes[1] / (ANTPL[0] * Object.TimeStep)
    DLTF[0] = ((Object.TZ2Planes[1] / ANTPL[0]) - (Object.TZPlanes[1] / ANTPL[0]) ** 2) / (2 * Object.TimeStep)
    DXTF[0] = ((Object.TX2Planes[1] / ANTPL[0]) - (Object.TXPlanes[1] / ANTPL[0]) ** 2) / (2 * Object.TimeStep)
    DYTF[0] = ((Object.TY2Planes[1] / ANTPL[0]) - (Object.TYPlanes[1] / ANTPL[0]) ** 2) / (2 * Object.TimeStep)

    for J in range(2, Object.NumberOfTimeSteps + 1):
        ANTPL[J - 1] = Object.NETPL[J]
        WR[J - 1] = ((Object.TZPlanes[J] / ANTPL[J - 1]) - (Object.TZPlanes[J - 1] / ANTPL[J - 2])) / (Object.TimeStep)

        DLTF[J - 1] = ((Object.TZ2Planes[J] / ANTPL[J - 1]) - (Object.TZPlanes[J] / ANTPL[J - 1]) ** 2 - (
                    Object.TZ2Planes[J - 1] / ANTPL[J - 2]) + (Object.TZPlanes[J - 1] / ANTPL[J - 2]) ** 2) / (
                                  2 * Object.TimeStep)

        DXTF[J - 1] = ((Object.TX2Planes[J] / ANTPL[J - 1]) - (Object.TXPlanes[J] / ANTPL[J - 1]) ** 2 - (
                    Object.TX2Planes[J - 1] / ANTPL[J - 2]) + (Object.TXPlanes[J - 1] / ANTPL[J - 2]) ** 2) / (
                                  2 * Object.TimeStep)

        DYTF[J - 1] = ((Object.TY2Planes[J] / ANTPL[J - 1]) - (Object.TYPlanes[J] / ANTPL[J - 1]) ** 2 - (
                    Object.TY2Planes[J - 1] / ANTPL[J - 2]) + (Object.TYPlanes[J - 1] / ANTPL[J - 2]) ** 2) / (
                                  2 * Object.TimeStep)

    for J in range(Object.NumberOfTimeSteps):
        WR[J]*=1e9
        DLTF[J] *=1e16
        DXTF[J] *=1e16
        DYTF[J] *=1e16

    if ConsoleOuput:
        print("Time of flight results at" + str(Object.NumberOfTimeSteps) + "sequential time planes")
        print('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Plane #", "DL", "DX", "DY",
                                                                         "WR"))
        for J in range(Object.NumberOfTimeSteps):
            print(
                '{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(J+1, DLTF[J], DXTF[J],
                                                                                         DYTF[J],WR[J]))

    if Object.NETPL[1] > Object.NETPL[Object.NumberOfTimeSteps]:
        # For net attachment the results are taken from plane 2
        Object.TOFENE = Object.EPT[1]
        Object.TOFENER = 100 * abs((Object.EPT[1]-Object.EPT[2])/(2*Object.EPT[1]))

        Object.TOFWV = Object.VZPT[1]
        Object.TOFWVER = 100 * abs((Object.VZPT[1]-Object.VZPT[2])/(2*Object.VZPT[1]))

        Object.TOFDL = DLTF[1]
        Object.TOFDLER = 100 * abs((DLTF[1]-DLTF[2])/(2*DLTF[1]))

        DT2 = (DXTF[1]+DYTF[1])/2
        DT3 = (DXTF[2]+DYTF[2])/2

        Object.TOFDT = DT2
        Object.TOFDTER = 100 * abs((DT2-DT3)/(2*DT2))

        Object.TOFWR = WR[1]
        Object.TOFWRER = 100 * abs((WR[1]-WR[2])/(2*WR[1]))

        ANST[1] = Object.NETPL[2]
        ANST[2] = Object.NETPL[3]
        ANST[3] = ANST[2] - sqrt(ANST[1]/ANST[2])
        ANST[4] = log(ANST[1]/ANST[2])
        ANST[5] = log(ANST[1]/ANST[3])
        ANST[6] = ANST[5]/ANST[4]
        ANST[7] = ANST[6]-1
        if Object.ATTOINT==-1:
            Object.RALPHA = 0.0
            Object.RALPER = 0.0
            Object.RATTOF = -1 * Object.RI[1]
            Object.RATTOFER = 100 *sqrt(ANST[7]**2+Object.ATTERT**2)
        else:
            Object.RALPHA = Object.RI[1]/(1-Object.ATTOINT)
            Object.RALPER = 100* sqrt(ANST[7]**2+Object.AIOERT**2)
            Object.RATTOF = Object.ATTOINT*Object.RI[1]/(1.0-Object.ATTOINT)
            Object.RATTOFER = 100*sqrt(ANST[7]**2+Object.ATTERT**2)

    else:
        # For net ionisation take results from final plane

        I1 = Object.NumberOfTimeSteps
        Object.TOFENE = Object.EPT[I1-1]
        Object.TOFENER = 100 * abs((Object.EPT[I1 - 1]-Object.EPT[I1 - 2])/(2*Object.EPT[I1 - 1]))

        Object.TOFWV = Object.VZPT[I1 - 1]
        Object.TOFWVER = 100 * abs((Object.VZPT[I1 - 1]-Object.VZPT[I1 - 2])/(2*Object.VZPT[I1 - 1]))

        Object.TOFDL = DLTF[I1 - 1]
        Object.TOFDLER = 100 * abs((DLTF[I1 - 1]-DLTF[I1 - 2])/(2*DLTF[I1 - 1]))

        DT2 = (DXTF[I1 - 1]+DYTF[I1 - 1])/2
        DT3 = (DXTF[I1 - 2]+DYTF[I1 - 2])/2

        Object.TOFDT = DT2
        Object.TOFDTER = 100 * abs((DT2-DT3)/(2*DT2))

        Object.TOFWR = WR[I1 - 1]
        Object.TOFWRER = 100 * abs((WR[I1 - 1]-WR[I1 - 2])/(2*WR[I1 - 1]))

        ATER = abs(Object.RI[I1-1]-Object.RI[I1-2])/(Object.RI[I1-1])

        Object.RALPHA = Object.RI[I1-1]/(1.0-Object.ATTOINT)
        Object.RALPER = 100*sqrt(ATER**2+Object.AIOERT**2)

        Object.RATTOF = Object.ATTOINT * Object.RI[I1-1]/(1.0-Object.ATTOINT)
        if Object.ATTOINT!=0.0:
            Object.RATTOFER = 100*sqrt(ATER**2+Object.ATTERT**2)
        else:
            Object.RATTOFER = 0.0
