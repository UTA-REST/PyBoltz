from Magboltz cimport Magboltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow,tan,atan



from CollisionFreqs cimport CollisionFreq, CollisionFreqT
cimport cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef TOF(Magboltz object, int JPRT):
    cdef double DUM[6],DLTF[8],DXTF[8],DYTF[8],WR[8],ANTPL[8],FRION,FRATT,ATTOINT,CORERR,TDT2,TDT3
    cdef double ANST2,ANST3,ANST4,ANST5,ANST6,ANST7,ANST8,ATER
    cdef int I1,I2,I
    if object.EnableThermalMotion == 1:
        DUM = CollisionFreqT(object)
    if object.EnableThermalMotion == 0:
        DUM = CollisionFreq(object)

    FRION = DUM[2]
    FRATT = DUM[3]
    ATTOINT = FRATT / FRION
    CORERR = abs((object.FAKEI + FRION - FRATT) / (FRION - FRATT))
    ANTPL[0] = float(object.NETPL[0])
    WR[0] = object.ZTPL[0] / (ANTPL[0] * object.TSTEP)
    DLTF[0] = ((object.ZZTPL[0] / ANTPL[0]) - (object.ZTPL[0] / ANTPL[0]) ** 2) / (2 * object.TSTEP)
    DXTF[0] = ((object.XXTPL[0] / ANTPL[0]) - (object.XTPL[0] / ANTPL[0]) ** 2) / (2 * object.TSTEP)
    DYTF[0] = ((object.YYTPL[0] / ANTPL[0]) - (object.YTPL[0] / ANTPL[0]) ** 2) / (2 * object.TSTEP)

    for I in range(1, <int>(object.ITFINAL)):
        ANTPL[I] = float(object.NETPL[I])
        WR[I] = ((object.ZTPL[I] / ANTPL[I]) - (object.ZTPL[I - 1] / ANTPL[I - 1])) / object.TSTEP
        DLTF[I] = ((object.ZZTPL[I] / ANTPL[I]) - (object.ZTPL[I] / ANTPL[I]) ** 2 - (
                object.ZZTPL[I - 1] / ANTPL[I - 1]) + (object.ZTPL[I - 1] / ANTPL[I - 1]) ** 2) / (
                          2 * object.TSTEP)
        DXTF[I] = ((object.XXTPL[I] / ANTPL[I]) - (object.XTPL[I] / ANTPL[I]) ** 2 - (
                object.XXTPL[I - 1] / ANTPL[I - 1]) + (object.XTPL[I - 1] / ANTPL[I - 1]) ** 2) / (
                          2 * object.TSTEP)
        DYTF[I] = ((object.YYTPL[I] / ANTPL[I]) - (object.YTPL[I] / ANTPL[I]) ** 2 - (
                object.YYTPL[I - 1] / ANTPL[I - 1]) + (object.YTPL[I - 1] / ANTPL[I - 1]) ** 2) / (
                          2 * object.TSTEP)
    for I in range(<int>(object.ITFINAL)):
        WR[I] *= 1e9
        DLTF[I] *= 1e16
        DXTF[I] *= 1e16
        DYTF[I] *= 1e16

    # TODO: PRINT TIME OF FLIGHT RSULTS

    if object.NETPL[0] > object.NETPL[<int>(object.ITFINAL)]:
        object.TOFENE = object.EPT[1]
        object.TOFENER = 100 * abs((object.EPT[1] - object.EPT[2]) / (2 * object.EPT[1]))
        object.TOFWV = object.VZPT[1]
        object.TOFWVER = 100 * abs((object.VZPT[1] - object.VZPT[2]) / (2 * object.VZPT[1]))
        object.TOFDL = DLTF[1]
        object.TOFDLER = 100 * abs((DLTF[1] - DLTF[2]) / (2 * DLTF[1]))
        TDT2 = (DXTF[1] + DYTF[1]) / 2
        TDT3 = (DXTF[2] + DYTF[2]) / 2
        object.TOFDT = TDT2
        object.TOFDTER = 100 * abs((TDT2 - TDT3) / (2 * TDT2))
        object.TOFWR = WR[1]
        object.TOFWRER = 100 * abs((WR[1] - WR[2]) / (2 * WR[1]))
        ANST2 = float(object.NETPL[1])
        ANST3 = float(object.NETPL[2])
        ANST4 = ANST3 - sqrt(ANST3)
        ANST5 = log(ANST2 / ANST3)
        ANST6 = log(ANST2 / ANST4)
        ANST7 = ANST6 / ANST5
        ANST8 = ANST7 - 1
        if ATTOINT == -1:
            object.RALPHA = 0.0
            object.RALPER = 0.0
            object.RATTOF = -1 * object.RI[1]
            object.RATTOFER = 100 * sqrt(ANST8 ** 2 + object.ATTERT ** 2)
        else:
            object.RALPHA = object.RI[1] / (1 - ATTOINT)
            object.RALPER = 100 * sqrt(ANST8 ** 2 + object.AIOERT ** 2)
            object.RATTOF = ATTOINT * object.RI[1] / (1 - ATTOINT)
            object.RATTOFER = 100 * sqrt(ANST8 ** 2 + object.ATTERT ** 2)
    else:
        I1 = <int>(object.ITFINAL)
        I2 = <int>(object.ITFINAL - 1)
        object.TOFENE = object.EPT[I1]
        object.TOFENER = 100 * abs((object.EPT[I1] - object.EPT[I2]) / object.EPT[I1])
        object.TOFWV = object.VZPT[I1]
        object.TOFWVER = 100 * abs((object.VZPT[I1] - object.VZPT[I2]) / object.VZPT[I1])
        object.TOFDL = DLTF[I1]
        object.TOFDLER = 100 * abs((DLTF[I1] - DLTF[I2]) / DLTF[I1])
        TDT1 = (DXTF[I1] + DYTF[I1]) / 2
        TDT2 = (DXTF[I2] + DYTF[I2]) / 2
        object.TOFDT = TDT1
        object.TOFDTER = 100 * abs((TDT1 - TDT2) / TDT1)
        object.TOFWR = WR[I1]
        object.TOFWRER = 100 * abs((WR[I1] - WR[I2]) / WR[I1])
        ATER = abs((object.RI[I1] - object.RI[I2]) / object.RI[I1])
        object.RALPHA = object.RI[I1] / (1 - ATTOINT)
        object.RALPER = 100 * sqrt(ATER ** 2 + object.AIOERT ** 2)
        object.RATTOF=ATTOINT+object.RI[I1]/(1-ATTOINT)
        if ATTOINT!=0.0:
            object.RATTOFER=100*sqrt(ATER**2+object.ATTERT**2)
        else:
            object.RATTOFER=0.0


