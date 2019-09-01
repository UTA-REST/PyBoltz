from Magboltz cimport Magboltz
from libc.math cimport sqrt


cimport cython
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef TPLANET(Magboltz object, double T,double E1,double DCX1, double DCY1,double DCZ1,double AP, double BP, int IPLANE):
    cdef double TIMESP,TIMLFT,T2LFT,A,B,EPLANE,CONST6,DCZ2,XPLANE,YPLANE,ZPLANE,VZPLANE
    TIMESP = IPLANE * object.TSTEP
    TIMLFT = TIMESP - object.ST
    T2LFT = TIMLFT ** 2
    A = AP * TIMLFT
    B = BP * TIMLFT
    EPLANE = E1 + A + B
    CONST6 = sqrt(E1 / EPLANE)

    DCZ2 = DCZ1 * CONST6 + object.EFIELD * TIMLFT * object.CONST5 / sqrt(EPLANE)
    XPLANE = object.X + DCX1 * TIMLFT * sqrt(E1) * object.CONST3 * 0.01
    YPLANE = object.Y + DCY1 * TIMLFT * sqrt(E1) * object.CONST3 * 0.01
    ZPLANE = object.Z + DCZ1 * TIMLFT * sqrt(
        E1) * object.CONST3 * 0.01 + T2LFT * object.EFIELD * object.CONST2
    VZPLANE = DCZ2 * sqrt(EPLANE) * object.CONST3 * 0.01
    object.XTPL[IPLANE] += XPLANE
    object.YTPL[IPLANE] += YPLANE
    object.ZTPL[IPLANE] += ZPLANE
    object.XXTPL[IPLANE] += XPLANE ** 2
    object.YYTPL[IPLANE] += YPLANE ** 2
    object.ZZTPL[IPLANE] += ZPLANE ** 2
    object.ETPL[IPLANE] += EPLANE
    object.TTPL[IPLANE] += object.ST + TIMLFT
    object.VZTPL[IPLANE] += VZPLANE
    object.NETPL[IPLANE] += 1


