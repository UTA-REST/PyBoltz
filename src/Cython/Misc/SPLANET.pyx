from Magboltz cimport Magboltz
from libc.math cimport sqrt


cimport cython
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef SPLANET(Magboltz object, double T,double E1,double DCX1,double DCY1,double DCZ1,double AP,double BP,double TIMLFT,int IZPLANE):
    cdef double T2LFT,A,B,EPLANE,CONST6,DCZ2,XPLANE,YPLANE,ZPLANE,VZPLANE,WGHT,RPLANE
    if IZPLANE > 8:
        return
    T2LFT = TIMLFT ** 2
    A = AP * TIMLFT
    B = BP * T2LFT
    EPLANE = E1 + A + B
    CONST6 = sqrt(E1 / EPLANE)
    DCZ2 = DCZ1 * CONST6 + object.EFIELD * TIMLFT * object.CONST5 / sqrt(EPLANE)
    XPLANE = object.X + DCX1 * TIMLFT * sqrt(E1) * object.CONST3 * 0.01
    YPLANE = object.Y + DCY1 * TIMLFT * sqrt(E1) * object.CONST3 * 0.01
    ZPLANE = object.Z + DCZ1 * TIMLFT * sqrt(E1) * object.CONST3 * 0.01
    VZPLANE = DCZ2 * sqrt(EPLANE) * object.CONST3 * 0.01
    WGHT = abs(1 / VZPLANE)
    RPLANE = sqrt(XPLANE ** 2 + YPLANE ** 2)
    object.XSPL[IZPLANE - 1] += XPLANE * WGHT
    object.YSPL[IZPLANE - 1] += YPLANE * WGHT
    object.RSPL[IZPLANE - 1] += RPLANE * WGHT
    object.ZSPL[IZPLANE - 1] += ZPLANE * WGHT
    object.TMSPL[IZPLANE - 1] += (object.ST + TIMLFT) * WGHT
    object.TTMSPL[IZPLANE - 1] += (object.ST + TIMLFT) * (object.ST + TIMLFT) * WGHT
    object.XXSPL[IZPLANE - 1] += WGHT * XPLANE ** 2
    object.YYSPL[IZPLANE - 1] += WGHT * YPLANE ** 2
    object.ZZSPL[IZPLANE - 1] += WGHT * ZPLANE ** 2
    object.RRSPM[IZPLANE - 1] += WGHT * RPLANE ** 2
    object.ESPL[IZPLANE - 1] += EPLANE * WGHT
    object.TSPL[IZPLANE - 1] += WGHT / (object.ST + TIMLFT)
    object.VZSPL[IZPLANE - 1] += VZPLANE * WGHT
    object.TSSUM[IZPLANE - 1] += WGHT
    object.TSSUM2[IZPLANE - 1] += WGHT ** 2


