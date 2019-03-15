from Magboltz cimport Magboltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow,tan,atan


cimport cython
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef TCALCT(Magboltz object, double DCZ1,double E1,double TZSTOP,double TZSTOP1,double ISOL,int IZPLANE):
    cdef double R[4],B,A,B2,C1,C2,FAC,TSTOP1,TSTOP2
    ISOL = 1
    A = object.EFIELD
    B = sqrt(E1) * object.CONST3 * 0.01 * DCZ1
    B2 = B * B
    R[0] = ISOL
    R[1] = IZPLANE
    R[2] = TZSTOP
    R[3] = TZSTOP1
    if object.Z < object.ZPLANE[0]:
        IZPLANE = 1
        C1 = object.Z - object.ZPLANE[0]
    elif object.Z < object.ZPLANE[1]:
        IZPLANE = 2
        C1 = object.Z - object.ZPLANE[1]
        C2 = object.Z - object.ZPLANE[0]
    elif object.Z < object.ZPLANE[2]:
        IZPLANE = 3
        C1 = object.Z - object.ZPLANE[2]
        C2 = object.Z - object.ZPLANE[1]
    elif object.Z < object.ZPLANE[3]:
        IZPLANE = 4
        C1 = object.Z - object.ZPLANE[3]
        C2 = object.Z - object.ZPLANE[2]
    elif object.Z < object.ZPLANE[4]:
        IZPLANE = 5
        C1 = object.Z - object.ZPLANE[4]
        C2 = object.Z - object.ZPLANE[3]
    elif object.Z < object.ZPLANE[5]:
        IZPLANE = 6
        C1 = object.Z - object.ZPLANE[5]
        C2 = object.Z - object.ZPLANE[4]
    elif object.Z < object.ZPLANE[6]:
        IZPLANE = 7
        C1 = object.Z - object.ZPLANE[6]
        C2 = object.Z - object.ZPLANE[5]
    elif object.Z < object.ZPLANE[7]:
        IZPLANE = 8
        C1 = object.Z - object.ZPLANE[7]
        C2 = object.Z - object.ZPLANE[6]
    else:
        IZPLANE = 9
        C1 = object.Z - object.ZPLANE[7] - 10 * object.ZSTEP
        C2 = object.Z - object.ZPLANE[7]
    FAC = B2 - 4 * A * C1
    R[1] = IZPLANE
    if FAC < 0:
        R[2] = -99
        return R
    TSTOP1 = (-B + sqrt(FAC)) / (2 * A)
    TSTOP2 = (-B - sqrt(FAC)) / (2 * A)
    if TSTOP1 < TSTOP2:
        if TSTOP1 > 0.0:
            TZSTOP = TSTOP1
        else:
            TZSTOP = TSTOP2
        if IZPLANE == 1:
            R[0] = ISOL
            R[1] = IZPLANE
            R[2] = TZSTOP
            R[3] = TZSTOP1
            return R
    else:
        if TSTOP2 > 0.0:
            TZSTOP = TSTOP2
        else:
            TZSTOP = TSTOP1
        if IZPLANE == 1:
            R[0] = ISOL
            R[1] = IZPLANE
            R[2] = TZSTOP
            R[3] = TZSTOP1
            return R

    FAC = B2 - 4 * A * C2
    TSTOP1 = (-B + sqrt(FAC)) / (2 * A)
    TSTOP2 = (-B - sqrt(FAC)) / (2 * A)
    if TSTOP1 < 0:
        return R
    ISOL = 2
    IZPLANE -= 1
    R[0] = ISOL
    if TSTOP1 < TSTOP2:
        TZSTOP = TSTOP1
        TZSTOP1 = TSTOP2
    else:
        TZSTOP = TSTOP2
        TZSTOP1 = TSTOP1
    R[0] = ISOL
    R[1] = IZPLANE
    R[2] = TZSTOP
    R[3] = TZSTOP1
    return R
