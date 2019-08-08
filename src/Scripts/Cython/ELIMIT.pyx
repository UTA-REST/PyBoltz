from Magboltz cimport Magboltz
cimport cython
from Magboltz cimport drand48
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow,log10
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from SORT cimport SORT



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = drand48(dummy)
    return r


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void GERJAN(double RDUM, double API, double *RNMX):
    cdef double RAN1, RAN2, TWOPI
    cdef int J
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RDUM)
        RAN2 = random_uniform(RDUM)
        TWOPI = 2.0 * API
        RNMX[J] = sqrt(-1 * log(RAN1)) * cos(RAN2 * TWOPI)
        RNMX[J + 1] = sqrt(-1 * log(RAN1)) * sin(RAN2 * TWOPI)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ELIMIT(Magboltz Object):

    cdef long long I, ISAMP, N4000, IMBPT, J1, KGAS, IE, INTEM
    cdef double SMALL, RDUM, E1, TDASH, CONST9, CONST10, DCZ1, DCX1, DCY1, BP, F1, F2, F4, J2M, R5, TEST1, R1, T, AP, E, CONST6, DCX2, DCY2, DCZ2, R2,
    cdef double VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, CONST11, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, F3, RAN, EPSI, R4, PHI0, F8, F9, ARG1
    cdef double D, Q, U, CSQD, F6, F5, ARGZ, CONST12, VXLAB, VYLAB, VZLAB, TEMP[4000],DELTAE
    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))

    ISAMP = 10
    SMALL = 1.0e-20
    I = 0
    RDUM = Object.RSTART
    E1 = Object.ESTART
    N4000 = 4000
    TDASH = 0.0
    INTEM = 8
    for J in range(N4000):
        TEMP[J] = Object.TCFN1[J] + Object.TCF1[J]

    # INITIAL DIRECTION COSINES
    DCZ1 = cos(Object.THETA)
    DCX1 = sin(Object.THETA) * cos(Object.PHI)
    DCY1 = sin(Object.THETA) * sin(Object.PHI)

    BP = (Object.EFIELD**2)*Object.CONST1
    F1 = Object.EFIELD * Object.CONST2
    F2 = Object.EFIELD * Object.CONST3
    F4 = 2 * acos(-1)
    DELTAE = Object.EFINAL/float(INTEM)
    E1 = Object.ESTART
    J2M = Object.NMAX / ISAMP
    for J1 in range(int(J2M)):
        if J1 != 0  and not int(str(J1)[-int(log10(J1)):]):
            print('* Num analyzed collisions: {}'.format(J1))
        print("HERE")
        print(IE)
        print(I)
        while True:
            R1 = random_uniform(RDUM)
            I = int(E1 / DELTAE) + 1
            I = min(I, INTEM) - 1
            TLIM = Object.TCFMAX1[I]
            T = -1 * log(R1) / TLIM + TDASH
            TDASH = T
            AP = DCZ1 * F2 * sqrt(E1)
            E = E1 + (AP + BP * T) * T
            IE = int(E / Object.ESTEP)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += log(R1) / TLIM
                Object.TCFMAX1[I] *= 1.05
                continue

            # TEST FOR NULL COLLISIONS
            R5 = random_uniform(RDUM)
            TEST1 = Object.TCF1[IE] / TLIM
            if R5<=TEST1:
                break

        print(IE)
        print(I)
        if IE == 3999:
            Object.IELOW = 1
            return

        # CALCULATE DIRECTION COSINES AT INSTANT BEFORE COLLISION
        TDASH = 0.0
        CONST6 = sqrt(E1 / E)
        DCX2 = DCX1 * CONST6
        DCY2 = DCY1 * CONST6
        DCZ2 = DCZ1 * CONST6 + Object.EFIELD * T * Object.CONST5 / sqrt(E)
        R2 = random_uniform(RDUM)


        I = SORT(I, R2, IE, Object)
        while Object.CF1[IE][I] < R2:
            I = I + 1

        S1 = Object.RGAS1[I]
        EI = Object.EIN1[I]
        if Object.IPN1[I] > 0:
            R9 = random_uniform(RDUM)
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI
        IPT = Object.IARRY1[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random_uniform(RDUM)

        if Object.INDEX1[I] == 1:
            R31 = random_uniform(RDUM)
            F3 = 1.0 - R3 * Object.ANGCT1[IE][I]
            if R31 > Object.PSCT1[IE][I]:
                F3 = -1 * F3
        elif Object.INDEX1[I] == 2:
            EPSI = Object.PSCT1[IE][I]
            F3 = 1 - (2 * R3 * (1 - EPSI)) / (1 + EPSI * (1 - 2 * R3))
        else:
            F3 = 1 - 2 * R3
        THETA0 = acos(F3)
        R4 = random_uniform(RDUM)
        PHI0 = F4 * R4
        F8 = sin(PHI0)
        F9 = cos(PHI0)
        ARG1 = 1 - S1 * EI / E
        ARG1 = max(ARG1, SMALL)

        D = 1 - F3 * sqrt(ARG1)
        E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = sqrt((E / E1) * ARG1) / S1
        Q = min(Q, 1)
        Object.THETA = asin(Q * sin(THETA0))

        F6 = cos(Object.THETA)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = F3 ** 2

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6

        F5 = sin(Object.THETA)
        DCZ2 = min(DCZ2, 1)
        ARGZ = sqrt(DCX2 * DCX2 + DCY2 * DCY2)
        if ARGZ == 0:
            DCZ1 = F6
            DCX1 = F9 * F5
            DCY1 = F8 * F5
        else:
            DCZ1 = DCZ2 * F6 + ARGZ * F5 * F8
            DCY1 = DCY2 * F6 + (F5 / ARGZ) * (DCX2 * F9 - DCY2 * DCZ2 * F8)
            DCX1 = DCX2 * F6 - (F5 / ARGZ) * (DCY2 * F9 + DCX2 * DCZ2 * F8)

    Object.IELOW = 0
