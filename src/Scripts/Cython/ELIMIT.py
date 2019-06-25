from Magboltz cimport Magboltz
cimport cython
from Magboltz cimport drand48
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
from SORTT cimport SORTT



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
    
    ISAMP = 10
    SMALL = 1.0e-20
    I = 0
    RDUM = Magboltz.RSTART
    E1 = Magboltz.ESTART
    N4000 = 4000
    TDASH = 0.0
    INTEM = 8
    TEMP = np.zeros(4000)
    for J in range(N4000):
        TEMP[J] = Magboltz.TCFN[J] + Magboltz.TCF[J]

    DCZ1 = math.cos(Magboltz.THETA)
    DCX1 = math.sin(Magboltz.THETA) * math.cos(Magboltz.PHI)
    DCY1 = math.sin(Magboltz.THETA) * math.sin(Magboltz.PHI)

    BP = (Magboltz.EFIELD ** 2) * Magboltz.CONST1
    F1 = Magboltz.EFIELD * Magboltz.CONST2
    F2 = Magboltz.EFIELD * Magboltz.CONST3
    F4 = 2 * math.acos(-1)
    J2M = Magboltz.NMAX / ISAMP
    DELTAE = Magboltz.EFINAL / float(INTEM)

    for J1 in range(int(J2M)):
        R5 = 1
        TEST1 = 0
        while R5 > TEST1:
            R1 = Magboltz.RAND48.drand()
            I = int(E1 / DELTAE) + 1
            I = min(I, INTEM) - 1
            TLIM = Magboltz.TCFMAX[I]
            T = -1 * np.log(R1) / TLIM + TDASH
            TDASH = T
            AP = DCZ1 * F2 * math.sqrt(E1)
            E = E1 + (AP + BP * T) * T
            IE = int(E / Magboltz.ESTEP)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += np.log(R1) / TLIM
                Magboltz.TCFMAX[I] *= 1.05
                continue
            R5 = Magboltz.RAND48.drand()
            TEST1 = Magboltz.TCF[IE] / TLIM
        if IE == 3999:
            Magboltz.IELOW = 1

        TDASH = 0.0
        CONST6 = math.sqrt(E1 / E)
        DCX2 = DCX1 * CONST6
        DCY2 = DCY1 * CONST6
        DCZ2 = DCZ1 * CONST6 + Magboltz.EFIELD * T * Magboltz.CONST5 / math.sqrt(E)
        R2 = Magboltz.RAND48.drand()
        I = 0
        I = SORT(I, R2, IE, Magboltz)
        while Magboltz.CF[IE][I] < R2:
            I = I + 1
        S1 = Magboltz.RGAS[I]
        EI = Magboltz.EIN[I]
        if Magboltz.IPN[I] > 0:
            R9 = Magboltz.RAND48.drand()
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI
        IPT = Magboltz.IARRY[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = Magboltz.RAND48.drand()

        if Magboltz.INDEX[I] == 1:
            R31 = Magboltz.RAND48.drand()
            F3 = 1.0 - R3 * Magboltz.ANGCT[IE][I]
            if R31 > Magboltz.PSCT[IE][I]:
                F3 = -1 * F3
            elif Magboltz.INDEX[I] == 2:
                EPSI = Magboltz.PSCT[IE][I]
                F3 = 1 - (2 * R3 * (1 - EPSI)) / (1 + EPSI * (1 - 2 * R3))
            else:
                F3 = 1 - 2 * R3
        THETA0 = math.acos(F3)
        R4 = Magboltz.RAND48.drand()
        PHI0 = F4 * R4
        F8 = math.sin(PHI0)
        F9 = math.cos(PHI0)
        ARG1 = 1 - S1 * EI / E
        ARG1 = max(ARG1, SMALL)

        D = 1 - F3 * math.sqrt(ARG1)
        E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = math.sqrt((E / E1) * ARG1) / S1
        Q = min(Q, 1)
        Magboltz.THETA = math.asin(Q * math.sin(THETA0))

        F6 = math.cos(Magboltz.THETA)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = F3 ** 2

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6

        F5 = math.sin(Magboltz.THETA)
        DCZ2 = min(DCZ2, 1)
        ARGZ = math.sqrt(DCX2 * DCX2 + DCY2 * DCY2)
        if ARGZ == 0:
            DCZ1 = F6
            DCX1 = F9 * F5
            DCY1 = F8 * F5
        else:
            DCZ1 = DCZ2 * F6 + ARGZ * F5 * F8
            DCY1 = DCY2 * F6 + (F5 / ARGZ) * (DCX2 * F9 - DCY2 * DCZ2 * F8)
            DCX1 = DCX2 * F6 - (F5 / ARGZ) * (DCY2 * F9 + DCX2 * DCZ2 * F8)

    Magboltz.IELOW = 0
