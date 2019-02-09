from Magboltz cimport Magboltz
cimport cython
from Magboltz cimport drand48
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow
from SORTT cimport SORTT

cdef void GERJAN(double RDUM, double API,double *RNMX):
    cdef double RAN1, RAN2, TWOPI
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RDUM)
        RAN2 = random_uniform(RDUM)
        TWOPI = 2.0 * API
        RNMX[J] = sqrt(-1*log(RAN1)) * cos(RAN2 * TWOPI)
        RNMX[J + 1] = sqrt(-1*log(RAN1)) * sin(RAN2 * TWOPI)




@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = drand48(dummy)
    return r



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ELIMITT(Magboltz Object):
    print("ELIMITT")
    cdef long long I, ISAMP, N4000, IMBPT, J1, KGAS, IE
    cdef double SMALL,RDUM,  E1, TDASH, CONST9, CONST10, DCZ1, DCX1, DCY1, BP, F1, F2, F4, J2M, R5, TEST1, R1, T, AP, E, CONST6, DCX2, DCY2, DCZ2, R2,
    cdef double VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, CONST11, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, F3, RAN, EPSI, R4, PHI0, F8, F9, ARG1
    cdef double D, Q, U, CSQD, F6, F5, ARGZ, CONST12, VXLAB, VYLAB, VZLAB

    ISAMP = 10
    SMALL = 1.0e-20
    RDUM = Object.RSTART
    E1 = Object.ESTART
    N4000 = 4000
    TDASH = 0.0
    CONST9 = Object.CONST3 * 0.01
    CONST10 = CONST9 * CONST9
    GERJAN(Object.RSTART, Object.API, Object.RNMX)
    IMBPT = 0
    DCZ1 = cos(Object.THETA)
    DCX1 = sin(Object.THETA) * cos(Object.PHI)
    DCY1 = sin(Object.THETA) * sin(Object.PHI)

    BP = pow(Object.EFIELD , 2) * Object.CONST1
    F1 = Object.EFIELD * Object.CONST2
    F2 = Object.EFIELD * Object.CONST3
    F4 = 2 * acos(-1)

    J2M = Object.NMAX / ISAMP
    print(J2M)

    for J1 in range(int(J2M)):
        if J1 % 100000 == 0:
            print(J1)
        while True:
            R1 = random_uniform(RDUM)
            T = -1 * log(R1) / Object.TCFMX + TDASH
            TDASH = T
            AP = DCZ1 * F2 * sqrt(E1)
            E = E1 + (AP + BP * T) * T
            CONST6 = sqrt(E1 / E)
            DCX2 = DCX1 * CONST6
            DCY2 = DCY1 * CONST6
            DCZ2 = DCZ1 * CONST6 + Object.EFIELD * T * Object.CONST5 / sqrt(E)
            R2 = random_uniform(RDUM)
            KGAS = 0
            for KGAS in range(Object.NGAS):
                if Object.TCFMXG[KGAS] >= R2:
                    break
            IMBPT = IMBPT + 1
            if IMBPT > 6:
                GERJAN(Object.RSTART, Object.NGAS, Object.RNMX)
                IMBPT = 1
            VGX = Object.VTMB[KGAS] * Object.RNMX[(IMBPT - 1) ]
            IMBPT = IMBPT + 1
            VGY = Object.VTMB[KGAS] * Object.RNMX[(IMBPT - 1) ]
            IMBPT = IMBPT + 1
            VGZ = Object.VTMB[KGAS] * Object.RNMX[(IMBPT - 1) ]

            VEX = DCX2 * CONST9 * sqrt(E)
            VEY = DCY2 * CONST9 * sqrt(E)
            VEZ = DCZ2 * CONST9 * sqrt(E)

            EOK = (pow((VEX - VGX) ,2) + pow((VEY - VGY) , 2) + pow((VEZ - VGZ) , 2)) / CONST10
            IE = int(EOK / Object.ESTEP)
            IE = min(IE, 3999)
            R5 = random_uniform(RDUM)
            TEST1 = Object.TCF[KGAS][IE] / Object.TCFMAX[KGAS]
            if R5<=TEST1:
                break

        if IE == 3999:
            Object.IELOW = 1
            return

        TDASH = 0.0

        CONST11 = 1 / (CONST9 * sqrt(EOK))
        DXCOM = (VEX - VGX) * CONST11
        DYCOM = (VEY - VGY) * CONST11
        DZCOM = (VEZ - VGZ) * CONST11

        R2 = random_uniform(RDUM)
        I = SORTT(KGAS, I, R2, IE, Object)
        while Object.CF[KGAS][IE][I] < R2:
            I = I + 1

        S1 = Object.RGAS[KGAS][I]
        EI = Object.EIN[KGAS][I]

        if Object.IPN[KGAS][I] > 0:
            R9 = random_uniform(RDUM)
            EXTRA = R9 * (EOK - EI)
            EI = EXTRA + EI
        IPT = Object.IARRY[KGAS][I]
        if EOK < EI:
            EI = EOK - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random_uniform(RDUM)

        if Object.INDEX[KGAS][I] == 1:
            R31 = random_uniform(RDUM)
            F3 = 1.0 - R3 * Object.ANGCT[KGAS][IE][I]
            if R31 > Object.PSCT[KGAS][IE][I]:
                F3 = -1 * F3
        elif Object.INDEX[KGAS][I] == 2:
            EPSI = Object.PSCT[KGAS][IE][I]
            F3 = 1 - (2 * R3 * (1 - EPSI)) / (1 + EPSI * (1 - 2 * R3))
        else:
            F3 = 1 - 2 * R3
        THETA0 = acos(F3)
        R4 = random_uniform(RDUM)
        PHI0 = F4 * R4
        F8 = sin(PHI0)
        F9 = cos(PHI0)
        ARG1 = 1 - S1 * EI / EOK
        ARG1 = max(ARG1, SMALL)

        D = 1 - F3 * sqrt(ARG1)
        E1 = EOK * (1 - EI / (S1 * EOK) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = sqrt((EOK / E1) * ARG1) / S1
        Q = min(Q, 1)
        Object.THETA = asin(Q * sin(THETA0))

        F6 = cos(Object.THETA)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = pow(F3 , 2)

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6
        F5 = sin(Object.THETA)
        DZCOM = min(DZCOM, 1)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
        if ARGZ == 0:
            DCZ1 = F6
            DCX1 = F9 * F5
            DCY1 = F8 * F5
        else:
            DCZ1 = DZCOM * F6 + ARGZ * F5 * F8
            DCY1 = DYCOM * F6 + (F5 / ARGZ) * (DXCOM * F9 - DYCOM * DZCOM * F8)
            DCX1 = DXCOM * F6 - (F5 / ARGZ) * (DYCOM * F9 + DXCOM * DZCOM * F8)
        # TRANSFORM VELOCITY VECTORS TO LAB FRAME
        CONST12 = CONST9 * sqrt(E1)
        VXLAB = DCX1 * CONST12 + VGX
        VYLAB = DCY1 * CONST12 + VGY
        VZLAB = DCZ1 * CONST12 + VGZ
        #  CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
        E1 = (VXLAB * VXLAB + VYLAB * VYLAB + VZLAB * VZLAB) / CONST10
        CONST11 = 1.0 / (CONST9 * sqrt(E1))
        DCX1 = VXLAB * CONST11
        DCY1 = VYLAB * CONST11
        DCZ1 = VZLAB * CONST11
        print(DCX1)
        print (DCY1)
        print(DCZ1)
        raise ValueError("STOP")
    Object.IELOW = 0
    print("THETAAAAAAAAA")
    print(Object.THETA)
    return
