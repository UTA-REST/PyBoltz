from Magboltz cimport Magboltz
cimport cython
from Magboltz cimport drand48
from libc.math cimport sin, cos, acos, asin, log,log10, sqrt, pow
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
cpdef ELIMITCT(Magboltz Object):
    cdef long long I, ISAMP, N4000, IMBPT, J1, KGAS, IE
    cdef double SMALL, RDUM, E1, TDASH, CONST9, CONST10, DCZ1, DCX1, DCY1, BP, F1, F2, F4, J2M, R5, TEST1, R1, T, AP, E, CONST6, DCX2, DCY2, DCZ2, R2,API
    cdef double VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, CONST11, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, F3, RAN, EPSI, R4, PHI0, F8, F9, ARG1,
    cdef double D, Q, U, CSQD, F6, F5, ARGZ,RTHETA, CONST12, VXLAB, VYLAB, EFX100,EFZ100,TLIM,CX2,CY2,CZ2,EOVBR
    ISAMP = 20
    SMALL = 1.0e-20

    I = 0
    API = acos(-1)

    RTHETA = Object.BTHETA*API/180.0
    EFZ100 = Object.EFIELD*100*sin(RTHETA)
    EFX100 = Object.EFIELD*100*cos(RTHETA)
    F1 = Object.EFIELD*Object.CONST2*cos(RTHETA)
    F4 =2*API
    EOVBR =Object.EOVB * sin(RTHETA)
    RDUM = Object.RSTART
    E1 = Object.ESTART
    TDASH =0.0

    CONST9 = Object.CONST3*0.01
    CONST10 = CONST9**2

    #GENERATE RANDOM NUMBER FOR MAXWELL BOLTZMAN
    GERJAN(Object.RSTART, Object.NGAS, Object.RNMX)
    IMBPT = 0

    #INITIAL DIRECTION COSINES
    DCZ1 = cos(Object.THETA)
    DCX1 = sin(Object.THETA) * cos(Object.PHI)
    DCY1 = sin(Object.THETA) * sin(Object.PHI)

    VTOT = CONST9 * sqrt(E1)
    CX1 = DCX1 * VTOT
    CY1 = DCY1 * VTOT
    CZ1 = DCZ1 * VTOT

    J2M = Object.NMAX / ISAMP

    for J1 in range(int(J2M)):
        if J1 != 0  and not int(str(J1)[-int(log10(J1)):]):
            print('* Num analyzed collisions: {}'.format(J1))
        while True:
            R1 = random_uniform(RDUM)
            T = -1 * log(R1) / Object.TCFMX + TDASH
            TDASH = T
            WBT = Object.WB * T
            COSWT = cos(WBT)
            SINWT = sin(WBT)
            DZ = (CZ1 * SINWT + (Object.EOVB - CY1) * (1 - COSWT)) / Object.WB
            DX = CX1 *T+F1*T*T

            E = E1 + DZ * EFZ100+DX*EFX100

            CX2 = CX1+2*F1*T
            CY2 = (CY1 - EOVBR) * COSWT + CZ1 * SINWT + EOVBR
            CZ2 = CZ1 * COSWT - (CY1 - EOVBR) * SINWT

            #FIND IDENTITY OF GAS FOR COLLISION
            KGAS = 0
            R2 = random_uniform(RDUM)
            while (Object.TCFMXG[KGAS] < R2):
                KGAS += 1
            #CALCULATE GAS VELOCITY VECTORS VGX,VGY,VGZ
            IMBPT += 1
            if IMBPT > 6:
                GERJAN(Object.RSTART, Object.API, Object.RNMX)
                IMBPT = 1
            VGX = Object.VTMB[KGAS] * Object.RNMX[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            VGY = Object.VTMB[KGAS] * Object.RNMX[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            VGZ = Object.VTMB[KGAS] * Object.RNMX[(IMBPT - 1) % 6]

            EOK = ((CX2 - VGX) ** 2 + (CY2 - VGY) ** 2 + (CZ2 - VGZ) ** 2) / CONST10
            IE = int(EOK / Object.ESTEP)
            IE = min(IE, 3999)

            # TEST FOR REAL OR NULL COLLISION
            R5 = random_uniform(RDUM)
            TLIM = Object.TCF[KGAS][IE] / Object.TCFMAX[KGAS]
            if R5 <= TLIM:
                break
        if IE == 3999:
            Object.IELOW = 1
            return

        #CALCULATE DIRECTION COSINES OF ELECTRON IN 0 KELVIN FRAME
        TDASH = 0.0
        CONST11 = 1.0 / (CONST9 * sqrt(EOK))
        DXCOM = (CX2 - VGX) * CONST11
        DYCOM = (CY2 - VGY) * CONST11
        DZCOM = (CZ2 - VGZ) * CONST11

        # ---------------------------------------------------------------------
        #     DETERMINATION OF REAL COLLISION TYPE
        # ---------------------------------------------------------------------

        R2 = random_uniform(RDUM)

        #FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
        I = SORTT(KGAS, I, R2, IE, Object)
        while Object.CF[KGAS][IE][I] < R2:
            I = I + 1


        S1 = Object.RGAS[KGAS][I]
        EI = Object.EIN[KGAS][I]
        if Object.IPN[KGAS][I] > 0:
            R9 = random_uniform(RDUM)
            EXTRA = R9 * (EOK - EI)
            EI = EXTRA + EI

        #  GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
        #   COLLISION ALSO UPDATE ENERGY OF ELECTRON.
        IPT = Object.IARRY[KGAS][I]
        if EOK < EI:
            EI = EOK - 0.0001
        S2 = (S1 * S1) / (S1 - 1)
        R3 = random_uniform(RDUM)

        if Object.INDEX[KGAS][I] == 1:
            R31 = random_uniform(RDUM)
            F3 = 1- R3 *Object.ANGCT[KGAS][IE][I]
            if R31 > Object.PSCT[KGAS][IE][I]:
                F3 = -1 * F3
            elif Object.INDEX[KGAS][I] == 2:
                EPSI = Object.PSCT[KGAS][IE][I]
                F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
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
        CSQD = F3 ** 2

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
        VTOT = CONST9 * sqrt(E1)
        CX1 = DCX1 * VTOT + VGX
        CY1 = DCY1 * VTOT + VGY
        CZ1 = DCZ1 * VTOT + VGZ
        #  CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
        E1 = (CX1 * CX1 + CY1 * CY1 + CZ1 * CZ1) / CONST10
        CONST11 = 1.0 / (CONST9 * sqrt(E1))
        DCX1 = CX1 * CONST11
        DCY1 = CY1 * CONST11
        DCZ1 = CZ1 * CONST11

    Object.IELOW = 0
    return
