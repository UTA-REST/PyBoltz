from Magboltz cimport Magboltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow
from libc.string cimport memset
from Magboltz cimport drand48
from SORT cimport SORT
from libc.stdlib cimport malloc, free
import cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = drand48(dummy)
    return r

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void GERJAN(double RDUM, double API,double *RNMX):
    cdef double RAN1, RAN2, TWOPI
    cdef int J
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RDUM)
        RAN2 = random_uniform(RDUM)
        TWOPI = 2.0 * API
        RNMX[J] = sqrt(-1*log(RAN1)) * cos(RAN2 * TWOPI)
        RNMX[J + 1] = sqrt(-1*log(RAN1)) * sin(RAN2 * TWOPI)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef MONTEA(Magboltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at temp =  tempc.

    This function is used when the magnetic field is parallel to the electric field in the z direction.    
    
    The object parameter is the Magboltz object to have the output results and to be used in the simulation.
    """
    cdef long long I, ID, XID, NCOL, IEXTRA, IMBPT, K, J, J2M, J1, J2, KGAS, IE, IT, KDUM, IPT, JDUM,NCOLDM
    cdef double ST1, RDUM,ST2, SUME2, SUMXX, SUMYY, SUMZZ, SUMVX, SUMVY, ZOLD, STOLD, ST1OLD, ST2OLD, SZZOLD, SXXOLD, SYYOLD, SVXOLD, SVYOLD, SME2OLD, TDASH
    cdef double ABSFAKEI, DCZ1, DCX1, DCY1, CX1, CY1, CZ1, BP, F1, F2, F4, DCX2, DCY2, DCZ2, CX2, CY2, CZ2, DZCOM, DYCOM, DXCOM, THETA0,
    cdef double  E1, CONST9, CONST10, AP, CONST6, R2, R1, VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, R5, TEST1, TEST2, TEST3, CONST11
    cdef double T2, A, B, CONST7, R3, S1, EI, R9, EXTRA, RAN, R31, F3, EPSI, R4, PHI0, F8, F9, ARG1, D, Q, F6, U, CSQD, F5, VXLAB, VYLAB, VZLAB
    cdef double TWZST, TAVE, T2WZST, T2AVE, TXXST, TYYST, T2XXST, T2YYST, TZZST, T2ZZST, ANCATT, ANCION, E,TEMP[4000]


    cdef double *STO, *XST, *YST, *ZST, *WZST, *AVEST, *DFZZST, *DFYYST, *DFXXST
    STO = <double *> malloc(2000000 * sizeof(double))
    memset(STO, 0, 2000000 * sizeof(double))
    XST = <double *> malloc(2000000 * sizeof(double))
    memset(XST, 0, 2000000 * sizeof(double))

    YST = <double *> malloc(2000000 * sizeof(double))
    memset(YST, 0, 2000000 * sizeof(double))

    ZST = <double *> malloc(2000000 * sizeof(double))
    memset(ZST, 0, 2000000 * sizeof(double))

    WZST = <double *> malloc(10 * sizeof(double))
    memset(WZST, 0, 10 * sizeof(double))

    AVEST = <double *> malloc(10 * sizeof(double))
    memset(AVEST, 0, 10 * sizeof(double))

    DFZZST = <double *> malloc(10 * sizeof(double))
    memset(DFZZST, 0, 10 * sizeof(double))

    DFYYST = <double *> malloc(10 * sizeof(double))
    memset(DFYYST, 0, 10 * sizeof(double))

    DFXXST = <double *> malloc(10 * sizeof(double))
    memset(DFXXST, 0, 10 * sizeof(double))


    Object.WX = 0.0
    Object.WY = 0.0
    Object.DWX = 0.0
    Object.DWY = 0.0
    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))
    for J in range(4000):
        TEMP[J] = Object.TCFNNT[J] + Object.TCFNT[J]

    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.ST = 0.0
    ST1 = 0.0
    ST2 = 0.0
    SUME2 = 0.0
    SUMXX = 0.0
    SUMYY = 0.0
    SUMZZ = 0.0
    SUMVX = 0.0
    SUMVY = 0.0
    ZOLD = 0.0
    STOLD = 0.0
    ST1OLD = 0.0
    ST2OLD = 0.0
    SZZOLD = 0.0
    SXXOLD = 0.0
    SYYOLD = 0.0
    SVXOLD = 0.0
    SVYOLD = 0.0
    SME2OLD = 0.0

    Object.SMALL = 1.0e-20
    Object.TMAX1 = 0.0
    RDUM = Object.RSTART
    E1 = Object.ESTART
    CONST9 = Object.CONST3 * 0.01
    INTEM = 8
    Object.ITMAX = 10
    ID = 0
    Object.XID = 0
    NCOL = 0
    IEXTRA = 0
    Object.NNULL = 0

    ABSFAKEI = Object.FAKEI
    Object.IFAKE = 0

    # INITIAL DIRECTION COSINES
    DCZ1 = cos(Object.THETA)
    DCX1 = sin(Object.THETA) * cos(Object.PHI)
    DCY1 = sin(Object.THETA) * sin(Object.PHI)

    # INITIAL VELOCITY
    VTOT = CONST9 * sqrt(E1)
    CX1 = DCX1 * VTOT
    CY1 = DCY1 * VTOT
    CZ1 = DCZ1 * VTOT

    BP = Object.EFIELD ** 2 * Object.CONST1
    F1 = Object.EFIELD * Object.CONST2
    F2 = Object.EFIELD * Object.CONST3
    F4 = 2 * acos(-1)

    J2M = <long long>(Object.NMAX / Object.ITMAX)
    DELTAE = Object.EFINAL / float(INTEM)
    if Object.OF:
        print('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity", "Position", "Time", "Energy",
                                                                       "DIFXX", "DIFYY", "DIFZZ"))
    for J1 in range(int(Object.ITMAX)):
        for J2 in range(int(J2M)):
            while True:
                R1 = random_uniform(RDUM)
                I = int(E1 / DELTAE) + 1
                I = min(I, INTEM) - 1
                TLIM = Object.TCFMAXNT[I]
                T = -1 * log(R1) / TLIM + TDASH
                Object.MCT = 0.9 * Object.MCT + 0.1 * T
                TDASH = T
                AP = DCZ1 * F2 * sqrt(E1)
                E = E1 + (AP + BP * T) * T
                IE = int(E / Object.ESTEP)
                IE = min(IE, 3999)
                if TEMP[IE] > TLIM:
                    TDASH += log(R1) / TLIM
                    Object.TCFMAXNT[I] *= 1.05
                    continue

                # TEST FOR REAL OR NULL COLLISION
                R5 = random_uniform(RDUM)
                TEST1 = Object.TCFNT[IE] / TLIM

                if R5 > TEST1:
                    Object.NNULL += 1
                    TEST2 = TEMP[IE] / TLIM
                    if R5 < TEST2:
                        if Object.NPLASTNT == 0:
                            continue
                        R2 = random_uniform(RDUM)
                        I = 0
                        while Object.CFNNT[IE][I] < R2:
                            I += 1

                        Object.ICOLNNNT[I] += 1
                        continue
                    else:
                        TEST3 = (TEMP[IE] + ABSFAKEI) / TLIM
                        if R5 < TEST3:
                            # FAKE IONISATION INCREMENT COUNTER
                            Object.IFAKE += 1
                            continue
                        continue
                else:
                    break


            T2 = T ** 2
            if (T >= Object.TMAX1):
                Object.TMAX1 = T
            TDASH = 0.0
            WBT = Object.WB * T
            COSWT = cos(WBT)
            SINWT = sin(WBT)
            CONST6 = sqrt(E1 / E)
            CX2 = CX1 * COSWT - CY1 * SINWT
            CY2 = CY1 * COSWT + CX1 * SINWT
            VTOT = CONST9 * sqrt(E)
            DCX2 = CX2 / VTOT
            DCY2 = CY2 / VTOT
            DCZ2 = DCZ1 * CONST6 + Object.EFIELD * T * Object.CONST5 / sqrt(E)
            A = AP * T
            B = BP * T2
            SUME2 = SUME2 + T * (E1 + A / 2.0 + B / 3.0)
            CONST7 = CONST9 * sqrt(E1)
            A = T * CONST7
            NCOL += 1
            DX = (CX1 * SINWT - CY1 * (1 - COSWT)) / Object.WB
            Object.X += DX
            DY = (CY1 * SINWT + CX1 * (1 - COSWT)) / Object.WB
            Object.Y += DY
            Object.Z += DCZ1 * A + T2 * F1
            Object.ST += T
            IT = int(T)
            IT = min(IT, 299)
            Object.TIME[IT] += 1
            Object.SPEC[IE] += 1
            Object.WZ = Object.Z / Object.ST

            SUMVX = SUMVX + DX ** 2
            SUMVY = SUMVY + DY ** 2

            if ID != 0:
                KDUM = 0
                for JDUM in range(int(Object.NCORST)):
                    ST2 = ST2 + T
                    NCOLDM = NCOL + KDUM
                    if NCOLDM > Object.NCOLM:
                        NCOLDM = NCOLDM - Object.NCOLM
                    SDIF = Object.ST - STO[NCOLDM-1]
                    SUMXX += ((Object.X - XST[NCOLDM-1]) ** 2) * T / SDIF
                    SUMYY += ((Object.Y - YST[NCOLDM-1]) ** 2) * T / SDIF
                    KDUM += Object.NCORLN
                    if J1 >= 2:
                        ST1 += T
                        SUMZZ += ((Object.Z - ZST[NCOLDM-1] - Object.WZ * SDIF) ** 2) * T / SDIF
            XST[NCOL-1] = Object.X
            YST[NCOL-1] = Object.Y
            ZST[NCOL-1] = Object.Z
            STO[NCOL-1] = Object.ST
            if NCOL >= Object.NCOLM:
                ID += 1
                Object.XID = float(ID)
                NCOL = 0

            R2 = random_uniform(RDUM)

            I = SORT(I, R2, IE, Object)
            while Object.CFNT[IE][I] < R2:
                I = I + 1

            S1 = Object.RGASNT[I]
            EI = Object.EINNT[I]

            if Object.IPNNT[I] > 0:
                R9 = random_uniform(RDUM)
                EXTRA = R9 * (E - EI)
                EI = EXTRA + EI
                IEXTRA += <long long>(Object.NC0NT[I])
            IPT = <long long>(Object.IARRYNT[I])
            Object.ICOLLNT[int(IPT)] += 1
            Object.ICOLNNT[I] += 1
            if E < EI:
                EI = E - 0.0001

            if Object.IPEN != 0:
                if Object.PENFRANT[0][I] != 0:
                    RAN = random_uniform(RDUM)
                    if RAN <= Object.PENFRANT[0][I]:
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

            R3 = random_uniform(RDUM)
            if Object.INDEXNT[I] == 1:
                R31 = random_uniform(RDUM)
                F3 = 1.0 - R3 * Object.ANGCTNT[IE][I]
                if R31 > Object.PSCTNT[IE][I]:
                    F3 = -1 * F3
            elif Object.INDEXNT[I] == 2:
                EPSI = Object.PSCTNT[IE][I]
                F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
            else:
                F3 = 1 - 2 * R3
            THETA0 = acos(F3)
            R4 = random_uniform(RDUM)
            PHI0 = F4 * R4
            F8 = sin(PHI0)
            F9 = cos(PHI0)
            ARG1 = 1 - S1 * EI / E
            ARG1 = max(ARG1, Object.SMALL)
            D = 1 - F3 * sqrt(ARG1)
            E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)
            E1 = max(E1, Object.SMALL)
            Q = sqrt((E / E1) * ARG1) / S1
            Q = min(Q, 1)
            Object.THETA = asin(Q * sin(THETA0))
            F6 = cos(Object.THETA)
            U = (S1 - 1) * (S1 - 1) / ARG1
            CSQD = F3 * F3
            if F3 < 0 and CSQD > U:
                F6 = -1 * F6
            F5 = sin(Object.THETA)
            DCZ2 = min(DCZ2, 1)
            VTOT = CONST9 * sqrt(E1)
            ARGZ = sqrt(DCX2 * DCX2 + DCY2 * DCY2)
            if ARGZ == 0:
                DCZ1 = F6
                DCX1 = F9 * F5
                DCY1 = F8 * F5
            else:
                DCZ1 = DCZ2 * F6 + ARGZ * F5 * F8
                DCY1 = DCY2 * F6 + (F5 / ARGZ) * (DCX2 * F9 - DCY2 * DCZ2 * F8)
                DCX1 = DCX2 * F6 - (F5 / ARGZ) * (DCY2 * F9 + DCX2 * DCZ2 * F8)
            CX1 = DCX1 * VTOT
            CY1 = DCY1 * VTOT
            CZ1 = DCZ1 * VTOT

        Object.WZ *= 1e9
        Object.AVE = SUME2 / Object.ST
        if Object.NISO == 0:
            Object.DIFXX = 5e15 * SUMVX / Object.ST
            Object.DIFYY = 5e15 * SUMVY / Object.ST
            DFXXST[J1] = 5e15 * (SUMVX - SVXOLD) / (Object.ST - STOLD)
            DFYYST[J1] = 5e15 * (SUMVY - SVYOLD) / (Object.ST - STOLD)
        else:
            if ST2 != 0.0:
                Object.DIFYY = 5e15 * SUMYY / ST2
                Object.DIFXX = 5e15 * SUMXX / ST2
                DFXXST[J1] = 5e15 * (SUMXX - SXXOLD) / (ST2 - ST2OLD)
                DFYYST[J1] = 5e15 * (SUMYY - SYYOLD) / (ST2 - ST2OLD)
            else:
                DFXXST[J1] = 0.0
                DFYYST[J1] = 0.0
        if ST1 != 0.0:
            Object.DIFZZ = 5e15 * SUMZZ / ST1
            DFZZST[J1] = 5e15 * (SUMZZ - SZZOLD) / (ST1 - ST1OLD)
        else:
            DFZZST[J1] = 0.0
        WZST[J1] = (Object.Z - ZOLD) / (Object.ST - STOLD) * 1e9
        AVEST[J1] = (SUME2 - SME2OLD) / (Object.ST - STOLD)
        ZOLD = Object.Z
        STOLD = Object.ST
        ST1OLD = ST1
        ST2OLD = ST2
        SVXOLD = SUMVX
        SVYOLD = SUMVY
        SZZOLD = SUMZZ
        SYYOLD = SUMYY
        SXXOLD = SUMXX
        SME2OLD = SUME2
        if Object.OF:
            print('{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(Object.WZ, Object.Z, Object.ST,
                                                                                    Object.AVE, Object.DIFXX, Object.DIFYY,
                                                                                    Object.DIFZZ))
    TWZST = 0.0
    TAVE = 0.0
    T2WZST = 0.0
    T2AVE = 0.0
    TYYST = 0.0
    TXXST = 0.0

    TZZST = 0.0
    T2ZZST = 0.0
    T2YYST = 0.0
    T2XXST = 0.0
    for K in range(10):
        TWZST = TWZST + WZST[K]
        TAVE = TAVE + AVEST[K]
        T2WZST = T2WZST + WZST[K] * WZST[K]
        T2AVE = T2AVE + AVEST[K] * AVEST[K]
        TXXST += DFXXST[K]
        TYYST += DFYYST[K]
        T2XXST += DFXXST[K] ** 2
        T2YYST += DFYYST[K] ** 2
        if K >= 2:
            TZZST = TZZST + DFZZST[K]
            T2ZZST += DFZZST[K] ** 2
    Object.DWZ = 100 * sqrt((T2WZST - TWZST * TWZST / 10.0) / 9.0) / Object.WZ
    Object.DEN = 100 * sqrt((T2AVE - TAVE * TAVE / 10.0) / 9.0) / Object.AVE
    Object.DXXER = 100 * sqrt((T2XXST - TXXST * TXXST / 10.0) / 9.0) / Object.DIFXX
    Object.DYYER = 100 * sqrt((T2YYST - TYYST * TYYST / 10.0) / 9.0) / Object.DIFYY
    Object.DZZER = 100 * sqrt((T2ZZST - TZZST * TZZST / 8.0) / 7.0) / Object.DIFZZ
    Object.DWZ = Object.DWZ / sqrt(10)
    Object.DEN = Object.DEN / sqrt(10)
    Object.DXXER = Object.DXXER / sqrt(10)
    Object.DYYER = Object.DYYER / sqrt(10)
    Object.DZZER = Object.DZZER / sqrt(8)
    Object.DIFLN = Object.DIFZZ
    Object.DIFTR = (Object.DIFXX + Object.DIFYY) / 2
    # CONVERT CM/SEC
    Object.WZ *= 1e5
    Object.DFLER = Object.DZZER
    Object.DFTER = (Object.DXXER + Object.DYYER) / 2.0

    ANCATT = 0.0
    ANCION = 0.0
    for I in range(Object.NGAS):
        ANCATT += Object.ICOLLNT[5 * (I + 1) - 3]
        ANCION += Object.ICOLLNT[5 * (I + 1) - 4]
    ANCION += IEXTRA
    Object.ATTER = 0.0
    if ANCATT != 0:
        Object.ATTER = 100 * sqrt(ANCATT) / ANCATT
    Object.ATT = ANCATT / (Object.ST * Object.WZ) * 1e12
    Object.ALPER = 0.0
    if ANCION != 0:
        Object.ALPER = 100 * sqrt(ANCION) / ANCION
    Object.ALPHA = ANCION / (Object.ST * Object.WZ) * 1e12

    return



