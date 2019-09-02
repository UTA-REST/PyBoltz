from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
from libc.string cimport memset
from PyBoltz cimport drand48
from MBSorts cimport MBSortT
import numpy as np
cimport numpy as np
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
cdef void GERJAN(double RDUM,  double *RNMX):
    cdef double RAN1, RAN2, TWOPI
    cdef int J
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RDUM)
        RAN2 = random_uniform(RDUM)
        TWOPI = 2.0 * np.pi
        RNMX[J] = sqrt(-1 * log(RAN1)) * cos(RAN2 * TWOPI)
        RNMX[J + 1] = sqrt(-1 * log(RAN1)) * sin(RAN2 * TWOPI)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at temp =  TemperatureCentigrade.

    This function is used for any magnetic field electric field in the z direction.    
    
    The object parameter is the PyBoltz object to have the output results and to be used in the simulation.
    """
    Object.VelocityX = 0.0
    Object.VelocityErrorX = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    cdef long long I, ID,  NCOL, IEXTRA, IMBPT, K, J, J2M, J1, J2, KGAS, IE, IT, KDUM, IPT, JDUM, NCOLDM
    cdef double ST1, RDUM, ST2, SUME2, SUMXX, SUMYY, SUMZZ, SUMXZ, SUMXY, ZOLD, STOLD, ST1OLD, ST2OLD, SZZOLD, SXXOLD, SYYOLD, SYZOLD, SXYOLD, SXZOLD, SME2OLD, TDASH
    cdef double ABSFAKEI, DCZ1, DCX1, DCY1, CX1, CY1, CZ1, BP, F1, F2, F4, DCX2, DCY2, DCZ2, CX2, CY2, CZ2, DZCOM, DYCOM, DXCOM, THETA0,
    cdef double  E1, CONST9, CONST10, AP, CONST6, R2, R1, VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, R5, TEST1, TEST2, TEST3, CONST11
    cdef double T2, A, B, CONST7, R3, S1, EI, R9, EXTRA, RAN, R31, F3, EPSI, R4, PHI0, F8, F9, ARG1, D, Q, F6, U, CSQD, F5, VXLAB, VYLAB, VZLAB
    cdef double TWZST, TAVE, T2WZST, T2AVE, TXXST, TYYST, TZZST, TXYST, TXZST, TYZST, T2XXST, T2YYST, T2ZZST, T2XYST, T2XZST, T2YZST, ANCATT, ANCION, E, SUMYZ, SUMLS, SUMTS
    cdef double SLNOLD, STROLD, EBAROLD, EFZ100, EFX100, EBAR, WZR, WYR, WXR, XR, ZR, YR, TWYST, TWXST, T2WYST, T2WXST
    cdef double *STO, *XST, *YST, *ZST, *WZST, *AVEST, *DFZZST, *DFYYST, *DFXXST, *DFYZST, *DFXYST, *DFXZST, *WYZST, *WXZST
    cdef double DIFXXR, DIFYYR, DIFZZR, DIFYZR, DIFXZR, DIFXYR, ZROLD, YROLD, XROLD, SZZR, SYYR, SXXR, SXYR, SXZR, RCS, RSN, RTHETA, EOVBR

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

    WYST = <double *> malloc(10 * sizeof(double))
    memset(WYST, 0, 10 * sizeof(double))

    WXST = <double *> malloc(10 * sizeof(double))
    memset(WXST, 0, 10 * sizeof(double))

    AVEST = <double *> malloc(10 * sizeof(double))
    memset(AVEST, 0, 10 * sizeof(double))

    DFZZST = <double *> malloc(10 * sizeof(double))
    memset(DFZZST, 0, 10 * sizeof(double))

    DFYYST = <double *> malloc(10 * sizeof(double))
    memset(DFYYST, 0, 10 * sizeof(double))

    DFXXST = <double *> malloc(10 * sizeof(double))
    memset(DFXXST, 0, 10 * sizeof(double))

    DFYZST = <double *> malloc(10 * sizeof(double))
    memset(DFYZST, 0, 10 * sizeof(double))

    DFXYST = <double *> malloc(10 * sizeof(double))
    memset(DFXYST, 0, 10 * sizeof(double))

    DFXZST = <double *> malloc(10 * sizeof(double))
    memset(DFXZST, 0, 10 * sizeof(double))

    DIFXXR = 0.0
    DIFYYR = 0.0
    DIFZZR = 0.0
    DIFYZR = 0.0
    DIFXZR = 0.0
    I = 0
    DIFXYR = 0.0
    Object.TimeSum = 0.0
    ST1 = 0.0
    SUMXX = 0.0
    SUMYY = 0.0
    SUMZZ = 0.0
    SUMYZ = 0.0
    SUMXY = 0.0
    SUMXZ = 0.0
    ZROLD = 0.0
    YROLD = 0.0
    XROLD = 0.0
    SZZR = 0.0
    SYYR = 0.0
    SXXR = 0.0
    SXYR = 0.0
    SYZR = 0.0
    SXZR = 0.0
    STOLD = 0.0
    ST1OLD = 0.0
    ST2OLD = 0.0
    SZZOLD = 0.0
    SYYOLD = 0.0
    SXXOLD = 0.0
    SYZOLD = 0.0
    SXYOLD = 0.0
    SXZOLD = 0.0

    EBAROLD = 0.0
    Object.SmallNumber = 1e-20
    Object.MaximumCollisionTime = 0.0

    # CALC ROTATION MATRIX ANGLES
    RCS = cos((Object.BFieldAngle - 90) * np.pi / 180)
    RSN = sin((Object.BFieldAngle - 90) * np.pi / 180)

    RTHETA = Object.BFieldAngle * np.pi / 180
    EFZ100 = Object.EField * 100 * sin(RTHETA)
    EFX100 = Object.EField * 100 * cos(RTHETA)

    F1 = Object.EField * Object.CONST2 * sin(RTHETA)
    F4 = 2 * np.pi
    CONST9 = Object.CONST3 * 0.01
    CONST10 = CONST9 ** 2
    EOVBR = Object.EFieldOverBField * sin(RTHETA)
    E1 = Object.InitialElectronEnergy

    Object.ITMAX = 10
    ID = 0
    NCOL = 0
    Object.NNULL = 0
    IEXTRA = 0
    cdef double ** TEMP = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TEMP[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TEMP[K][J] = Object.TCF[K][J] + Object.TCFN[K][J]
    GERJAN(Object.RSTART,  Object.RNMX)
    ABSFAKEI = Object.FAKEI
    Object.IFAKE = 0

    IMBPT = 0
    TDASH = 0.0

    # INTIAL DIRECTION COSINES
    DCZ1 = cos(Object.AngleFromZ)
    DCX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DCY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    # INITIAL VELOCITY
    VTOT = CONST9 * sqrt(E1)
    CX1 = DCX1 * VTOT
    CY1 = DCY1 * VTOT
    CZ1 = DCZ1 * VTOT
    RDUM = Object.RSTART

    J2M = <long long>(Object.MaxNumberOfCollisions / Object.ITMAX)
    if Object.ConsoleOutputFlag:
        print('{:^12s}{:^12s}{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity Z", "Velocity Y", "Velocity X","Energy",
                                                                       "DIFXX", "DIFYY", "DIFZZ", "DIFYZ","DIFXZ","DIFXY"))
    for J1 in range(int(Object.ITMAX)):
        for J2 in range(int(J2M)):
            while True:
                R1 = random_uniform(RDUM)
                T = -1 * log(R1) / Object.TCFMX + TDASH
                Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T
                TDASH = T
                WBT = Object.AngularSpeedOfRotation * T
                COSWT = cos(WBT)
                SINWT = sin(WBT)
                DZ = (CZ1 * SINWT + (EOVBR - CY1) * (1 - COSWT)) / Object.AngularSpeedOfRotation

                DX = CX1 * T + F1 * T * T

                E = E1 + DZ * EFZ100 + DX * EFX100

                # CALCULATE ELECTRON VELOCITY IN LAB FRAME
                CX2 = CX1 + 2 * F1 * T
                CY2 = (CY1 - EOVBR) * COSWT + CZ1 * SINWT + EOVBR
                CZ2 = CZ1 * COSWT - (CY1 - EOVBR) * SINWT

                # FIND IDENTITY OF GAS FOR COLLISION
                KGAS = 0
                R2 = random_uniform(RDUM)
                if Object.NumberOfGases == 1:
                    KGAS = 0
                else:
                    while (Object.TCFMXG[KGAS] < R2):
                        KGAS = KGAS + 1

                IMBPT += 1
                if (IMBPT > 6):
                    GERJAN(Object.RSTART,  Object.RNMX)
                    IMBPT = 1
                VGX = Object.VTMB[KGAS] * Object.RNMX[(IMBPT - 1) % 6]
                IMBPT += 1
                VGY = Object.VTMB[KGAS] * Object.RNMX[(IMBPT - 1) % 6]
                IMBPT += 1
                VGZ = Object.VTMB[KGAS] * Object.RNMX[(IMBPT - 1) % 6]

                # CALCULATE ENERGY WITH STATIONARY GAS TARGET,EOK
                EOK = ((CX2 - VGX) ** 2 + (CY2 - VGY) ** 2 + (CZ2 - VGZ) ** 2) / CONST10
                IE = int(EOK / Object.ElectronEnergyStep)
                IE = min(IE, 3999)

                # TEST FOR REAL OR NULL COLLISION
                R5 = random_uniform(RDUM)
                TEST1 = Object.TCF[KGAS][IE] / Object.TCFMAX[KGAS]
                if R5 > TEST1:
                    Object.NNULL += 1
                    TEST2 = TEMP[KGAS][IE] / Object.TCFMAX[KGAS]
                    if R5 < TEST2:
                        # TEST FOR NULL LEVELS
                        if Object.NPLAST[KGAS] == 0:
                            continue
                        R2 = random_uniform(RDUM)
                        I = 0
                        while Object.CFN[KGAS][IE][I] < R2:
                            # INCREMENT NULL SCATTER SUM
                            I += 1

                        Object.ICOLNN[KGAS][I] += 1
                        continue
                    else:
                        TEST3 = (TEMP[KGAS][IE] + ABSFAKEI) / Object.TCFMAX[KGAS]
                        if R5 < TEST3:
                            # FAKE IONISATION INCREMENT COUNTER
                            Object.IFAKE += 1
                            continue
                        continue
                else:
                    break

            NCOL += 1
            # CALCULATE DIRECTION COSINES OF ELECTRON IN 0 KELVIN FRAME
            CONST11 = 1 / (CONST9 * sqrt(EOK))
            #     VTOT=1.0D0/CONST11
            DXCOM = (CX2 - VGX) * CONST11
            DYCOM = (CY2 - VGY) * CONST11
            DZCOM = (CZ2 - VGZ) * CONST11
            #  CALCULATE POSITIONS AT INSTANT BEFORE COLLISION
            #    ALSO UPDATE DIFFUSION  AND ENERGY CALCULATIONS.
            T2 = T ** 2
            if (T >= Object.MaximumCollisionTime):
                Object.MaximumCollisionTime = T
            TDASH = 0.0

            Object.X += DX
            Object.Y += EOVBR * T + ((CY1 - EOVBR) * SINWT + CZ1 * (1 - COSWT)) / Object.AngularSpeedOfRotation
            Object.Z += DZ
            Object.TimeSum += T
            IT = int(T)
            IT = min(IT, 299)
            Object.TIME[IT] += 1
            Object.SPEC[IE] += 1
            Object.VelocityZ = Object.Z / Object.TimeSum
            Object.VelocityY = Object.Y / Object.TimeSum
            Object.VelocityX = Object.X / Object.TimeSum
            if J1 >= 2:
                KDUM = 0
                for J in range(int(Object.NCORST)):
                    NCOLDM = NCOL + KDUM
                    if NCOLDM > Object.NCOLM:
                        NCOLDM = NCOLDM - Object.NCOLM
                    ST1 += T
                    SDIF = Object.TimeSum - STO[NCOLDM]
                    KDUM += Object.NCORLN
                    SUMZZ += ((Object.Z - ZST[NCOLDM] - Object.VelocityZ * SDIF) ** 2) * T / SDIF
                    SUMYY += ((Object.Y - YST[NCOLDM] - Object.VelocityY * SDIF) ** 2) * T / SDIF
                    SUMXX += ((Object.X - XST[NCOLDM] - Object.VelocityX * SDIF) ** 2) * T / SDIF
                    SUMYZ += (Object.Z - ZST[NCOLDM] - Object.VelocityZ * SDIF) * (
                            Object.Y - YST[NCOLDM] - Object.VelocityY * SDIF) * T / SDIF
                    SUMXY += (Object.X - XST[NCOLDM] - Object.VelocityX * SDIF) * (
                            Object.Y - YST[NCOLDM] - Object.VelocityY * SDIF) * T / SDIF
                    SUMXZ += (Object.X - XST[NCOLDM] - Object.VelocityX * SDIF) * (
                            Object.Z - ZST[NCOLDM] - Object.VelocityZ * SDIF) * T / SDIF
            XST[NCOL] = Object.X
            YST[NCOL] = Object.Y
            ZST[NCOL] = Object.Z
            STO[NCOL] = Object.TimeSum

            if NCOL >= Object.NCOLM:
                ID += 1
                NCOL = 0
            # DETERMENATION OF REAL COLLISION TYPE
            R2 = random_uniform(RDUM)

            # FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
            I = MBSortT(KGAS, I, R3, IE, Object)
            while Object.CF[KGAS][IE][I] < R3:
                I += 1

            S1 = Object.RGAS[KGAS][I]
            EI = Object.EIN[KGAS][I]

            if Object.IPN[KGAS][I] > 0:
                #  USE FLAT DISTRIBUTION OF  ELECTRON ENERGY BETWEEN E-EION AND 0.0 EV
                #  SAME AS IN BOLTZMANN
                R9 = random_uniform(RDUM)
                EXTRA = R9 * (EOK - EI)
                EI = EXTRA + EI
                # IF FLOUORESCENCE OR AUGER ADD EXTRA ELECTRONS
                IEXTRA += <long long>Object.NC0[KGAS][I]

            #  GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
            #   COLLISION ALSO UPDATE ENERGY OF ELECTRON.
            IPT = <long long>Object.IARRY[KGAS][I]
            Object.ICOLL[KGAS][int(IPT)] += 1
            Object.ICOLN[KGAS][I] += 1
            if EOK < EI:
                EI = EOK - 0.0001
            #IF EXCITATION THEN ADD PROBABILITY,PENFRAC(1,I), OF TRANSFER TO GIVE
            # IONISATION OF THE OTHER GASES IN THE MIXTURE
            if Object.EnablePenning != 0:
                if Object.PENFRA[KGAS][0][I] != 0:
                    RAN = random_uniform(RDUM)
                    if RAN <= Object.PENFRA[KGAS][0][I]:
                        # ADD EXTRA IONISATION COLLISION
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

            # ANISOTROPIC SCATTERING
            R3 = random_uniform(RDUM)
            if Object.INDEX[KGAS][I] == 1:
                R31 = random_uniform(RDUM)
                F3 = 1.0 - R3 * Object.ANGCT[KGAS][IE][I]
                if R31 > Object.PSCT[KGAS][IE][I]:
                    F3 = -1 * F3
            elif Object.INDEX[KGAS][I] == 2:
                EPSI = Object.PSCT[KGAS][IE][I]
                F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
            else:
                #ISOTROPIC SCATTERING
                F3 = 1 - 2 * R3

            THETA0 = acos(F3)
            R4 = random_uniform(RDUM)
            PHI0 = F4 * R4
            F8 = sin(PHI0)
            F9 = cos(PHI0)
            ARG1 = 1 - S1 * EI / EOK
            ARG1 = max(ARG1, Object.SmallNumber)
            D = 1 - F3 * sqrt(ARG1)
            E1 = EOK * (1 - EI / (S1 * EOK) - 2 * D / S2)
            E1 = max(E1, Object.SmallNumber)
            Q = sqrt((EOK / E1) * ARG1) / S1
            Q = min(Q, 1)
            Object.AngleFromZ = asin(Q * sin(THETA0))
            F6 = cos(Object.AngleFromZ)
            U = (S1 - 1) * (S1 - 1) / ARG1

            CSQD = F3 * F3
            if F3 < 0 and CSQD > U:
                F6 = -1 * F6
            F5 = sin(Object.AngleFromZ)
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

            #TRANSFORM VELOCITY VECTORS TO LAB FRAME
            VTOT = CONST9 * sqrt(E1)
            CX1 = DCX1 * VTOT + VGX
            CY1 = DCY1 * VTOT + VGY
            CZ1 = DCZ1 * VTOT + VGZ
            # CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
            E1 = (CX1 * CX1 + CY1 * CY1 + CZ1 * CZ1) / CONST10
            CONST11 = 1 / (CONST9 * sqrt(E1))
            DCX1 = CX1 * CONST11
            DCY1 = CY1 * CONST11
            DCZ1 = CZ1 * CONST11

        Object.VelocityZ *= 1e9
        Object.VelocityY *= 1e9
        Object.VelocityX *= 1e9

        # CALCULATE ROTATED VECTORS AND POSITIONS
        WZR = Object.VelocityZ * RCS - Object.VelocityX * RSN
        WYR = Object.VelocityY
        WXR = Object.VelocityZ * RSN + Object.VelocityX * RCS
        ZR = Object.Z * RCS - Object.X * RSN
        YR = Object.Y
        XR = Object.Z * RSN + Object.X * RCS
        EBAR = 0.0
        for IK in range(4000):
            TCFSUM = 0.0
            for KI in range(Object.NumberOfGases):
                TCFSUM += Object.TCF[KI][IK]
            EBAR += Object.E[IK] * Object.SPEC[IK] / TCFSUM
        Object.MeanElectronEnergy = EBAR / Object.TimeSum
        WZST[J1] = (ZR - ZROLD) / (Object.TimeSum - STOLD) * 1e9
        WYST[J1] = (YR - YROLD) / (Object.TimeSum - STOLD) * 1e9
        WXST[J1] = (XR - XROLD) / (Object.TimeSum - STOLD) * 1e9
        AVEST[J1] = (EBAR - EBAROLD) / (Object.TimeSum - STOLD)
        EBAROLD = EBAR

        if J1 >= 2:
            Object.DiffusionX = 5e15 * SUMXX / ST1
            Object.DiffusionY = 5e15 * SUMYY / ST1
            Object.DiffusionZ = 5e15 * SUMZZ / ST1
            Object.DiffusionXY = 5e15 * SUMXY / ST1
            Object.DiffusionYZ = 5e15 * SUMYZ / ST1
            Object.DiffusionXZ = 5e15 * SUMXZ / ST1
            # CALCULATE  ROTATED TENSOR
            DIFXXR = Object.DiffusionX * RCS * RCS + Object.DiffusionZ * RSN * RSN + 2 * RCS * RSN * Object.DiffusionXZ
            DIFYYR = Object.DiffusionY
            DIFZZR = Object.DiffusionX * RSN * RSN + Object.DiffusionZ * RCS * RCS - 2 * RCS * RSN * Object.DiffusionXZ
            DIFXYR = RCS * Object.DiffusionXY + RSN * Object.DiffusionYZ
            DIFYZR = RCS * Object.DiffusionXY - RCS * Object.DiffusionYZ
            DIFXZR = (RCS * RCS - RSN * RSN) * Object.DiffusionXZ - RSN * RCS * (Object.DiffusionX - Object.DiffusionZ)

            SXXR = SUMXX * RCS * RCS + SUMZZ * RSN * RSN + 2 * RCS * RSN * SUMXZ
            SYYR = SUMYY
            SZZR = SUMXX * RSN * RSN + SUMZZ * RCS * RCS - 2 * RCS * RSN * SUMXZ
            SXYR = RCS * SUMXY + RSN * SUMYZ
            SYZR = RSN * SUMXY - RCS * SUMYZ
            SXZR = (RCS * RCS - RSN * RSN) * SUMXZ - RSN * RCS * (SUMXX - SUMZZ)
        DFZZST[J1] = 0.0
        DFXXST[J1] = 0.0
        DFYYST[J1] = 0.0
        DFYZST[J1] = 0.0
        DFXZST[J1] = 0.0
        DFXYST[J1] = 0.0
        if J1 > 1:
            DFZZST[J1] = 5e15 * (SZZR - SZZOLD) / (ST1 - ST1OLD)
            DFXXST[J1] = 5e15 * (SXXR - SXXOLD) / (ST1 - ST1OLD)
            DFYYST[J1] = 5e15 * (SYYR - SYYOLD) / (ST1 - ST1OLD)
            DFYZST[J1] = 5e15 * (SYZR - SYZOLD) / (ST1 - ST1OLD)
            DFXZST[J1] = 5e15 * (SXZR - SXZOLD) / (ST1 - ST1OLD)
            DFXYST[J1] = 5e15 * (SXYR - SXYOLD) / (ST1 - ST1OLD)
        ZROLD = ZR
        YROLD = YR
        XROLD = XR
        STOLD = Object.TimeSum
        ST1OLD = ST1
        SZZOLD = SZZR
        SYYOLD = SYYR
        SXXOLD = SXXR
        SXYOLD = SXYR
        SYZOLD = SYZR
        SXZOLD = SXZR
        if Object.ConsoleOutputFlag:
            print('{:^12.1f}{:^12.1f}{:^12.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(WZR,WYR,WXR,
                                                                                    Object.MeanElectronEnergy, DIFXXR, DIFYYR,
                                                                                    DIFZZR,DIFYZR,DIFXZR,DIFXYR))

    #CALCULATE ERRORS AND CHECK AVERAGES

    TWZST = 0.0
    TWYST = 0.0
    TWXST = 0.0
    TAVE = 0.0
    T2WZST = 0.0
    T2WYST = 0.0
    T2WXST = 0.0
    T2AVE = 0.0
    TZZST = 0.0
    TYYST = 0.0
    TXXST = 0.0
    TXYST = 0.0
    TXZST = 0.0
    TYZST = 0.0
    T2ZZST = 0.0
    T2YYST = 0.0
    T2XXST = 0.0
    T2XYST = 0.0
    T2XZST = 0.0
    T2YZST = 0.0

    for K in range(10):
        TWZST = TWZST + WZST[K]
        TWYST = TWYST + WYST[K]
        TWXST = TWXST + WXST[K]
        TAVE = TAVE + AVEST[K]
        T2WZST = T2WZST + WZST[K] * WZST[K]
        T2WYST = T2WYST + WYST[K] * WYST[K]
        T2WXST = T2WXST + WXST[K] * WXST[K]
        T2AVE = T2AVE + AVEST[K] * AVEST[K]
        if K >= 2:
            TZZST = TZZST + DFZZST[K]
            TYYST = TYYST + DFYYST[K]
            TXXST = TXXST + DFXXST[K]
            TYZST = TYZST + DFYZST[K]
            TXYST = TXYST + DFXYST[K]
            TXZST = TXZST + DFXZST[K]

            T2ZZST += DFZZST[K] ** 2
            T2XXST += DFXXST[K] ** 2
            T2YYST += DFYYST[K] ** 2
            T2YZST += DFYZST[K] ** 2
            T2XYST += DFXYST[K] ** 2
            T2XZST += DFXZST[K] ** 2
    Object.VelocityErrorZ = 100 * sqrt((T2WZST - TWZST * TWZST / 10.0) / 9.0) / WZR
    Object.VelocityErrorY = 100 * sqrt((T2WYST - TWYST * TWYST / 10.0) / 9.0) / abs(WYR)
    Object.VelocityErrorX = 100 * sqrt((T2WXST - TWXST * TWXST / 10.0) / 9.0) / abs(WXR)
    Object.MeanElectronEnergyError = 100 * sqrt((T2AVE - TAVE * TAVE / 10.0) / 9.0) / Object.MeanElectronEnergy
    Object.ErrorDiffusionZ = 100 * sqrt((T2ZZST - TZZST * TZZST / 8.0) / 7.0) / DIFZZR
    Object.ErrorDiffusionY = 100 * sqrt((T2YYST - TYYST * TYYST / 8.0) / 7.0) / DIFYYR
    Object.ErrorDiffusionX = 100 * sqrt((T2XXST - TXXST * TXXST / 8.0) / 7.0) / DIFXXR
    Object.ErrorDiffusionXY = 100 * sqrt((T2XYST - TXYST * TXYST / 8.0) / 7.0) / abs(DIFXYR)
    Object.ErrorDiffusionXZ = 100 * sqrt((T2XZST - TXZST * TXZST / 8.0) / 7.0) / abs(DIFXZR)
    Object.ErrorDiffusionYZ = 100 * sqrt((T2YZST - TYZST * TYZST / 8.0) / 7.0) / abs(DIFYZR)

    Object.VelocityErrorZ = Object.VelocityErrorZ / sqrt(10)
    Object.VelocityErrorX = Object.VelocityErrorX / sqrt(10)
    Object.VelocityErrorY = Object.VelocityErrorY / sqrt(10)
    Object.MeanElectronEnergyError = Object.MeanElectronEnergyError / sqrt(10)
    Object.ErrorDiffusionX = Object.ErrorDiffusionX / sqrt(8)
    Object.ErrorDiffusionY = Object.ErrorDiffusionY / sqrt(8)
    Object.ErrorDiffusionZ = Object.ErrorDiffusionZ / sqrt(8)
    Object.ErrorDiffusionYZ = Object.ErrorDiffusionYZ / sqrt(8)
    Object.ErrorDiffusionXY = Object.ErrorDiffusionXY / sqrt(8)
    Object.ErrorDiffusionXZ = Object.ErrorDiffusionXZ / sqrt(8)

    #LOAD ROTATED VALUES INTO ARRAYS

    Object.VelocityZ = WZR
    Object.VelocityX = WXR
    Object.VelocityY = WYR
    Object.DiffusionX = DIFXXR
    Object.DiffusionY = DIFYYR
    Object.DiffusionZ = DIFZZR
    Object.DiffusionYZ = DIFYZR
    Object.DiffusionXY = DIFXYR
    Object.DiffusionXZ = DIFXZR

    #CONVERT TO CM/SEC.
    Object.VelocityZ *= 1e5
    Object.VelocityY *= 1e5
    Object.VelocityX *= 1e5

    #CALCULATE TOWNSEND COEFICIENTS AND ERRORS
    ANCATT = 0.0
    ANCION = 0.0
    for I in range(Object.NumberOfGases):
        ANCATT += Object.ICOLL[I][2]
        ANCION += Object.ICOLL[I][1]
    ANCION += IEXTRA
    Object.AttachmentRateError = 0.0

    if ANCATT != 0:
        Object.AttachmentRateError = 100 * sqrt(ANCATT) / ANCATT
    Object.AttachmentRate = ANCATT / (Object.TimeSum * Object.VelocityZ) * 1e12
    Object.IonisationRateError = 0.0
    if ANCION != 0:
        Object.IonisationRateError = 100 * sqrt(ANCION) / ANCION
    Object.IonisationRate = ANCION / (Object.TimeSum * Object.VelocityZ) * 1e12
