from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow
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
cdef void GERJAN(double RandomSeed, double *RNMX):
    cdef double RAN1, RAN2, TWOPI
    cdef int J
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RandomSeed)
        RAN2 = random_uniform(RandomSeed)
        TWOPI = 2.0 * np.pi
        RNMX[J] = sqrt(-1*log(RAN1)) * cos(RAN2 * TWOPI)
        RNMX[J + 1] = sqrt(-1*log(RAN1)) * sin(RAN2 * TWOPI)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run(PyBoltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at temp =  TemperatureCentigrade.

    This function is used when the magnetic field is parallel to the electric field in the z direction.    
    
    The object parameter is the PyBoltz object to have the output results and to be used in the simulation.
    """
    Object.VelocityX = 0.0
    Object.VelocityY = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.TimeSum = 0.0
    cdef long long I, ID, NCOL, IEXTRA, IMBPT, K, J, iCollisionM, iSample, iCollision, GasIndex, IE, IT, KDUM, IPT, JDUM,NC_LastSampleM
    cdef double ST1, RandomSeed,ST2, SUME2, SUMXX, SUMYY, SUMZZ, SUMVX, SUMVY, Z_LastSample, ST_LastSample, ST1_LastSample, ST2_LastSample, SZZ_LastSample, SXX_LastSample, SYY_LastSample, SVX_LastSample, SVY_LastSample, SME2_LastSample, TDash
    cdef double ABSFAKEI, DCZ1, DCX1, DCY1, CX1, CY1, CZ1, BP, F1, F2, F4, DCX2, DCY2, DCZ2, CX2, CY2, CZ2, DZCOM, DYCOM, DXCOM, THETA0,
    cdef double  E1, Sqrt2M, TwoM, AP, CONST6, R2, R1, VGX, VGY, VGZ, VEX, VEY, VEZ, COMEnergy, R5, TEST1, TEST2, TEST3, CONST11
    cdef double T2, A, B, CONST7, R3, S1, EI, R9, EXTRA, RAN, R31, F3, EPSI, R4, PHI0, F8, F9, ARG1, D, Q, F6, U, CSQD, F5, VXLAB, VYLAB, VZLAB
    cdef double TWZST, TAVE, T2WZST, T2AVE, TXXST, TYYST, T2XXST, T2YYST, TZZST, T2ZZST, Attachment, Ionization, E
    cdef double NumSamples
    ST1 = 0.0
    ST2 = 0.0
    SUME2 = 0.0
    SUMXX = 0.0
    SUMYY = 0.0
    SUMZZ = 0.0
    SUMVX = 0.0
    SUMVY = 0.0
    Z_LastSample = 0.0
    ST_LastSample = 0.0
    ST1_LastSample = 0.0
    ST2_LastSample = 0.0
    SZZ_LastSample = 0.0
    SXX_LastSample = 0.0
    SYY_LastSample = 0.0
    I=0
    SVX_LastSample = 0.0
    SVY_LastSample = 0.0
    SME2_LastSample = 0.0
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
    RandomSeed = Object.RandomSeed
    E1 = Object.InitialElectronEnergy
    Sqrt2M = Object.CONST3 * 0.01
    TwoM = Sqrt2M ** 2
    NumSamples = 10
    ID = 0
    NCOL = 0
    IEXTRA = 0
    GERJAN(Object.RandomSeed,  Object.RNMX)
    IMBPT = 0
    TDash = 0.0
    cdef double ** TEMP = <double **> malloc(6 * sizeof(double *))
    for i in range(6):
        TEMP[i] = <double *> malloc(4000 * sizeof(double))
    for K in range(6):
        for J in range(4000):
            TEMP[K][J] = Object.TotalCollisionFrequency[K][J] + Object.TotalCollisionFrequencyN[K][J]

    ABSFAKEI = Object.FAKEI
    Object.FakeIonizations = 0

    # INITIAL DIRECTION COSINES
    DCZ1 = cos(Object.AngleFromZ)
    DCX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DCY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    VTOT = Sqrt2M * sqrt(E1)
    CX1 = DCX1 * VTOT
    CY1 = DCY1 * VTOT
    CZ1 = DCZ1 * VTOT
    BP = Object.EField ** 2 * Object.CONST1
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3
    F4 = 2 * acos(-1)
    iCollisionM = <long long>(Object.MaxNumberOfCollisions / NumSamples)
    if Object.ConsoleOutputFlag:
        print('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity", "Position", "Time", "Energy",
                                                                       "DIFXX", "DIFYY", "DIFZZ"))

    for iSample in range(int(NumSamples)):
        for iCollision in range(int(iCollisionM)):
            while True:
                R1 = random_uniform(RandomSeed)
                T = -1 * log(R1) / Object.MaxCollisionFreqTotal + TDash
                Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T
                TDash = T
                AP = DCZ1 * F2 * sqrt(E1)
                E = E1 + (AP + BP * T) * T
                WBT = Object.AngularSpeedOfRotation * T
                COSWT = cos(WBT)
                SINWT = sin(WBT)
                CONST6 = sqrt(E1 / E)
                # FIND IDENTITY OF GAS FOR COLLISION
                GasIndex = 0
                R2 = random_uniform(RandomSeed)
                if Object.NumberOfGases == 1:
                    GasIndex = 0
                else:
                    while (Object.MaxCollisionFreqTotalG[GasIndex] < R2):
                        GasIndex = GasIndex + 1
                # CALCULATE ELECTRON VELOCITY VECTORS BEFORE COLLISION
                CX2 = CX1 * COSWT - CY1 * SINWT
                CY2 = CY1 * COSWT + CX1 * SINWT
                VTOT = Sqrt2M * sqrt(E)
                CZ2 = VTOT * (DCZ1 * CONST6 + Object.EField * T * Object.CONST5 / sqrt(E))

                # CALCULATE GAS VELOCITY VECTORS VGX,VGY,VGZ
                IMBPT += 1
                if (IMBPT > 6):
                    GERJAN(Object.RandomSeed, Object.RNMX)
                    IMBPT = 1
                VGX = Object.VTMB[GasIndex] * Object.RNMX[(IMBPT - 1) % 6]
                IMBPT += 1
                VGY = Object.VTMB[GasIndex] * Object.RNMX[(IMBPT - 1) % 6]
                IMBPT += 1
                VGZ = Object.VTMB[GasIndex] * Object.RNMX[(IMBPT - 1) % 6]

                # CALCULATE ENERGY WITH STATIONRhydbergConst GAS TARGET
                COMEnergy = ((CX2 - VGX) ** 2 + (CY2 - VGY) ** 2 + (CZ2 - VGZ) ** 2) / TwoM
                IE = int(COMEnergy / Object.ElectronEnergyStep)
                IE = min(IE, 3999)

                R5 = random_uniform(RandomSeed)
                TEST1 = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]

                # TEST FOR REAL OR NULL COLLISION
                if R5 > TEST1:
                    TEST2 = TEMP[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
                    if R5 < TEST2:
                        # TEST FOR NULL LEVELS
                        if Object.NPLAST[GasIndex] == 0:
                            continue
                        R2 = random_uniform(RandomSeed)
                        I = 0
                        while Object.NullCollisionFreq[GasIndex][IE][I] < R2:
                            # INCREMENT NULL SCATTER SUM
                            I += 1

                        Object.ICOLNN[GasIndex][I] += 1
                        continue
                    else:
                        TEST3 = (TEMP[GasIndex][IE] + ABSFAKEI) / Object.MaxCollisionFreq[GasIndex]
                        if R5 < TEST3:
                            # FAKE IONISATION INCREMENT COUNTER
                            Object.FakeIonizations += 1
                            continue
                        continue
                else:
                    break
            NCOL += 1
            #CALCULATE DIRECTION COSINES OF ELECTRON IN O KELVIN FRAME
            CONST11 = 1 / (Sqrt2M * sqrt(COMEnergy))
            DXCOM = (CX2 - VGX) * CONST11
            DYCOM = (CY2 - VGY) * CONST11
            DZCOM = (CZ2 - VGZ) * CONST11
            #  CALCULATE POSITIONS AT INSTANT BEFORE COLLISION IN LAB FRAME
            #    ALSO UPDATE DIFFUSION  AND ENERGY CALCULATIONS.
            T2 = T ** 2

            TDash = 0.0
            A = AP * T
            B = BP * T2
            SUME2 = SUME2 + T * (E1 + A / 2.0 + B / 3.0)
            CONST7 = Sqrt2M * sqrt(E1)
            A = T * CONST7

            DX = (CX1 * SINWT - CY1 * (1 - COSWT)) / Object.AngularSpeedOfRotation
            Object.X += DX
            DY = (CY1 * SINWT + CX1 * (1 - COSWT)) / Object.AngularSpeedOfRotation
            Object.Y += DY

            Object.Z += DCZ1 * A + T2 * F1
            Object.TimeSum += T
            IT = int(T)
            IT = min(IT, 299)
            Object.CollisionTimes[IT] += 1
            # ENERGY CollisionEnergiesTRUM IN 0 KELVIN FRAME
            Object.CollisionEnergies[IE] += 1
            Object.VelocityZ = Object.Z / Object.TimeSum
            SUMVX = SUMVX + DX ** 2
            SUMVY = SUMVY + DY ** 2

            if ID != 0:
                KDUM = 0
                for JDUM in range(int(Object.Decor_NCORST)):
                    ST2 = ST2 + T
                    NC_LastSampleM = NCOL + KDUM
                    if NC_LastSampleM > Object.Decor_NCOLM:
                        NC_LastSampleM = NC_LastSampleM - Object.Decor_NCOLM
                    SDIF = Object.TimeSum - STO[NC_LastSampleM]
                    SUMXX += ((Object.X - XST[NC_LastSampleM]) ** 2) * T / SDIF
                    SUMYY += ((Object.Y - YST[NC_LastSampleM]) ** 2) * T / SDIF
                    KDUM += Object.Decor_NCORLN
                    if iSample >= 2:
                        ST1 += T
                        SUMZZ += ((Object.Z - ZST[NC_LastSampleM] - Object.VelocityZ * SDIF) ** 2) * T / SDIF
            XST[NCOL] = Object.X
            YST[NCOL] = Object.Y
            ZST[NCOL] = Object.Z
            STO[NCOL] = Object.TimeSum
            if NCOL >= Object.Decor_NCOLM:
                ID += 1
                NCOL = 0

            # ---------------------------------------------------------------------
            #     DETERMINATION OF REAL COLLISION TYPE
            # ---------------------------------------------------------------------
            R2 = random_uniform(RandomSeed)


            # FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
            I = MBSortT(GasIndex, I, R3, IE, Object)
            while Object.CF[GasIndex][IE][I] < R3:
                I += 1

            S1 = Object.RGAS[GasIndex][I]
            EI = Object.EIN[GasIndex][I]

            if Object.IPN[GasIndex][I] > 0:
                # USE FLAT DISTRIBUTION OF  ELECTRON ENERGY BETWEEN E-EION AND 0.0 EV
                # SAME AS IN BOLTZMANN
                R9 = random_uniform(RandomSeed)
                EXTRA = R9 * (COMEnergy - EI)
                EI = EXTRA + EI
                # IF FLUORESCENCE OR AUGUR ADD EXTRA ELEDCTRONS
                IEXTRA += <long long>Object.NC0[GasIndex][I]

            #  GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
            #   COLLISION ALSO UPDATE ENERGY OF ELECTRON.
            IPT = <long long>Object.IARRY[GasIndex][I]
            Object.ICOLL[GasIndex][int(IPT)] += 1
            Object.ICOLN[GasIndex][I] += 1
            if COMEnergy < EI:
                EI = COMEnergy - 0.0001

            # IF EXCITATION THEN ADD PROBABLITY,PenningFractionC(1,I),OF TRANSFER TO GIVE
            # IONISATION OF THE OTHER GASES IN THE MIXTURE.
            if Object.EnablePenning != 0:
                if Object.PenningFraction[GasIndex][0][I] != 0:
                    RAN = random_uniform(RandomSeed)
                    if RAN <= Object.PenningFraction[GasIndex][0][I]:
                        # ADD EXTRA IONISATION COLLISION
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)
            # AAnisotropicDetectedTROPIC SCATTERING
            R3 = random_uniform(RandomSeed)
            if Object.INDEX[GasIndex][I] == 1:
                R31 = random_uniform(RandomSeed)
                F3 = 1.0 - R3 * Object.ANGCT[GasIndex][IE][I]
                if R31 > Object.PSCT[GasIndex][IE][I]:
                    F3 = -1 * F3
            elif Object.INDEX[GasIndex][I] == 2:
                EPSI = Object.PSCT[GasIndex][IE][I]
                F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
            else:
                # ISOTROPIC SCATTERING
                F3 = 1 - 2 * R3
            THETA0 = acos(F3)
            R4 = random_uniform(RandomSeed)
            PHI0 = F4 * R4
            F8 = sin(PHI0)
            F9 = cos(PHI0)
            ARG1 = 1 - S1 * EI / COMEnergy
            ARG1 = max(ARG1, Object.SmallNumber)
            D = 1 - F3 * sqrt(ARG1)
            E1 = COMEnergy * (1 - EI / (S1 * COMEnergy) - 2 * D / S2)
            E1 = max(E1, Object.SmallNumber)
            Q = sqrt((COMEnergy / E1) * ARG1) / S1
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
            VTOT = Sqrt2M * sqrt(E1)
            CX1 = DCX1 * VTOT + VGX
            CY1 = DCY1 * VTOT + VGY
            CZ1 = DCZ1 * VTOT + VGZ

            # CALCULATE ENERGY AND DIRECTION IN LAB FRAME
            E1 = (CX1 * CX1 + CY1 * CY1 + CZ1 * CZ1) / TwoM
            CONST11 = 1 / (Sqrt2M * sqrt(E1))
            DCX1 = CX1 * CONST11
            DCY1 = CY1 * CONST11
            DCZ1 = CZ1 * CONST11

        Object.VelocityZ *= 1e9
        Object.MeanElectronEnergy = SUME2 / Object.TimeSum
        if Object.AnisotropicDetected == 0:
            Object.DiffusionX = 5e15 * SUMVX / Object.TimeSum
            Object.DiffusionY = 5e15 * SUMVY / Object.TimeSum
            DFXXST[iSample] = 5e15 * (SUMVX - SVX_LastSample) / (Object.TimeSum - ST_LastSample)
            DFYYST[iSample] = 5e15 * (SUMVY - SVY_LastSample) / (Object.TimeSum - ST_LastSample)
        else:
            if ST2 != 0.0:
                Object.DiffusionY = 5e15 * SUMYY / ST2
                Object.DiffusionX = 5e15 * SUMXX / ST2
                DFXXST[iSample] = 5e15 * (SUMXX - SXX_LastSample) / (ST2 - ST2_LastSample)
                DFYYST[iSample] = 5e15 * (SUMYY - SYY_LastSample) / (ST2 - ST2_LastSample)
            else:
                DFXXST[iSample] = 0.0
                DFYYST[iSample] = 0.0

        if ST1 != 0.0:
            Object.DiffusionZ = 5e15 * SUMZZ / ST1
            DFZZST[iSample] = 5e15 * (SUMZZ - SZZ_LastSample) / (ST1 - ST1_LastSample)
        else:
            DFZZST[iSample] = 0.0
        WZST[iSample] = (Object.Z - Z_LastSample) / (Object.TimeSum - ST_LastSample) * 1e9
        AVEST[iSample] = (SUME2 - SME2_LastSample) / (Object.TimeSum - ST_LastSample)
        Z_LastSample = Object.Z
        ST_LastSample = Object.TimeSum
        ST1_LastSample = ST1
        ST2_LastSample = ST2
        SVX_LastSample = SUMVX
        SVY_LastSample = SUMVY
        SZZ_LastSample = SUMZZ
        SYY_LastSample = SUMYY
        SXX_LastSample = SUMXX
        SME2_LastSample = SUME2
        if Object.ConsoleOutputFlag:
            print('{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(Object.VelocityZ, Object.Z, Object.TimeSum,
                                                                                    Object.MeanElectronEnergy, Object.DiffusionX, Object.DiffusionY,
                                                                                    Object.DiffusionZ))
    #CALCULATE ERRORS AND CHECK AVERAGES
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
    Object.VelocityErrorZ = 100 * sqrt((T2WZST - TWZST * TWZST / 10.0) / 9.0) / Object.VelocityZ
    Object.MeanElectronEnergyError = 100 * sqrt((T2AVE - TAVE * TAVE / 10.0) / 9.0) / Object.MeanElectronEnergy
    Object.ErrorDiffusionX = 100 * sqrt((T2XXST - TXXST * TXXST / 10.0) / 9.0) / Object.DiffusionX
    Object.ErrorDiffusionY = 100 * sqrt((T2YYST - TYYST * TYYST / 10.0) / 9.0) / Object.DiffusionY
    Object.ErrorDiffusionZ = 100 * sqrt((T2ZZST - TZZST * TZZST / 8.0) / 7.0) / Object.DiffusionZ
    Object.VelocityErrorZ = Object.VelocityErrorZ / sqrt(10)
    Object.MeanElectronEnergyError = Object.MeanElectronEnergyError / sqrt(10)
    Object.ErrorDiffusionX = Object.ErrorDiffusionX / sqrt(10)
    Object.ErrorDiffusionY = Object.ErrorDiffusionY / sqrt(10)
    Object.ErrorDiffusionZ = Object.ErrorDiffusionZ / sqrt(8)
    Object.LongitudinalDiffusion = Object.DiffusionZ
    Object.TransverseDiffusion = (Object.DiffusionX + Object.DiffusionY) / 2
    # CONVERT CM/SEC
    Object.VelocityZ *= 1e5
    Object.LongitudinalDiffusionError = Object.ErrorDiffusionZ
    Object.TransverseDiffusionError = (Object.ErrorDiffusionX + Object.ErrorDiffusionY) / 2.0

    #CALCULATE TOWNSEND COEFICIENTS AND ERRORS
    Attachment = 0.0
    Ionization = 0.0
    for I in range(Object.NumberOfGases):
        Attachment += Object.ICOLL[I][2]
        Ionization += Object.ICOLL[I][1]
    Ionization += IEXTRA
    Object.AttachmentRateError = 0.0

    if Attachment != 0:
        Object.AttachmentRateError = 100 * sqrt(Attachment) / Attachment
    Object.AttachmentRate = Attachment / (Object.TimeSum * Object.VelocityZ) * 1e12
    Object.IonisationRateError = 0.0
    if Ionization != 0:
        Object.IonisationRateError = 100 * sqrt(Ionization) / Ionization
    Object.IonisationRate = Ionization / (Object.TimeSum * Object.VelocityZ) * 1e12


