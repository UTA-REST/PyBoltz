from PyBoltz cimport PyBoltz
from libc.math cimport sin, cos, acos, asin, log, sqrt,pow
from libc.string cimport memset
from PyBoltz cimport drand48
from MBSorts cimport MBSort
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
cpdef run(PyBoltz Object):
    """
    This function is used to calculates collision events and updates diffusion and velocity.Background gas motion included at temp =  TemperatureCentigrade.

    This function is used when there is no magnetic field.     
    
    Electric field in z direction.

    The object parameter is the PyBoltz object to have the output results and to be used in the simulation.
    """

    cdef long long I, ID,  NCOL, IEXTRA, IMBPT, K, J, J2M, J1, J2, GasIndex, IE, IT, KDUM, IPT, JDUM,NCOLDM
    cdef double ST1, RandomSeed,ST2, SUME2, SUMXX, SUMYY, SUMZZ, SUMVX, SUMVY, ZOLD, STOLD, ST1OLD, ST2OLD, SZZOLD, SXXOLD, SYYOLD, SVXOLD, SVYOLD, SME2OLD, TDASH
    cdef double ABSFAKEI, DCZ1, DCX1, DCY1, CX1, CY1, CZ1, BP, F1, F2, F4, DCX2, DCY2, DCZ2, CX2, CY2, CZ2, DZCOM, DYCOM, DXCOM, THETA0,
    cdef double  E1, CONST9, CONST10, AP, CONST6, R2, R1, VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, R5, TEST1, TEST2, TEST3, CONST11
    cdef double T2, A, B, CONST7, R3, S1, EI, R9, EXTRA, RAN, R31, F3, EPSI, R4, PHI0, F8, F9, ARG1, D, Q, F6, U, CSQD, F5, VXLAB, VYLAB, VZLAB
    cdef double TWZST, TAVE, T2WZST, T2AVE, TXXST, TYYST, T2XXST, T2YYST, TZZST, T2ZZST, ANCATT, ANCION, E,ARAT,NTPMFLG,TEMP[4000]
    cdef double NumSamples
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

    Object.VelocityX = 0.0
    Object.VelocityY = 0.0
    Object.VelocityErrorX = 0.0
    Object.VelocityErrorY = 0.0
    Object.X = 0.0
    Object.Y = 0.0
    Object.Z = 0.0
    Object.TimeSum = 0.0
    ST1 = 0.0
    ST2 = 0.0
    I=0
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

    Object.SmallNumber = 1.0e-20
    RandomSeed = Object.RandomSeed
    E1 = Object.InitialElectronEnergy
    CONST9 = Object.CONST3 * 0.01

    CONST10 = CONST9 ** 2

    INTEM = 8

    NumSamples = 10
    ID = 0
    NCOL = 0
    IEXTRA = 0
    NTPMFLG = 0.0
    TDASH = 0.0

    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))
    for J in range(4000):
        TEMP[J] = Object.TCFNNT[J] + Object.TCFNT[J]
    ABSFAKEI = abs(Object.FAKEI)
    Object.IFAKE = 0

    #INITIAL DIRECTION COSINES
    DCZ1 = cos(Object.AngleFromZ)
    DCX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DCY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    BP = (Object.EField ** 2) * Object.CONST1
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3
    F4 = 2 * acos(-1)

    J2M = <long long>(Object.MaxNumberOfCollisions / NumSamples)

    DELTAE = Object.FinalElectronEnergy / float(INTEM)
    if Object.ConsoleOutputFlag:
        print('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format("Velocity", "Position", "Time", "Energy",
                                                                       "DIFXX", "DIFYY", "DIFZZ"))
    for J1 in range(int(NumSamples)):
        for J2 in range(int(J2M)):
            while True:
                R1 = random_uniform(RandomSeed)
                I = int(E1 / DELTAE) + 1
                I = min(I, INTEM) - 1
                TLIM = Object.MaxCollisionFreqNT[I]
                T = -1 * log(R1) / TLIM + TDASH
                Object.MeanCollisionTime = 0.9 * Object.MeanCollisionTime + 0.1 * T
                TDASH = T
                AP = DCZ1 * F2 * sqrt(E1)
                E = E1 + (AP + BP * T) * T
                IE = int(E / Object.ElectronEnergyStep)
                IE = min(IE, 3999)
                if TEMP[IE] > TLIM:
                    TDASH += log(R1) / TLIM
                    Object.MaxCollisionFreqNT[I] *= 1.05
                    continue

                # TEST FOR REAL OR NULL COLLISION
                R5 = random_uniform(RandomSeed)
                TEST1 = Object.TCFNT[IE] / TLIM

                if R5 > TEST1:
                    TEST2 = TEMP[IE] / TLIM
                    if R5 < TEST2:
                        # Test for null levels
                        if Object.NPLASTNT == 0:
                            continue
                        R2 = random_uniform(RandomSeed)
                        I = 0
                        while Object.CFNNT[IE][I] < R2:
                            I += 1
                        # Increment null scatter ssum
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
            #  CALCULATE POSITIONS AT INSTANT BEFORE COLLISION
            #    ALSO UPDATE DIFFUSION  AND ENERGY CALCULATIONS.
            T2 = T ** 2
            TDASH = 0.0
            CONST6 = sqrt(E1 / E)
            DCX2 = DCX1 * CONST6
            DCY2 = DCY1 * CONST6
            DCZ2 = DCZ1 * CONST6 + Object.EField * T * Object.CONST5 / sqrt(E)
            NCOL += 1
            A = AP * T
            B = BP * T2
            SUME2 = SUME2 + T * (E1 + A / 2.0 + B / 3.0)
            CONST7 = CONST9 * sqrt(E1)
            A = T * CONST7
            CX1 = DCX1 * CONST7
            CY1 = DCY1 * CONST7
            CZ1 = DCZ1 * CONST7
            Object.X = Object.X + DCX1 * A
            Object.Y = Object.Y + DCY1 * A
            Object.Z = Object.Z + DCZ1 * A + T2 * F1
            Object.TimeSum = Object.TimeSum + T
            IT = int(T)
            IT = min(IT, 299)
            Object.TIME[IT] += 1
            Object.SPEC[IE] += 1
            Object.VelocityZ = Object.Z / Object.TimeSum
            SUMVX = SUMVX + CX1 * CX1 * T2
            SUMVY = SUMVY + CY1 * CY1 * T2
            if ID != 0:
                KDUM = 0
                for JDUM in range(int(Object.Decor_NCORST)):
                    ST2 += T
                    NCOLDM = NCOL + KDUM
                    if NCOLDM > Object.Decor_NCOLM:
                        NCOLDM = NCOLDM - Object.Decor_NCOLM
                    SDIF = Object.TimeSum - STO[NCOLDM - 1]
                    SUMXX =SUMXX+ pow((Object.X - XST[NCOLDM - 1]) , 2) * T / SDIF
                    SUMYY = SUMYY+pow((Object.Y - YST[NCOLDM - 1]) , 2) * T / SDIF
                    KDUM += Object.Decor_NCORLN
                    if J1 >= 2:
                        ST1 += T
                        SUMZZ += pow((Object.Z - ZST[NCOLDM - 1] - Object.VelocityZ * SDIF) , 2) * T / SDIF
            XST[NCOL - 1] = Object.X
            YST[NCOL - 1] = Object.Y
            ZST[NCOL - 1] = Object.Z
            STO[NCOL - 1] = Object.TimeSum
            if NCOL >= Object.Decor_NCOLM:
                ID += 1
                NCOL = 0

            # Determination of real collision type
            R2 = random_uniform(RandomSeed)
            # Find location within 4 units in collision array
            I = MBSort(I, R2, IE, Object)
            while Object.CFNT[IE][I] < R2:
                I = I + 1

            S1 = Object.RGASNT[I]
            EI = Object.EINNT[I]
            if Object.IPNNT[I] > 0:
                # Use flat distributioon of electron energy between E-EION and 0.0 EV, same as in Boltzmann
                R9 = random_uniform(RandomSeed)
                EXTRA = R9 * (E - EI)
                EI = EXTRA + EI
                # Add extra ionisation collision
                IEXTRA += <long long>Object.NC0NT[I]

            # Generate scattering angles and update laboratory cosines after collision also update energy of electron
            IPT = <long long>Object.IARRYNT[I]
            Object.ICOLLNT[int(IPT) - 1] += 1
            Object.ICOLNNT[I] += 1
            if E < EI:
                EI = E - 0.0001

            # IF EXCITATION THEN ADD PROBABILITY ,PENFRA(1,I), OF TRANSFER TO
            # IONISATION OF THE OTHER GASES IN MIXTURE
            if Object.EnablePenning != 0:
                if Object.PENFRANT[0][I] != 0:
                    RAN = random_uniform(RandomSeed)
                    if RAN <= Object.PENFRANT[0][I]:
                        # add extra ionisation collision
                        IEXTRA += 1
            S2 = (S1 ** 2) / (S1 - 1.0)

            # Anisotropic scattering
            R3 = random_uniform(RandomSeed)
            if Object.INDEXNT[I] == 1:
                R31 = random_uniform(RandomSeed)
                F3 = 1.0 - R3 * Object.ANGCTNT[IE][I]
                if R31 > Object.PSCTNT[IE][I]:
                    F3 = -1 * F3
            elif Object.INDEXNT[I] == 2:
                EPSI = Object.PSCTNT[IE][I]
                F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
            else:
                # Isotropic scattering
                F3 = 1 - 2 * R3
            THETA0 = acos(F3)
            R4 = random_uniform(RandomSeed)
            PHI0 = F4 * R4
            F8 = sin(PHI0)
            F9 = cos(PHI0)
            ARG1 = 1 - S1 * EI / E
            ARG1 = max(ARG1, Object.SmallNumber)
            D = 1 - F3 * sqrt(ARG1)
            E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)
            E1 = max(E1, Object.SmallNumber)
            Q = sqrt((E / E1) * ARG1) / S1
            Q = min(Q, 1)
            Object.AngleFromZ = asin(Q * sin(THETA0))
            F6 = cos(Object.AngleFromZ)
            U = (S1 - 1) * (S1 - 1) / ARG1
            CSQD = F3 * F3
            if F3 < 0 and CSQD > U:
                F6 = -1 * F6
            F5 = sin(Object.AngleFromZ)
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
        Object.VelocityZ *= 1e9
        Object.MeanElectronEnergy = SUME2 / Object.TimeSum
        Object.LongitudinalDiffusion = 0.0
        if Object.AnisotropicDetected == 0:
            Object.DiffusionX = 5e15 * SUMVX / Object.TimeSum
            Object.DiffusionY = 5e15 * SUMVY / Object.TimeSum
            DFXXST[J1] = 5e15 * (SUMVX - SVXOLD) / (Object.TimeSum - STOLD)
            DFYYST[J1] = 5e15 * (SUMVY - SVYOLD) / (Object.TimeSum - STOLD)
        else:
            if ST2 != 0.0:
                Object.DiffusionY = 5e15 * SUMYY / ST2
                Object.DiffusionX = 5e15 * SUMXX / ST2
                DFXXST[J1] = 5e15 * (SUMXX - SXXOLD) / (ST2 - ST2OLD)
                DFYYST[J1] = 5e15 * (SUMYY - SYYOLD) / (ST2 - ST2OLD)
            else:
                DFXXST[J1] = 0.0
                DFYYST[J1] = 0.0

        if ST1 != 0.0:
            Object.DiffusionZ = 5e15 * SUMZZ / ST1
            DFZZST[J1] = 5e15 * (SUMZZ - SZZOLD) / (ST1 - ST1OLD)
        else:
            DFZZST[J1] = 0.0
        WZST[J1] = (Object.Z - ZOLD) / (Object.TimeSum - STOLD) * 1e9
        AVEST[J1] = (SUME2 - SME2OLD) / (Object.TimeSum - STOLD)
        ZOLD = Object.Z
        STOLD = Object.TimeSum
        ST1OLD = ST1
        ST2OLD = ST2
        SVXOLD = SUMVX
        SVYOLD = SUMVY
        SZZOLD = SUMZZ
        SYYOLD = SUMYY
        SXXOLD = SUMXX
        SME2OLD = SUME2
        if Object.ConsoleOutputFlag:
            print('{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}{:^10.1f}'.format(Object.VelocityZ, Object.Z, Object.TimeSum,
                                                                                    Object.MeanElectronEnergy, Object.DiffusionX, Object.DiffusionY,
                                                                                    Object.DiffusionZ))
        if Object.SPEC[3999] > (1000 * float(J1+1)):
            raise ValueError("WARNING ENERGY OUT OF RANGE, INCREASE ELECTRON ENERGY INTEGRATION RANGE")
    # Calculate errors and check averages
    TWZST = 0.0
    TAVE = 0.0
    T2WZST = 0.0
    T2AVE = 0.0
    TZZST = 0.0
    TYYST = 0.0
    TXXST = 0.0
    T2ZZST = 0.0
    T2YYST = 0.0
    T2XXST = 0.0
    for K in range(10):
        TWZST = TWZST + WZST[K]
        TAVE = TAVE + AVEST[K]
        T2WZST = T2WZST + WZST[K] * WZST[K]
        T2AVE = T2AVE + AVEST[K] * AVEST[K]
        TXXST = TXXST + DFXXST[K]
        TYYST = TYYST + DFYYST[K]
        T2YYST = T2YYST + DFYYST[K] ** 2
        T2XXST = T2XXST + DFXXST[K] ** 2
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

    # Calculate Townsend coeficients and errors
    ANCATT = 0.0
    ANCION = 0.0
    for I in range(Object.NumberOfGases):
        ANCATT += Object.ICOLLNT[5 * (I + 1) - 3]
        ANCION += Object.ICOLLNT[5 * (I + 1) - 4]
    ANCION += IEXTRA
    Object.AttachmentRateError = 0.0
    if ANCATT != 0:
        Object.AttachmentRateError = 100 * sqrt(ANCATT) / ANCATT
    Object.AttachmentRate = ANCATT / (Object.TimeSum * Object.VelocityZ) * 1e12
    Object.IonisationRateError = 0.0
    if ANCION != 0:
        Object.IonisationRateError = 100 * sqrt(ANCION) / ANCION
    Object.IonisationRate = ANCION / (Object.TimeSum * Object.VelocityZ) * 1e12

    return
    
