from PyBoltz cimport PyBoltz
import cython
from PyBoltz cimport drand48
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, log10
from libc.stdlib cimport malloc, free
from libc.string cimport memset
import numpy as np
cimport numpy as np

from MBSorts cimport MBSort, MBSortT

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int isNaN(double num):
    return num != num

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double dummy):
    cdef double r = drand48(dummy)
    return r

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void GenerateMaxBoltz(double RandomSeed, double *RandomMaxBoltzArray):
    cdef double RAN1, RAN2, TWOPI
    cdef int J
    for J in range(0, 5, 2):
        RAN1 = random_uniform(RandomSeed)
        RAN2 = random_uniform(RandomSeed)
        TWOPI = 2.0 * np.pi
        RandomMaxBoltzArray[J] = sqrt(-1 * log(RAN1)) * cos(RAN2 * TWOPI)
        RandomMaxBoltzArray[J + 1] = sqrt(-1 * log(RAN1)) * sin(RAN2 * TWOPI)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimit(PyBoltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the PyBoltz object to try a higher FinalEnergy value.
    This function is used when there is no magnetic field. 

    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    
    Electric field in z direction.

    The object parameter is the PyBoltz object to be setup and used in the simulation.
    """

    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE, INTEM
    cdef double SMALL, RandomSeed, E1, TDASH, CONST5, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, CONST11, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, F3, RAN, EPSI, R4, PHI0, F8, F9, ARG1
    cdef double D, Q, U, CSQD, F6, F5, ARGZ, CONST12, VXLAB, VYLAB, VZLAB, TEMP[4000], DELTAE
    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))

    CONST5 = Object.CONST3 / 2.0
    ISAMP = 10
    SMALL = 1.0e-20
    I = 0
    RandomSeed = Object.RandomSeed
    E1 = Object.InitialElectronEnergy
    N4000 = 4000
    TDASH = 0.0
    INTEM = 8
    for J in range(N4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]

    # INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    BP = (Object.EField ** 2) * Object.CONST1
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3
    F4 = 2 * acos(-1)
    DELTAE = Object.FinalElectronEnergy / float(INTEM)
    E1 = Object.InitialElectronEnergy
    J2M = Object.MaxNumberOfCollisions / ISAMP
    for J1 in range(int(J2M)):
        if J1 != 0 and not int(str(J1)[-int(log10(J1)):]) and Object.ConsoleOutputFlag:
            print('* Num analyzed collisions: {}'.format(J1))
        while True:
            R1 = random_uniform(RandomSeed)
            I = int(E1 / DELTAE) + 1
            I = min(I, INTEM) - 1
            TLIM = Object.MaxCollisionFreqNT[I]
            T = -1 * log(R1) / TLIM + TDASH
            TDASH = T
            AP = DirCosineZ1 * F2 * sqrt(E1)
            E = E1 + (AP + BP * T) * T
            IE = int(E / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += log(R1) / TLIM
                Object.MaxCollisionFreqNT[I] *= 1.05
                continue

            # Test FOR NULL COLLISIONS
            R5 = random_uniform(RandomSeed)
            Test1 = Object.TotalCollisionFrequencyNT[IE] / TLIM
            if R5 <= Test1:
                break

        if IE == 3999:
            # Electron energy is out of limits
            return 1

        # CALCULATE DIRECTION COSINES AT INSTANT BEFORE COLLISION
        TDASH = 0.0
        CONST6 = sqrt(E1 / E)
        DirCosineX2 = DirCosineX1 * CONST6
        DirCosineY2 = DirCosineY1 * CONST6
        DirCosineZ2 = DirCosineZ1 * CONST6 + Object.EField * T * CONST5 / sqrt(E)
        R2 = random_uniform(RandomSeed)

        # Determination of real collision type
        I = MBSort(I, R2, IE, Object)
        # Find the location within 4 units in collision array
        while Object.CollisionFrequencyNT[IE][I] < R2:
            I = I + 1

        S1 = Object.RGASNT[I]
        EI = Object.EnergyLevelsNT[I]
        if Object.IPNNT[I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI

        # Generate scattering angles and update laboratory cosines after collision also update energy of electron
        IPT = Object.IARRYNT[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random_uniform(RandomSeed)

        if Object.INDEXNT[I] == 1:
            R31 = random_uniform(RandomSeed)
            F3 = 1.0 - R3 * Object.AngleCutNT[IE][I]
            if R31 > Object.ScatteringParameterNT[IE][I]:
                F3 = -1 * F3
        elif Object.INDEXNT[I] == 2:
            EPSI = Object.ScatteringParameterNT[IE][I]
            F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
        else:
            F3 = 1 - 2 * R3
        THETA0 = acos(F3)
        R4 = random_uniform(RandomSeed)
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
        Object.AngleFromZ = asin(Q * sin(THETA0))

        F6 = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = F3 ** 2

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6

        F5 = sin(Object.AngleFromZ)
        DirCosineZ2 = min(DirCosineZ2, 1)
        ARGZ = sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
        if ARGZ == 0:
            DirCosineZ1 = F6
            DirCosineX1 = F9 * F5
            DirCosineY1 = F8 * F5
        else:
            DirCosineZ1 = DirCosineZ2 * F6 + ARGZ * F5 * F8
            DirCosineY1 = DirCosineY2 * F6 + (F5 / ARGZ) * (DirCosineX2 * F9 - DirCosineY2 * DirCosineZ2 * F8)
            DirCosineX1 = DirCosineX2 * F6 - (F5 / ARGZ) * (DirCosineY2 * F9 + DirCosineX2 * DirCosineZ2 * F8)

    return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitB(PyBoltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the PyBoltz object to try a higher FinalEnergy value.
    This function is used when the magnetic field angle is 90 degrees to the electric field. 

    The object parameter is the PyBoltz object to be setup and used in the simulation.
    
    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE, INTEM
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, CONST11, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, F3, RAN, EPSI, R4, PHI0, F8, F9, ARG1
    cdef double D, Q, U, CSQD, F6, F5, ARGZ, CONST12, VXLAB, VYLAB, VZLAB, TEMP[4000], DELTAE, EF100
    TEMP = <double *> malloc(4000 * sizeof(double))

    memset(TEMP, 0, 4000 * sizeof(double))

    Object.SmallNumber = 1.0e-20
    ISAMP = 20
    EF100 = Object.EField * 100
    RandomSeed = Object.RandomSeed
    E1 = Object.InitialElectronEnergy

    INTEM = 8
    TDASH = 0.0
    CONST9 = Object.CONST3 * 0.01

    # INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    for J in range(4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]

    VTOT = CONST9 * sqrt(E1)
    CX1 = DirCosineX1 * VTOT
    CY1 = DirCosineY1 * VTOT
    CZ1 = DirCosineZ1 * VTOT

    F4 = 2 * acos(-1)

    DELTAE = Object.FinalElectronEnergy / float(INTEM)

    J2M = Object.MaxNumberOfCollisions / ISAMP

    for J1 in range(int(J2M)):
        if J1 != 0 and not int(str(J1)[-int(log10(J1)):]) and Object.ConsoleOutputFlag:
            print('* Num analyzed collisions: {}'.format(J1))
        while True:
            R1 = random_uniform(RandomSeed)
            I = int(E1 / DELTAE) + 1
            I = min(I, INTEM) - 1
            TLIM = Object.MaxCollisionFreqNT[I]
            T = -1 * log(R1) / TLIM + TDASH
            TDASH = T
            WBT = Object.AngularSpeedOfRotation * T
            COSWT = cos(WBT)
            SINWT = sin(WBT)
            DZ = (CZ1 * SINWT + (Object.EFieldOverBField - CY1) * (1 - COSWT)) / Object.AngularSpeedOfRotation
            E = E1 + DZ * EF100
            IE = int(E / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += log(R1) / TLIM
                Object.MaxCollisionFreqNT[I] *= 1.05
                continue
            R5 = random_uniform(RandomSeed)
            Test1 = Object.TotalCollisionFrequencyNT[IE] / TLIM

            # Test FOR REAL OR NULL COLLISION
            if R5 <= Test1:
                break

        if IE == 3999:
            # Electron energy out of range
            return 1

        TDASH = 0.0
        CX2 = CX1
        CY2 = (CY1 - Object.EFieldOverBField) * COSWT + CZ1 * SINWT + Object.EFieldOverBField
        CZ2 = CZ1 * COSWT - (CY1 - Object.EFieldOverBField) * SINWT
        VTOT = sqrt(CX2 ** 2 + CY2 ** 2 + CZ2 ** 2)
        DirCosineX2 = CX2 / VTOT
        DirCosineY2 = CY2 / VTOT
        DirCosineZ2 = CZ2 / VTOT

        # DETERMINATION OF REAL COLLISION TYPE
        R2 = random_uniform(RandomSeed)

        # FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
        I = MBSort(I, R2, IE, Object)
        while Object.CollisionFrequencyNT[IE][I] < R2:
            I = I + 1

        S1 = Object.RGASNT[I]
        EI = Object.EnergyLevelsNT[I]
        if Object.IPNNT[I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI

        # GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
        # COLLISION ALSO UPDATE ENERGY OF ELECTRON.
        IPT = Object.IARRYNT[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random_uniform(RandomSeed)

        if Object.INDEXNT[I] == 1:
            R31 = random_uniform(RandomSeed)
            F3 = 1.0 - R3 * Object.AngleCutNT[IE][I]
            if R31 > Object.ScatteringParameterNT[IE][I]:
                F3 = -1 * F3
        elif Object.INDEXNT[I] == 2:
            EPSI = Object.ScatteringParameterNT[IE][I]
            F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
        else:
            F3 = 1 - 2 * R3
        THETA0 = acos(F3)
        R4 = random_uniform(RandomSeed)
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
        Object.AngleFromZ = asin(Q * sin(THETA0))

        F6 = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = F3 ** 2

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6
        F5 = sin(Object.AngleFromZ)
        DirCosineZ2 = min(DirCosineZ2, 1)
        VTOT = CONST9 * sqrt(E1)
        ARGZ = sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
        if ARGZ == 0:
            DirCosineZ1 = F6
            DirCosineX1 = F9 * F5
            DirCosineY1 = F8 * F5
        else:
            DirCosineZ1 = DirCosineZ2 * F6 + ARGZ * F5 * F8
            DirCosineY1 = DirCosineY2 * F6 + (F5 / ARGZ) * (DirCosineX2 * F9 - DirCosineY2 * DirCosineZ2 * F8)
            DirCosineX1 = DirCosineX2 * F6 - (F5 / ARGZ) * (DirCosineY2 * F9 + DirCosineX2 * DirCosineZ2 * F8)
        CX1 = DirCosineX1 * VTOT
        CY1 = DirCosineY1 * VTOT
        CZ1 = DirCosineZ1 * VTOT

    return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitBT(PyBoltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the PyBoltz object to try a higher FinalEnergy value.
    This function is used when the magnetic field angle is 90 degrees to the electric field. 

    The object parameter is the PyBoltz object to be setup and used in the simulation.
    
    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, CONST11, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, F3, RAN, EPSI, R4, PHI0, F8, F9, ARG1
    cdef double D, Q, U, CSQD, F6, F5, ARGZ, CONST12, VXLAB, VYLAB, EF100, TLIM, CX2, CY2, CZ2

    ISAMP = 20
    SMALL = 1.0e-20
    EF100 = Object.EField * 100
    RandomSeed = Object.RandomSeed
    E1 = Object.InitialElectronEnergy
    N4000 = 4000
    TDASH = 0.0

    # GENRATE RANDOM NUMBER FOR MAXWELL BOLTZMAN
    GenerateMaxBoltz(Object.RandomSeed, Object.RandomMaxBoltzArray)
    IMBPT = 0

    CONST9 = Object.CONST3 * 0.01
    CONST10 = CONST9 * CONST9

    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    VTOT = CONST9 * sqrt(E1)
    CX1 = DirCosineX1 * VTOT
    CY1 = DirCosineY1 * VTOT
    CZ1 = DirCosineZ1 * VTOT

    F4 = 2 * acos(-1)
    J2M = Object.MaxNumberOfCollisions / ISAMP

    for J1 in range(int(J2M)):
        if J1 != 0 and not int(str(J1)[-int(log10(J1)):]) and Object.ConsoleOutputFlag:
            print('* Num analyzed collisions: {}'.format(J1))
        while True:
            R1 = random_uniform(RandomSeed)
            T = -1 * log(R1) / Object.MaxCollisionFreqTotal + TDASH
            TDASH = T
            WBT = Object.AngularSpeedOfRotation * T
            COSWT = cos(WBT)
            SINWT = sin(WBT)

            DZ = (CZ1 * SINWT + (Object.EFieldOverBField - CY1) * (1 - COSWT)) / Object.AngularSpeedOfRotation
            E = E1 + DZ * EF100
            #CALC ELECTRON VELOCITY IN LAB FRAME
            CX2 = CX1
            CY2 = (CY1 - Object.EFieldOverBField) * COSWT + CZ1 * SINWT + Object.EFieldOverBField
            CZ2 = CZ1 * COSWT - (CY1 - Object.EFieldOverBField) * SINWT
            #FIND IDENTITY OF GAS FOR COLLISION
            GasIndex = 0
            R2 = random_uniform(RandomSeed)
            while Object.MaxCollisionFreqTotalG[GasIndex] < R2:
                GasIndex += 1
            #CALCULATE GAS VELOCITY VECTORS VGX,VGY,VGZ
            IMBPT += 1
            if IMBPT > 6:
                GenerateMaxBoltz(Object.RandomSeed, Object.RandomMaxBoltzArray)
                IMBPT = 1
            VGX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            VGY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            VGZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]

            #CALCULATE ENERGY WITH STATIONARY GAS TARGET , EOK
            EOK = (pow((CX2 - VGX), 2) + pow((CY2 - VGY), 2) + pow((CZ2 - VGZ), 2)) / CONST10
            IE = int(EOK / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            #Test FOR REAL OR NULL COLLISION
            R5 = random_uniform(RandomSeed)
            TLIM = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
            if R5 <= TLIM:
                break
        if IE == 3999:
            #ELECTRON ENERGY OUT OF RANGE
            return 1

        TDASH = 0.0
        #CALCULATE DIRECTION COSINES OF ELECTRON IN 0 KELVIN FRAME
        CONST11 = 1.0 / (CONST9 * sqrt(EOK))
        DXCOM = (CX2 - VGX) * CONST11
        DYCOM = (CY2 - VGY) * CONST11
        DZCOM = (CZ2 - VGZ) * CONST11

        #FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
        R2 = random_uniform(RandomSeed)
        I = MBSortT(GasIndex, I, R2, IE, Object)
        while Object.CollisionFrequency[GasIndex][IE][I] < R2:
            I = I + 1

        S1 = Object.RGAS[GasIndex][I]
        EI = Object.EnergyLevels[GasIndex][I]
        if Object.IPN[GasIndex][I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (EOK - EI)
            EI = EXTRA + EI

        # GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
        # COLLISION ALSO UPDATE ENERGY OF ELECTRON.
        IPT = Object.IARRY[GasIndex][I]
        if EOK < EI:
            EI = EOK - 0.0001
        S2 = (S1 * S1) / (S1 - 1)
        R3 = random_uniform(RandomSeed)

        if Object.INDEX[GasIndex][I] == 1:
            R31 = random_uniform(RandomSeed)
            F3 = 1.0 - R3 * Object.AngleCut[GasIndex][IE][I]
            if R31 > Object.ScatteringParameter[GasIndex][IE][I]:
                F3 = -1 * F3
        elif Object.INDEX[GasIndex][I] == 2:
            EPSI = Object.ScatteringParameter[GasIndex][IE][I]
            F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
        else:
            F3 = 1 - 2 * R3

        THETA0 = acos(F3)
        R4 = random_uniform(RandomSeed)
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
        Object.AngleFromZ = asin(Q * sin(THETA0))

        F6 = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = F3 ** 2

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6
        F5 = sin(Object.AngleFromZ)
        DirCosineZ2 = min(DZCOM, 1)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
        if ARGZ == 0:
            DirCosineZ1 = F6
            DirCosineX1 = F9 * F5
            DirCosineY1 = F8 * F5
        else:
            DirCosineZ1 = DZCOM * F6 + ARGZ * F5 * F8
            DirCosineY1 = DYCOM * F6 + (F5 / ARGZ) * (DXCOM * F9 - DYCOM * DZCOM * F8)
            DirCosineX1 = DXCOM * F6 - (F5 / ARGZ) * (DYCOM * F9 + DXCOM * DZCOM * F8)
        # TRANSFORM VELOCITY VECTORS TO LAB FRAME
        VTOT = CONST9 * sqrt(E1)
        CX1 = DirCosineX1 * VTOT + VGX
        CY1 = DirCosineY1 * VTOT + VGY
        CZ1 = DirCosineZ1 * VTOT + VGZ
        #  CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
        E1 = (CX1 * CX1 + CY1 * CY1 + CZ1 * CZ1) / CONST10
        CONST11 = 1.0 / (CONST9 * sqrt(E1))
        DirCosineX1 = CX1 * CONST11
        DirCosineY1 = CY1 * CONST11
        DirCosineZ1 = CZ1 * CONST11

    return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitC(PyBoltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the PyBoltz object to try a higher FinalEnergy value.
    
    The object parameter is the PyBoltz object to be setup and used in the simulation.
    
    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE, INTEM
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, CONST11, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, F3, RAN, EPSI, R4, PHI0, F8, F9, ARG1
    cdef double D, Q, U, CSQD, F6, F5, ARGZ, CONST12, VXLAB, VYLAB, VZLAB, TEMP[4000], DELTAE, EFX100, EFZ100, RTHETA,
    TEMP = <double *> malloc(4000 * sizeof(double))
    memset(TEMP, 0, 4000 * sizeof(double))
    ISAMP = 20
    SMALL = 1.0e-20
    RTHETA = Object.BFieldAngle * acos(-1) / 180
    EFZ100 = Object.EField * 100 * sin(RTHETA)
    EFX100 = Object.EField * 100 * cos(RTHETA)
    F1 = Object.EField * Object.CONST2 * cos(RTHETA)
    EOVBR = Object.EFieldOverBField * sin(RTHETA)
    E1 = Object.InitialElectronEnergy
    INTEM = 8
    TDASH = 0.0
    CONST9 = Object.CONST3 * 0.01
    for J in range(4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]

    # INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)
    VTOT = CONST9 * sqrt(E1)
    CX1 = DirCosineX1 * VTOT
    CY1 = DirCosineY1 * VTOT
    CZ1 = DirCosineZ1 * VTOT

    F4 = 2 * acos(-1)
    DELTAE = Object.FinalElectronEnergy / float(INTEM)
    J2M = Object.MaxNumberOfCollisions / ISAMP
    RandomSeed = Object.RandomSeed

    for J1 in range(int(J2M)):
        if J1 != 0 and not int(str(J1)[-int(log10(J1)):]) and Object.ConsoleOutputFlag:
            print('* Num analyzed collisions: {}'.format(J1))
        while True:
            R1 = random_uniform(RandomSeed)
            I = int(E1 / DELTAE) + 1
            I = min(I, INTEM) - 1
            TLIM = Object.MaxCollisionFreqNT[I]
            T = -1 * log(R1) / TLIM + TDASH
            TDASH = T
            WBT = Object.AngularSpeedOfRotation * T
            COSWT = cos(WBT)
            SINWT = sin(WBT)
            DZ = (CZ1 * SINWT + (EOVBR - CY1) * (1 - COSWT)) / Object.AngularSpeedOfRotation
            DX = CX1 * T + F1 * T * T
            E = E1 + DZ * EFZ100 + DX * EFX100
            IE = int(E / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += log(R1) / TLIM
                Object.MaxCollisionFreqNT[I] *= 1.05
                continue
            R5 = random_uniform(RandomSeed)
            Test1 = Object.TotalCollisionFrequencyNT[IE] / TLIM

            # Test FOR REAL OR NULL COLLISION
            if R5 <= Test1:
                break

        if IE == 3999:
            # Electron energy out of range
            return 1

        TDASH = 0.0
        CX2 = CX1 + 2 * F1 * T
        CY2 = (CY1 - EOVBR) * COSWT + CZ1 * SINWT + EOVBR
        CZ2 = CZ1 * COSWT - (CY1 - EOVBR) * SINWT
        VTOT = sqrt(CX2 ** 2 + CY2 ** 2 + CZ2 ** 2)
        DirCosineX2 = CX2 / VTOT
        DirCosineY2 = CY2 / VTOT
        DirCosineZ2 = CZ2 / VTOT
        R2 = random_uniform(RandomSeed)
        # DETERMINATION OF REAL COLLISION TYPE

        # FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
        I = MBSort(I, R2, IE, Object)
        while Object.CollisionFrequencyNT[IE][I] < R2:
            I = I + 1

        S1 = Object.RGASNT[I]
        EI = Object.EnergyLevelsNT[I]
        if Object.IPNNT[I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI

        # GENERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
        # COLLISION ALSO UPDATE ENERGY OF ELECTRON.
        IPT = Object.IARRYNT[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random_uniform(RandomSeed)

        if Object.INDEXNT[I] == 1:
            R31 = random_uniform(RandomSeed)
            F3 = 1.0 - R3 * Object.AngleCutNT[IE][I]
            if R31 > Object.ScatteringParameterNT[IE][I]:
                F3 = -1 * F3
        elif Object.INDEXNT[I] == 2:
            EPSI = Object.ScatteringParameterNT[IE][I]
            F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
        else:
            F3 = 1 - 2 * R3
        THETA0 = acos(F3)
        R4 = random_uniform(RandomSeed)
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
        Object.AngleFromZ = asin(Q * sin(THETA0))

        F6 = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = F3 ** 2

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6
        F5 = sin(Object.AngleFromZ)
        DirCosineZ2 = min(DirCosineZ2, 1)
        VTOT = CONST9 * sqrt(E1)
        ARGZ = sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
        if ARGZ == 0:
            DirCosineZ1 = F6
            DirCosineX1 = F9 * F5
            DirCosineY1 = F8 * F5
        else:
            DirCosineZ1 = DirCosineZ2 * F6 + ARGZ * F5 * F8
            DirCosineY1 = DirCosineY2 * F6 + (F5 / ARGZ) * (DirCosineX2 * F9 - DirCosineY2 * DirCosineZ2 * F8)
            DirCosineX1 = DirCosineX2 * F6 - (F5 / ARGZ) * (DirCosineY2 * F9 + DirCosineX2 * DirCosineZ2 * F8)
        CX1 = DirCosineX1 * VTOT
        CY1 = DirCosineY1 * VTOT
        CZ1 = DirCosineZ1 * VTOT

    return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitCT(PyBoltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the PyBoltz object to try a higher FinalEnergy value. 
    
    The object parameter is the PyBoltz object to be setup and used in the simulation.
    
    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, BP, F1, F2, F4, J2M, R5, Test1, R1, T, AP, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, CONST11, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, F3, RAN, EPSI, R4, PHI0, F8, F9, ARG1,
    cdef double D, Q, U, CSQD, F6, F5, ARGZ, RTHETA, CONST12, VXLAB, VYLAB, EFX100, EFZ100, TLIM, CX2, CY2, CZ2, EOVBR

    ISAMP = 20
    SMALL = 1.0e-20

    I = 0

    RTHETA = Object.BFieldAngle * np.pi / 180.0
    EFZ100 = Object.EField * 100 * sin(RTHETA)
    EFX100 = Object.EField * 100 * cos(RTHETA)
    F1 = Object.EField * Object.CONST2 * cos(RTHETA)
    F4 = 2 * np.pi
    EOVBR = Object.EFieldOverBField * sin(RTHETA)
    RandomSeed = Object.RandomSeed
    E1 = Object.InitialElectronEnergy
    TDASH = 0.0

    CONST9 = Object.CONST3 * 0.01
    CONST10 = CONST9 ** 2

    #GENERATE RANDOM NUMBER FOR MAXWELL BOLTZMAN
    GenerateMaxBoltz(Object.RandomSeed, Object.RandomMaxBoltzArray)
    IMBPT = 0

    #INITIAL DIRECTION COSINES
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    VTOT = CONST9 * sqrt(E1)
    CX1 = DirCosineX1 * VTOT
    CY1 = DirCosineY1 * VTOT
    CZ1 = DirCosineZ1 * VTOT

    J2M = Object.MaxNumberOfCollisions / ISAMP

    for J1 in range(int(J2M)):
        if J1 != 0 and not int(str(J1)[-int(log10(J1)):]) and Object.ConsoleOutputFlag:
            print('* Num analyzed collisions: {}'.format(J1))
        while True:
            R1 = random_uniform(RandomSeed)
            T = -1 * log(R1) / Object.MaxCollisionFreqTotal + TDASH
            TDASH = T
            WBT = Object.AngularSpeedOfRotation * T
            COSWT = cos(WBT)
            SINWT = sin(WBT)
            DZ = (CZ1 * SINWT + (Object.EFieldOverBField - CY1) * (1 - COSWT)) / Object.AngularSpeedOfRotation
            DX = CX1 * T + F1 * T * T

            E = E1 + DZ * EFZ100 + DX * EFX100

            CX2 = CX1 + 2 * F1 * T
            CY2 = (CY1 - EOVBR) * COSWT + CZ1 * SINWT + EOVBR
            CZ2 = CZ1 * COSWT - (CY1 - EOVBR) * SINWT

            #FIND IDENTITY OF GAS FOR COLLISION
            GasIndex = 0
            R2 = random_uniform(RandomSeed)
            while (Object.MaxCollisionFreqTotalG[GasIndex] < R2):
                GasIndex += 1
            #CALCULATE GAS VELOCITY VECTORS VGX,VGY,VGZ
            IMBPT += 1
            if IMBPT > 6:
                GenerateMaxBoltz(Object.RandomSeed, Object.RandomMaxBoltzArray)
                IMBPT = 1
            VGX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            VGY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]
            IMBPT = IMBPT + 1
            VGZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1) % 6]

            EOK = ((CX2 - VGX) ** 2 + (CY2 - VGY) ** 2 + (CZ2 - VGZ) ** 2) / CONST10
            IE = int(EOK / Object.ElectronEnergyStep)
            IE = min(IE, 3999)

            # Test FOR REAL OR NULL COLLISION
            R5 = random_uniform(RandomSeed)
            TLIM = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
            if R5 <= TLIM:
                break
        if IE == 3999:
            return 1

        #CALCULATE DIRECTION COSINES OF ELECTRON IN 0 KELVIN FRAME
        TDASH = 0.0
        CONST11 = 1.0 / (CONST9 * sqrt(EOK))
        DXCOM = (CX2 - VGX) * CONST11
        DYCOM = (CY2 - VGY) * CONST11
        DZCOM = (CZ2 - VGZ) * CONST11

        # ---------------------------------------------------------------------
        #     DETERMINATION OF REAL COLLISION TYPE
        # ---------------------------------------------------------------------

        R2 = random_uniform(RandomSeed)

        #FIND LOCATION WITHIN 4 UNITS IN COLLISION ARRAY
        I = MBSortT(GasIndex, I, R2, IE, Object)
        while Object.CollisionFrequency[GasIndex][IE][I] < R2:
            I = I + 1

        S1 = Object.RGAS[GasIndex][I]
        EI = Object.EnergyLevels[GasIndex][I]
        if Object.IPN[GasIndex][I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (EOK - EI)
            EI = EXTRA + EI

        #  GENEERATE SCATTERING ANGLES AND UPDATE  LABORATORY COSINES AFTER
        #   COLLISION ALSO UPDATE ENERGY OF ELECTRON.
        IPT = Object.IARRY[GasIndex][I]
        if EOK < EI:
            EI = EOK - 0.0001
        S2 = (S1 * S1) / (S1 - 1)
        R3 = random_uniform(RandomSeed)

        if Object.INDEX[GasIndex][I] == 1:
            R31 = random_uniform(RandomSeed)
            F3 = 1 - R3 * Object.AngleCut[GasIndex][IE][I]
            if R31 > Object.ScatteringParameter[GasIndex][IE][I]:
                F3 = -1 * F3
            elif Object.INDEX[GasIndex][I] == 2:
                EPSI = Object.ScatteringParameter[GasIndex][IE][I]
                F3 = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
            else:
                F3 = 1 - 2 * R3
        THETA0 = acos(F3)
        R4 = random_uniform(RandomSeed)
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
        Object.AngleFromZ = asin(Q * sin(THETA0))

        F6 = cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CSQD = F3 ** 2

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6
        F5 = sin(Object.AngleFromZ)
        DZCOM = min(DZCOM, 1)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
        if ARGZ == 0:
            DirCosineZ1 = F6
            DirCosineX1 = F9 * F5
            DirCosineY1 = F8 * F5
        else:
            DirCosineZ1 = DZCOM * F6 + ARGZ * F5 * F8
            DirCosineY1 = DYCOM * F6 + (F5 / ARGZ) * (DXCOM * F9 - DYCOM * DZCOM * F8)
            DirCosineX1 = DXCOM * F6 - (F5 / ARGZ) * (DYCOM * F9 + DXCOM * DZCOM * F8)
        # TRANSFORM VELOCITY VECTORS TO LAB FRAME
        VTOT = CONST9 * sqrt(E1)
        CX1 = DirCosineX1 * VTOT + VGX
        CY1 = DirCosineY1 * VTOT + VGY
        CZ1 = DirCosineZ1 * VTOT + VGZ
        #  CALCULATE ENERGY AND DIRECTION COSINES IN LAB FRAME
        E1 = (CX1 * CX1 + CY1 * CY1 + CZ1 * CZ1) / CONST10
        CONST11 = 1.0 / (CONST9 * sqrt(E1))
        DirCosineX1 = CX1 * CONST11
        DirCosineY1 = CY1 * CONST11
        DirCosineZ1 = CZ1 * CONST11

    return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef EnergyLimitT(PyBoltz Object):
    """
    This function is used to calculate the upper electron energy limit by simulating the collisions. If it crosses the 
    Object.FinalEnergy value it will set self.IELOW to 1 which would get the PyBoltz object to try a higher FinalEnergy value.
    This function is used when there is no magnetic field. 

    The test is carried out for a sample of collisions that are smaller than the full sample by a factor of 1/isamp
    
    Electric field in z direction.

    The object parameter is the PyBoltz object to be setup and used in the simulation.
    """
    cdef long long I, ISAMP, N4000, IMBPT, J1, GasIndex, IE
    cdef double SMALL, RandomSeed, E1, TDASH, CONST9, CONST10, DirCosineZ1, DirCosineX1, DirCosineY1, AP, BP, F1, F2, F4, J2M, R5, Test1, R1, T, E, CONST6, DirCosineX2, DirCosineY2, DirCosineZ2, R2,
    cdef double VGX, VGY, VGZ, VEX, VEY, VEZ, EOK, CONST11, DXCOM, DYCOM, DZCOM, S1, EI, R9, EXTRA, IPT, S2, R3, R31, F3, RAN, EPSI, R4, PHI0, F8, F9, ARG1
    cdef double D, Q, U, CSQD, F6, F5, ARGZ, CONST12, VXLAB, VYLAB, VZLAB,A,B

    ISAMP = 10
    SMALL = 1.0e-20
    RandomSeed = Object.RandomSeed
    E1 = Object.InitialElectronEnergy
    N4000 = 4000
    TDASH = 0.0
    CONST5 = Object.CONST3 / 2.0
    CONST9 = Object.CONST3 * 0.01
    CONST10 = CONST9 * CONST9
    GenerateMaxBoltz(Object.RandomSeed, Object.RandomMaxBoltzArray)
    IMBPT = 0
    DirCosineZ1 = cos(Object.AngleFromZ)
    DirCosineX1 = sin(Object.AngleFromZ) * cos(Object.AngleFromX)
    DirCosineY1 = sin(Object.AngleFromZ) * sin(Object.AngleFromX)

    BP = pow(Object.EField, 2) * Object.CONST1
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3
    F4 = 2 * acos(-1)
    J2M = Object.MaxNumberOfCollisions / ISAMP
    for J1 in range(int(J2M)):
        if J1 != 0 and not int(str(J1)[-int(log10(J1)):]) and Object.ConsoleOutputFlag:
            print('* Num analyzed collisions: {}'.format(J1))
        while True:
            R1 = random_uniform(RandomSeed)
            T = -1 * log(R1) / Object.MaxCollisionFreqTotal + TDASH
            TDASH = T
            AP = DirCosineZ1 * F2 * sqrt(E1)
            E = E1 + (AP + BP * T) * T
            CONST6 = sqrt(E1 / E)
            DirCosineX2 = DirCosineX1 * CONST6
            DirCosineY2 = DirCosineY1 * CONST6
            DirCosineZ2 = DirCosineZ1 * CONST6 + Object.EField * T * CONST5 / sqrt(E)
            R2 = random_uniform(RandomSeed)
            GasIndex = 0
            for GasIndex in range(Object.NumberOfGases):
                if Object.MaxCollisionFreqTotalG[GasIndex] >= R2:
                    break
            IMBPT += 1
            if (IMBPT > 6):
                GenerateMaxBoltz(Object.RandomSeed, Object.RandomMaxBoltzArray)
                IMBPT = 1
            VGX = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1)]
            IMBPT += 1
            VGY = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1)]
            IMBPT += 1
            VGZ = Object.VTMB[GasIndex] * Object.RandomMaxBoltzArray[(IMBPT - 1)]
            # CALCULATE ELECTRON VELOCITY VECTORS VEX VEY VEZ
            VEX = DirCosineX2 * CONST9 * sqrt(E)
            VEY = DirCosineY2 * CONST9 * sqrt(E)
            VEZ = DirCosineZ2 * CONST9 * sqrt(E)

            EOK = (pow((VEX - VGX), 2) + pow((VEY - VGY), 2) + pow((VEZ - VGZ), 2)) / CONST10
            IE = int(EOK / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            R5 = random_uniform(RandomSeed)
            Test1 = Object.TotalCollisionFrequency[GasIndex][IE] / Object.MaxCollisionFreq[GasIndex]
            if R5 <= Test1:
                break

        if IE == 3999:
            return 1

        TDASH = 0.0
        A = AP * T
        B = BP * T * T
        CONST11 = 1 / (CONST9 * sqrt(EOK))
        DXCOM = (VEX - VGX) * CONST11
        DYCOM = (VEY - VGY) * CONST11
        DZCOM = (VEZ - VGZ) * CONST11
        Object.Z += DirCosineZ1 * A + T * T * F1
        Object.TimeSum += T
        Object.VelocityZ = Object.Z/Object.TimeSum
        # Determination of real collision type
        R3 = random_uniform(RandomSeed)
        # Find location within 4 units in collision array
        I = MBSortT(GasIndex, I, R3, IE, Object)
        while Object.CollisionFrequency[GasIndex][IE][I] < R3:
            I += 1
        S1 = Object.RGAS[GasIndex][I]
        EI = Object.EnergyLevels[GasIndex][I]

        if Object.IPN[GasIndex][I] > 0:
            R9 = random_uniform(RandomSeed)
            EXTRA = R9 * (EOK - EI)
            EI = EXTRA + EI
        IPT = Object.IARRY[GasIndex][I]
        if EOK < EI:
            EI = EOK - 0.0001
        S2 = pow(S1, 2) / (S1 - 1.0)

        # Anisotropic scattering
        R3 = random_uniform(RandomSeed)
        if Object.INDEX[GasIndex][I] == 1:
            R31 = random_uniform(RandomSeed)
            F3 = 1.0 - R3 * Object.AngleCut[GasIndex][IE][I]
            if R31 > Object.ScatteringParameter[GasIndex][IE][I]:
                F3 = -1.0 * F3
        elif Object.INDEX[GasIndex][I] == 2:
            EPSI = Object.ScatteringParameter[GasIndex][IE][I]
            F3 = 1.0 - (2.0 * R3 * (1.0 - EPSI) / (1.0 + EPSI * (1.0 - 2.0 * R3)))
        else:
            # Isotropic scattering
            F3 = 1.0 - 2.0 * R3

        THETA0 = acos(F3)
        R4 = random_uniform(RandomSeed)
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
        Object.AngleFromZ = asin(Q * sin(THETA0))

        F6 = cos(Object.AngleFromZ)
        U = (S1 - 1.0) * (S1 - 1.0) / ARG1
        CSQD = pow(F3, 2)

        if F3 < 0 and CSQD > U:
            F6 = -1 * F6
        F5 = sin(Object.AngleFromZ)
        DZCOM = min(DZCOM, 1.0)
        ARGZ = sqrt(DXCOM * DXCOM + DYCOM * DYCOM)
        if ARGZ == 0:
            DirCosineZ1 = F6
            DirCosineX1 = F9 * F5
            DirCosineY1 = F8 * F5
        else:
            DirCosineZ1 = DZCOM * F6 + ARGZ * F5 * F8
            DirCosineY1 = DYCOM * F6 + (F5 / ARGZ) * (DXCOM * F9 - DYCOM * DZCOM * F8)
            DirCosineX1 = DXCOM * F6 - (F5 / ARGZ) * (DYCOM * F9 + DXCOM * DZCOM * F8)

        # Transform velocity vectors to lab frame
        CONST12 = CONST9 * sqrt(E1)
        VXLAB = DirCosineX1 * CONST12 + VGX
        VYLAB = DirCosineY1 * CONST12 + VGY
        VZLAB = DirCosineZ1 * CONST12 + VGZ
        # Calculate energy and direction cosines in lab frame
        E1 = (VXLAB * VXLAB + VYLAB * VYLAB + VZLAB * VZLAB) / CONST10
        CONST11 = 1.0 / (CONST9 * sqrt(E1))
        DirCosineX1 = VXLAB * CONST11
        DirCosineY1 = VYLAB * CONST11
        DirCosineZ1 = VZLAB * CONST11

    return 0
