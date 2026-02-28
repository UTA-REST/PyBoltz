from __future__ import print_function
import math
import random
import numpy as np
from PyBoltzPy.MBSorts import MBSort


def EnergyLimit(Object):
    """
    Calculate the upper electron energy limit by simulating collisions.
    Returns 1 if energy goes out of range (need higher Max_Electron_Energy), 0 otherwise.
    No magnetic field, no thermal motion variant.
    """
    CONST5 = Object.CONST3 / 2.0
    ISAMP = 10
    SMALL = 1.0e-20
    I = 0
    E1 = Object.InitialElectronEnergy
    N4000 = 4000
    TDASH = 0.0
    INTEM = 8

    TEMP = np.zeros(4000)
    for J in range(N4000):
        TEMP[J] = Object.TotalCollisionFrequencyNullNT[J] + Object.TotalCollisionFrequencyNT[J]

    # Initial direction cosines
    DirCosineZ1 = math.cos(Object.AngleFromZ)
    DirCosineX1 = math.sin(Object.AngleFromZ) * math.cos(Object.AngleFromX)
    DirCosineY1 = math.sin(Object.AngleFromZ) * math.sin(Object.AngleFromX)

    BP = (Object.EField ** 2) * Object.CONST1
    F1 = Object.EField * Object.CONST2
    F2 = Object.EField * Object.CONST3
    F4 = 2 * math.acos(-1)
    DELTAE = Object.Max_Electron_Energy / float(INTEM)
    E1 = Object.InitialElectronEnergy
    J2M = Object.MaxNumberOfCollisions / ISAMP

    for J1 in range(int(J2M)):
        if J1 != 0 and J1 >= 10 and not int(str(J1)[-int(math.log10(J1)):]) and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(
                Object.Max_Electron_Energy, J1), end='\r')
        elif J1 >= 1e6 and J1 % 1e5 == 0 and Object.Console_Output_Flag:
            print('Trying {:^4.4f} Ev for final electron energy - Num analyzed collisions: {}'.format(
                Object.Max_Electron_Energy, J1), end='\r')

        while True:
            R1 = random.random()
            I = int(E1 / DELTAE) + 1
            I = min(I, INTEM) - 1
            TLIM = Object.MaxCollisionFreqNT[I]
            T = -1 * math.log(R1) / TLIM + TDASH
            TDASH = T
            AP = DirCosineZ1 * F2 * math.sqrt(E1)
            E = E1 + (AP + BP * T) * T
            IE = int(E / Object.ElectronEnergyStep)
            IE = min(IE, 3999)
            if TEMP[IE] > TLIM:
                TDASH += math.log(R1) / TLIM
                Object.MaxCollisionFreqNT[I] *= 1.05
                continue

            # Test for null collisions
            R5 = random.random()
            Test1 = Object.TotalCollisionFrequencyNT[IE] / TLIM
            if R5 <= Test1:
                break

        if IE == 3999:
            return 1

        # Calculate direction cosines at instant before collision
        TDASH = 0.0
        CONST6 = math.sqrt(E1 / E)
        DirCosineX2 = DirCosineX1 * CONST6
        DirCosineY2 = DirCosineY1 * CONST6
        DirCosineZ2 = DirCosineZ1 * CONST6 + Object.EField * T * CONST5 / math.sqrt(E)
        R2 = random.random()

        # Determination of real collision type
        I = MBSort(I, R2, IE, Object)
        while Object.CollisionFrequencyNT[IE][I] < R2:
            I = I + 1

        S1 = Object.RGasNT[I]
        EI = Object.EnergyLevelsNT[I]
        if Object.ElectronNumChangeNT[I] > 0:
            R9 = random.random()
            EXTRA = R9 * (E - EI)
            EI = EXTRA + EI

        # Generate scattering angles
        IPT = Object.InteractionTypeNT[I]
        if E < EI:
            EI = E - 0.0001
        S2 = (S1 * S1) / (S1 - 1.0)
        R3 = random.random()

        if Object.AngularModelNT[I] == 1:
            R31 = random.random()
            CosTheta = 1.0 - R3 * Object.AngleCutNT[IE][I]
            if R31 > Object.ScatteringParameterNT[IE][I]:
                CosTheta = -1 * CosTheta
        elif Object.AngularModelNT[I] == 2:
            EPSI = Object.ScatteringParameterNT[IE][I]
            CosTheta = 1 - (2 * R3 * (1 - EPSI) / (1 + EPSI * (1 - 2 * R3)))
        else:
            CosTheta = 1 - 2 * R3

        Theta = math.acos(CosTheta)
        R4 = random.random()
        Phi = F4 * R4
        SinPhi = math.sin(Phi)
        CosPhi = math.cos(Phi)
        ARG1 = 1 - S1 * EI / E
        ARG1 = max(ARG1, SMALL)

        D = 1 - CosTheta * math.sqrt(ARG1)
        E1 = E * (1 - EI / (S1 * E) - 2 * D / S2)
        E1 = max(E1, SMALL)
        Q = math.sqrt((E / E1) * ARG1) / S1
        Q = min(Q, 1)
        Object.AngleFromZ = math.asin(Q * math.sin(Theta))

        CosZAngle = math.cos(Object.AngleFromZ)
        U = (S1 - 1) * (S1 - 1) / ARG1
        CosSquareTheta = CosTheta ** 2

        if CosTheta < 0 and CosSquareTheta > U:
            CosZAngle = -1 * CosZAngle

        SinZAngle = math.sin(Object.AngleFromZ)
        DirCosineZ2 = min(DirCosineZ2, 1)
        ARGZ = math.sqrt(DirCosineX2 * DirCosineX2 + DirCosineY2 * DirCosineY2)
        if ARGZ == 0:
            DirCosineZ1 = CosZAngle
            DirCosineX1 = CosPhi * SinZAngle
            DirCosineY1 = SinPhi * SinZAngle
        else:
            DirCosineZ1 = DirCosineZ2 * CosZAngle + ARGZ * SinZAngle * SinPhi
            DirCosineY1 = (DirCosineY2 * CosZAngle +
                           (SinZAngle / ARGZ) * (DirCosineX2 * CosPhi - DirCosineY2 * DirCosineZ2 * SinPhi))
            DirCosineX1 = (DirCosineX2 * CosZAngle -
                           (SinZAngle / ARGZ) * (DirCosineY2 * CosPhi + DirCosineX2 * DirCosineZ2 * SinPhi))

    return 0
