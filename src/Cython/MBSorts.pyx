from PyBoltz cimport PyBoltz
import sys
import cython
cimport numpy as np
import  numpy as np


# TODO: figure out PyBoltz.LAST
cdef long long MBSort(int I, double R2, int IE,PyBoltz Object):
    """
    This function selects collision type from collision array by binary step sampling reduces sampling range to within 4 
    positions in array output =  i ( position within 4 of correct value).
    :param I: 
    :param R2: 
    :param IE: 
    :param Object: 
    :return: a new index 
    """
    cdef long long ISTEP,INCR
    cdef int K
    ISTEP = long(Object.ISIZENT)
    INCR = 0

    for K in range(12):
        I = INCR
        if ISTEP == 2:
            if I==0:
                return I
            return I -1

        I = INCR + ISTEP
        if I <= Object.NumMomCrossSectionPointsNT:
            if Object.CollisionFrequencyNT[IE][I - 1] < R2:
                INCR = INCR + ISTEP
        ISTEP = ISTEP / 2
    if I==0:
        return I
    return I - 1


cdef long long MBSortT(int GasIndex, int I, double R2, int IE,PyBoltz Object):
    """
    This function selects collision type from collision array by binary step sampling reduces sampling range to within 4 
    positions in array output =  i ( position within 4 of correct value).
    :param I: 
    :param R2: 
    :param IE: 
    :param Object: 
    :return: a new index 
    """

    cdef long long ISTEP,INCR
    cdef int K
    ISTEP = long(Object.ISIZE[GasIndex])

    INCR = 0
    for K in range(12):
        I = INCR
        if ISTEP == 2:
            if I==0:
                return I
            return I -1
        I = INCR + ISTEP
        if I <= Object.NumMomCrossSectionPoints[GasIndex]-1:
            if Object.CollisionFrequency[GasIndex][IE][I-1] < R2:
                INCR = INCR + ISTEP
        ISTEP = ISTEP / 2
    if I==0:
        return I
    return I -1
