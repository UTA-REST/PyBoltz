from Magboltz cimport Magboltz
cimport numpy as np
import  numpy as np

# TODO: figure out Magboltz.LAST
cdef long long SORT(int I, double R2, int IE,Magboltz Object):
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
            return I
        I = INCR + ISTEP
        if I <= Object.IPLASTNT:
            if Object.CFNT[IE][I - 1] < R2:
                INCR = INCR + ISTEP
        ISTEP = ISTEP / 2

    return I
