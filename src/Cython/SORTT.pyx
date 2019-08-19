# TODO: figure out Magboltz.LAST
from Magboltz cimport Magboltz
cimport numpy as np
import  numpy as np
cdef long long SORTT(int KGAS, int I, double R2, int IE,Magboltz Object):
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
    ISTEP = long(Object.ISIZE[KGAS])
    INCR = 0
    for K in range(12):
        I = INCR -1
        if ISTEP == 2:
            return I
        I = INCR + ISTEP -1
        if I <= Object.IPLAST[KGAS]:
            if Object.CF[KGAS][IE][I] < R2:
                INCR = INCR + ISTEP
        ISTEP = ISTEP / 2

    return I
