from Magboltz cimport Magboltz
cimport numpy as np
import  numpy as np

# TODO: figure out Magboltz.LAST
cdef long long SORT(I, R2, IE, Magboltz):
    
    ISTEP = int(Magboltz.ISIZE)
    INCR = 0
    I = 0
    for K in range(12):
        I = INCR
        if ISTEP == 2:
            return I
        I = INCR + ISTEP
        if I <= Magboltz.LAST:
            if Magboltz.CF[IE][I] < R2:
                INCR = INCR + ISTEP
        ISTEP = int(ISTEP / 2)

    return I

# TODO: figure out Magboltz.LAST
from Magboltz cimport Magboltz
cimport numpy as np
import  numpy as np
cdef long long SORTT(int KGAS, int I, double R2, int IE,Magboltz Object):
    cdef long long ISTEP,INCR
    cdef int K
    ISTEP = long(Object.ISIZE[KGAS])
    INCR = 0
    for K in range(12):
        I = INCR
        if ISTEP == 2:
            return I
        I = INCR + ISTEP
        if I <= Object.IPLAST[KGAS]:
            if Object.CF[KGAS][IE][I-1] < R2:
                INCR = INCR + ISTEP
        ISTEP = ISTEP / 2

    return I
