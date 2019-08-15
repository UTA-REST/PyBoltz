# TODO: figure out Magboltz.LAST
def SORTT(KGAS, I, R2, IE, Magboltz):
    
    ISTEP = Magboltz.ISIZE[KGAS]
    INCR = 0
    for K in range(12):
        I = INCR
        if ISTEP == 2:
            return I
        I = INCR + ISTEP
        if I <= Magboltz.LAST[KGAS]:
            if Magboltz.CF[KGAS][IE][I] < R2:
                INCR = INCR + ISTEP
        ISTEP = ISTEP / 2

    return int(I)
