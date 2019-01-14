# TODO: figure out object.LAST
def SORT(I, R2, IE, object):
    ISTEP = int(object.ISIZE)
    INCR = 0
    I = 0
    for K in range(12):
        I = INCR
        if ISTEP == 2:
            return I
        I = INCR + ISTEP
        if I <= object.LAST:
            if object.CF[IE][I] < R2:
                INCR = INCR + ISTEP
        ISTEP = int(ISTEP / 2)

    return I