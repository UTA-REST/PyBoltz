from PyBoltz cimport PyBoltz

cpdef run(PyBoltz Object):

    # Calculate the frequency for the different types of collisions
    cdef int NINEL, NELA, NATT, NION,NREAL
    cdef double FREQ,  FREEL,  FREION,  FREATT,  FREIN,  NTOTAL,
    NINEL = <int>(Object.CollisionsPerGasPerTypeNT[3] + Object.CollisionsPerGasPerTypeNT[4] + Object.CollisionsPerGasPerTypeNT[8] + Object.CollisionsPerGasPerTypeNT[
        9] + Object.CollisionsPerGasPerTypeNT[13] + Object.CollisionsPerGasPerTypeNT[14] + Object.CollisionsPerGasPerTypeNT[18] + \
            Object.CollisionsPerGasPerTypeNT[19] + Object.CollisionsPerGasPerTypeNT[23] + Object.CollisionsPerGasPerTypeNT[24] + \
            Object.CollisionsPerGasPerTypeNT[28] + Object.CollisionsPerGasPerTypeNT[29])
    # Elastic
    NELA = <int>(Object.CollisionsPerGasPerTypeNT[0] + Object.CollisionsPerGasPerTypeNT[5] + Object.CollisionsPerGasPerTypeNT[10] + Object.CollisionsPerGasPerTypeNT[
        15] + Object.CollisionsPerGasPerTypeNT[20] + Object.CollisionsPerGasPerTypeNT[25])
    # Attachment
    NATT = <int>(Object.CollisionsPerGasPerTypeNT[2] + Object.CollisionsPerGasPerTypeNT[7] + Object.CollisionsPerGasPerTypeNT[12] + Object.CollisionsPerGasPerTypeNT[
        17] + Object.CollisionsPerGasPerTypeNT[22] + Object.CollisionsPerGasPerTypeNT[27])
    # Ionisation
    NION = <int>(Object.CollisionsPerGasPerTypeNT[1] + Object.CollisionsPerGasPerTypeNT[6] + Object.CollisionsPerGasPerTypeNT[11] + Object.CollisionsPerGasPerTypeNT[
        16] + Object.CollisionsPerGasPerTypeNT[21] + Object.CollisionsPerGasPerTypeNT[26])

    NTOTAL = NELA+NINEL+NATT+NION

    if Object.TotalTimeSecondary== 0.0:
        NREAL = <int>NTOTAL
        Object.TotalTimeSecondary = Object.TimeSum
    else:
        NREAL = <int>NTOTAL

    FREQ = NREAL/Object.TotalTimeSecondary
    FREIN = NINEL/Object.TotalTimeSecondary
    FREEL = NELA/Object.TotalTimeSecondary
    FREION = NION/Object.TotalTimeSecondary
    FREATT = NATT/Object.TotalTimeSecondary
    return FREQ, FREEL, FREION, FREATT, FREIN, NTOTAL
