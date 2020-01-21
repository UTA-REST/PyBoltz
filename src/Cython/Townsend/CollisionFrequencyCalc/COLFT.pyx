from PyBoltz cimport PyBoltz

cpdef run(PyBoltz Object):

    # Calculate the frequency for the different types of collisions
    cdef int NINEL=0, NELA=0, NATT=0, NION=0,NREAL=0,J
    cdef double FREQ,  FREEL,  FREION,  FREATT,  FREIN,  NTOTAL

    NTOTAL  = 0

    for J in range(Object.NumberOfGases):
        NINEL += <int>(Object.CollisionsPerGasPerType[J][3]+ Object.CollisionsPerGasPerType[J][4])
        # Elastic
        NELA += <int>Object.CollisionsPerGasPerType[J][0]
        # Attachment
        NATT += <int>Object.CollisionsPerGasPerType[J][2]
        # Ionisation
        NION += <int>Object.CollisionsPerGasPerType[J][1]

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

