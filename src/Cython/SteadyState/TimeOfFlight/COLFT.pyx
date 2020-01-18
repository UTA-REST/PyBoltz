from PyBoltz cimport PyBoltz

cdef void COLFT(double *FREQ, double *FREEL, double *FREION, double *FREATT, double *FREIN, double *NTOTAL,
               PyBoltz Object):

    # Calculate the frequency for the different types of collisions
    cdef int NINEL=0, NELA=0, NATT=0, NION=0,NREAL=0,J
    NTOTAL[0]  = 0

    for J in range(Object.NumberOfGases):
        NINEL += <int>(Object.CollisionsPerGasPerType[J][3]+ Object.CollisionsPerGasPerType[J][4])
        # Elastic
        NELA += <int>Object.CollisionsPerGasPerType[J][0]
        # Attachment
        NATT += <int>Object.CollisionsPerGasPerType[2]
        # Ionisation
        NION += <int>Object.CollisionsPerGasPerType[1]

    NTOTAL[0] = NELA+NINEL+NATT+NION

    if Object.TotalTimeSecondary== 0.0:
        NREAL = <int>NTOTAL[0]
        Object.TotalTimeSecondary = Object.TimeSum
    else:
        NREAL = <int>NTOTAL[0]

    FREQ[0] = NREAL/Object.TotalTimeSecondary
    FREIN[0] = NINEL/Object.TotalTimeSecondary
    FREEL[0] = NELA/Object.TotalTimeSecondary
    FREION[0] = NION/Object.TotalTimeSecondary
    FREATT[0] = NATT/Object.TotalTimeSecondary

    return
