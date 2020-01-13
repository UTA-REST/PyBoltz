from PyBoltz cimport PyBoltz

cdef void COLF(double *FREQ, double *FREEL, double *FREION, double *FREATT, double *FREIN, double *NTOTAL,
               PyBoltz Object):

    # Calculate the frequency for the different types of collisions
    cdef int NINEL, NELA, NATT, NION,NREAL

    NINEL = <int>(Object.InteractionTypeNT[3] + Object.InteractionTypeNT[4] + Object.InteractionTypeNT[8] + Object.InteractionTypeNT[
        9] + Object.InteractionTypeNT[13] + Object.InteractionTypeNT[14] + Object.InteractionTypeNT[18] + \
            Object.InteractionTypeNT[19] + Object.InteractionTypeNT[23] + Object.InteractionTypeNT[24] + \
            Object.InteractionTypeNT[28] + Object.InteractionTypeNT[29])
    # Elastic
    NELA = <int>(Object.InteractionTypeNT[0] + Object.InteractionTypeNT[5] + Object.InteractionTypeNT[10] + Object.InteractionTypeNT[
        15] + Object.InteractionTypeNT[20] + Object.InteractionTypeNT[25])
    # Attachment
    NATT = <int>(Object.InteractionTypeNT[2] + Object.InteractionTypeNT[7] + Object.InteractionTypeNT[12] + Object.InteractionTypeNT[
        17] + Object.InteractionTypeNT[22] + Object.InteractionTypeNT[27])
    # Ionisation
    NION = <int>(Object.InteractionTypeNT[1] + Object.InteractionTypeNT[6] + Object.InteractionTypeNT[11] + Object.InteractionTypeNT[
        16] + Object.InteractionTypeNT[21] + Object.InteractionTypeNT[26])

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
