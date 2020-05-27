from Boltz cimport Boltz

cpdef run(Boltz Object):
    # Calculate the frequency for the different types of collisions
    cdef int NumberOfInelasticColli, NumberOfElasticColli, NumberOfAttachmentColli, NumberOfIonisationColli,NumberOfRealColli
    cdef double FrequencyOfRealColli,  FrequencyOfElasticColli,  FrequencyOfIonisationColli,  FrequencyOfAttachmentColli,  FrequencyOfInelasticColli
    cdef int TotalSumOfColli,
    NumberOfInelasticColli = <int>(Object.CollisionsPerGasPerTypeNT[3] + Object.CollisionsPerGasPerTypeNT[4] + Object.CollisionsPerGasPerTypeNT[8] + Object.CollisionsPerGasPerTypeNT[
        9] + Object.CollisionsPerGasPerTypeNT[13] + Object.CollisionsPerGasPerTypeNT[14] + Object.CollisionsPerGasPerTypeNT[18] + \
                                   Object.CollisionsPerGasPerTypeNT[19] + Object.CollisionsPerGasPerTypeNT[23] + Object.CollisionsPerGasPerTypeNT[24] + \
                                   Object.CollisionsPerGasPerTypeNT[28] + Object.CollisionsPerGasPerTypeNT[29])
    # Elastic
    NumberOfElasticColli = <int>(Object.CollisionsPerGasPerTypeNT[0] + Object.CollisionsPerGasPerTypeNT[5] + Object.CollisionsPerGasPerTypeNT[10] + Object.CollisionsPerGasPerTypeNT[
        15] + Object.CollisionsPerGasPerTypeNT[20] + Object.CollisionsPerGasPerTypeNT[25])
    # Attachment
    NumberOfAttachmentColli = <int>(Object.CollisionsPerGasPerTypeNT[2] + Object.CollisionsPerGasPerTypeNT[7] + Object.CollisionsPerGasPerTypeNT[12] + Object.CollisionsPerGasPerTypeNT[
        17] + Object.CollisionsPerGasPerTypeNT[22] + Object.CollisionsPerGasPerTypeNT[27])
    # Ionisation
    NumberOfIonisationColli = <int>(Object.CollisionsPerGasPerTypeNT[1] + Object.CollisionsPerGasPerTypeNT[6] + Object.CollisionsPerGasPerTypeNT[11] + Object.CollisionsPerGasPerTypeNT[
        16] + Object.CollisionsPerGasPerTypeNT[21] + Object.CollisionsPerGasPerTypeNT[26])

    TotalSumOfColli = NumberOfElasticColli + NumberOfInelasticColli + NumberOfAttachmentColli + NumberOfIonisationColli

    NumberOfRealColli = TotalSumOfColli

    if Object.TotalTimeSecondary== 0.0:
        Object.TotalTimeSecondary = Object.TimeSum

    FrequencyOfRealColli = NumberOfRealColli / Object.TotalTimeSecondary
    FrequencyOfInelasticColli = NumberOfInelasticColli / Object.TotalTimeSecondary
    FrequencyOfElasticColli = NumberOfElasticColli / Object.TotalTimeSecondary
    FrequencyOfIonisationColli = NumberOfIonisationColli / Object.TotalTimeSecondary
    FrequencyOfAttachmentColli = NumberOfAttachmentColli / Object.TotalTimeSecondary

    return FrequencyOfRealColli, FrequencyOfElasticColli, FrequencyOfIonisationColli, FrequencyOfAttachmentColli, FrequencyOfInelasticColli, TotalSumOfColli
