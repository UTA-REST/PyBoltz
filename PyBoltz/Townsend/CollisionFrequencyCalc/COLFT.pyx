from Boltz cimport Boltz

cpdef run(Boltz Object):
    # Calculate the frequency for the different types of collisions
    cdef int NumberOfInelastic=0, NumberOfElastic=0, NumberOfAttachment=0, NumberOfIon=0,NumberOfReal=0,J
    cdef double FrequencyOfRealColli,  FrequencyOfElasticColli,  FrequencyOfIonisationColli,  FrequencyOfAttachmentColli,  FrequencyOfInelasticColli
    cdef int TotalSumOfColli

    TotalSumOfColli  = 0

    for J in range(Object.NumberOfGases):
        # Inelastic
        NumberOfInelastic += <int>(Object.CollisionsPerGasPerType[J][3]+ Object.CollisionsPerGasPerType[J][4])
        # Elastic
        NumberOfElastic += <int>Object.CollisionsPerGasPerType[J][0]
        # Attachment
        NumberOfAttachment += <int>Object.CollisionsPerGasPerType[J][2]
        # Ionisation
        NumberOfIon += <int>Object.CollisionsPerGasPerType[J][1]

    TotalSumOfColli = NumberOfElastic+NumberOfInelastic+NumberOfAttachment+NumberOfIon
    NumberOfReal = TotalSumOfColli

    if Object.TotalTimeSecondary== 0.0:
        Object.TotalTimeSecondary = Object.TimeSum

    FrequencyOfRealColli = NumberOfReal / Object.TotalTimeSecondary
    FrequencyOfInelasticColli = NumberOfInelastic / Object.TotalTimeSecondary
    FrequencyOfElasticColli = NumberOfElastic / Object.TotalTimeSecondary
    FrequencyOfIonisationColli = NumberOfIon / Object.TotalTimeSecondary
    FrequencyOfAttachmentColli = NumberOfAttachment / Object.TotalTimeSecondary

    return FrequencyOfRealColli, FrequencyOfElasticColli, FrequencyOfIonisationColli, FrequencyOfAttachmentColli, FrequencyOfInelasticColli, TotalSumOfColli

