from libc.math cimport sin, cos, acos, asin, log, sqrt, pow, tan, atan
from PyBoltz.Boltz cimport Boltz


cpdef run(Boltz Object):
    '''
    Calculate the distribution function using Friedland technique, J.Friedland Physics of Fluids 20(1461)1977
    Use distribution function to calculate average energy ionisation rate and attachment rate
    :param Object: 
    :return: 
    '''
    cdef double ALFBAR=0.0,ATTBAR=0.0,EBAR=0.0,FSUM=0.0,TCFSUM = 0.0,FR[4000]
    cdef int I =0,GasIndex
    for I in range(4000):
        TCFSUM = 0.0
        for GasIndex in range(Object.NumberOfGases):
            TCFSUM+=Object.TotalCollisionFrequency[GasIndex][I]
        FR[I] = Object.CollisionEnergies[I]/TCFSUM
        EBAR +=  Object.E[I]*  FR[I]
        ALFBAR += Object.IonCollisionFreq[I] * FR[I]
        ATTBAR += Object.AttCollisionFreq[I] * FR[I]
        FSUM+=FR[I]

    for I in range(4000):
        FR[I]/=FSUM

    EBAR /= Object.TotalTimeSecondary
    ALFBAR /= Object.TotalTimeSecondary
    ATTBAR /= Object.TotalTimeSecondary

    print("Estimate using Friedland:\nAverage Energy = {:^5.5f} Ev\nAverage Ionisation = {:^5.5f} * 10**12/Sec".format(EBAR,ALFBAR))
    print("Average Attachment = {:^5.5f} *10**12/Sec\n".format(ATTBAR))
    return
