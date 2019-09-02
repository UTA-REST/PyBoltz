cdef extern from "C/RM48.h":
    double DRAND48(double dummy)
    void RM48(double lenv)

cdef double drand48(double dummy):
    return DRAND48(dummy)

def getR(seed):
    A = []
    for i in range(1000000):
        A.append(drand48(0.666))
    return A
