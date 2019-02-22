from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow

cdef double QLSCALE(double EN, int n, double Y[], double X[]):
    cdef double YXJ, YXJ1, XNJ, XNJ1, A, B, R
    cdef int J
    for J in range(1, n):
        if EN <= X[J]:
            break
    YXJ = log(Y[J])
    YXJ1 = log(Y[J - 1])
    XNJ = log(X[J])
    XNJ1 = log(X[J - 1])
    A = (YXJ - YXJ1) / (XNJ - XNJ1)
    B = (XNJ1 * YXJ - XNJ * YXJ1) / (XNJ1 - XNJ)
    R = exp(A * log(EN) + B) * 1.e-16
    return R

cdef double CALPQ3(double EN, int n, double Y[], double X[]):
    cdef double A, B, R
    cdef int J
    for J in range(1, n):
        if EN <= X[J]:
            break
    A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
    B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
    return A * EN + B

cdef double CALQIONX(double EN, int n, double Y[], double X[], double BETA2, double CONST1, double CONST, double DEN,
                     double C, double AM2):
    cdef double A, B, X1, X2
    cdef int J
    if EN <= X[n - 1]:
        for J in range(1, n):
            if EN <= X[J]:
                break
        A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
        B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
        return (A * EN + B) * 1.0e-16
    else:
        #USE BORN-BETHE X-SECTION ABOVE X[n] EV
        X2 = 1 / BETA2
        X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
        return CONST * (AM2 * (X1 - DEN / 2) + C * X2) * CONST1

cdef double CALQION(double EN, int n, double Y[], double X[]):
    cdef double A, B, X1, X2
    cdef int J
    if EN <= X[n - 1]:
        for J in range(1, n):
            if EN <= X[J]:
                break
        A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
        B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
        return (A * EN + B) * 1.0e-16
    return 0

cdef double CALQINP(double EN, int n, double Y[], double X[], double pow):
    cdef double A, B, X1, X2
    cdef int J
    if EN <= X[n - 1]:
        for J in range(1, n):
            if EN <= X[J]:
                break
        A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
        B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
        return (A * EN + B) * 1.0e-18
    else:
        return Y[n - 1] * (X[n - 1] / EN) ** pow * 1.0e-18

cdef double CALQIN(double EN, int n, double Y[], double X[]):
    cdef double A, B, X1, X2
    cdef int J
    if EN <= X[n - 1]:
        for J in range(1, n):
            if EN <= X[J]:
                break
        A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
        B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
        return (A * EN + B) * 1.0e-18
    return 0.0
