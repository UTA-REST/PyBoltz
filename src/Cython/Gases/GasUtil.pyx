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

cdef double CALIonizationCrossSectionX(double EN, int n, double Y[], double X[], double BETA2, double CONST1, double CONST, double DEN,
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

cdef double CALIonizationCrossSectionREG(double EN, int n, double Y[], double X[]):
    cdef double A, B, X1, X2
    cdef int J
    for J in range(1, n):
        if EN <= X[J]:
            break
    A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
    B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
    return (A * EN + B) * 1.0e-16

cdef double CALIonizationCrossSection(double EN, int n, double Y[], double X[]):
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

cdef double CALInelasticCrossSectionPerGasP(double EN, int n, double Y[], double X[], double pow):
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

cdef double CALInelasticCrossSectionPerGas(double EN, int n, double Y[], double X[]):
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

cdef double CALInelasticCrossSectionPerGasBEF(double EN, double ENP, int n, double Y[], double X[], double BETA2, double GAMMA2, double ElectronMass2, double DEN,
                      double BBCONST, double EnergyLevels, double E, double SCA):
    cdef double A, B, X1, X2
    cdef int J
    if ENP <= X[n - 1]:
        for J in range(1, n):
            if ENP <= X[J]:
                break
        A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
        B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
        return (A * ENP + B) * 1.0e-18
    else:
        return SCA / (EnergyLevels * BETA2) * (log(BETA2 * GAMMA2 * ElectronMass2 / (4.0 * EnergyLevels)) - BETA2 - DEN / 2.0) * BBCONST * EN / (
                EN + EnergyLevels + E)
    return 0.0


cdef double CALInelasticCrossSectionPerGasVISO(double EN, int n, double Y[], double X[], double APOP, double EnergyLevels2, double DEG, double EnergyLevels1,
                    double CONST):
    cdef double A, B, X1, X2, EFAC, temp
    cdef int J
    if (EN + EnergyLevels2) <= X[n - 1]:
        for J in range(1, n):
            if (EN + EnergyLevels2) <= X[J]:
                break
        A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
        B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
        temp = (EN + EnergyLevels2) * (A * (EN + EnergyLevels2) + B) / EN
    else:
        temp = Y[n - 1] * (X[n - 1] / (EN + EnergyLevels2)) ** 2
    EFAC = sqrt(1.0 - (EnergyLevels1 / EN))
    temp = temp + CONST * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
    temp = temp * APOP * 1.0e-16
    temp = temp / DEG
    return temp

cdef double CALInelasticCrossSectionPerGasVISELA(double EN, int n, double Y[], double X[], double APOP, double EnergyLevels2, double DEG, double EnergyLevels1,
                    double CONST,int EFACFLAG):
    cdef double A, B, X1, X2, EFAC, temp
    cdef int J
    if (EN + EnergyLevels2) <= X[n - 1]:
        for J in range(1, n):
            if (EN + EnergyLevels2) <= X[J]:
                break
        A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
        B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
        temp = (EN + EnergyLevels2) * (A * (EN + EnergyLevels2) + B) / EN
    else:
        temp = Y[n - 1] * X[n - 1] * (EN + EnergyLevels2) /(EN*EN)
    EFAC = sqrt(1.0 - (EnergyLevels1 / EN))
    if EFACFLAG ==0:
        temp = temp + CONST * log((EFAC + 1.0) / (EFAC - 1.0)) / EN
    else:
        temp = temp + CONST * log((EFAC + 1.0) / (1.0-EFAC)) / EN
    temp = temp * APOP * 1.0e-16
    temp = temp / DEG
    return temp


cdef double CALInelasticCrossSectionPerGasVAAnisotropicDetected(double EN, int n, double Y[], double X[],double EnergyLevels, double APOP,double RAT,double CONST):
    cdef double A, B, X1, X2, EFAC, temp,ADIP,FWD,BCK,XMT,ELF
    cdef int J
    if EN <= X[n-1]:
        for J in range(1, n):
            if EN <= X[J]:
                break
        A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
        B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
        temp =  A * EN + B
    else:
        temp = Y[n-1]*(X[n-1]/EN)**2
    EFAC = sqrt(1.0 - (EnergyLevels/EN))
    ADIP = CONST * log((EFAC + 1.0) / (1.0-EFAC)) / EN
    ELF = EN -EnergyLevels
    FWD = log((EN+ELF)/(EN+ELF-2.0*sqrt(EN*ELF)))
    BCK = log((EN+ELF+2.0*sqrt(EN*ELF))/(EN+ELF))
    XMT = ((1.5-FWD/(FWD+BCK))*ADIP+RAT*temp)*APOP*1.0e-16
    temp = (temp+ADIP)*APOP*1.0e-16

    return temp

cdef double CALXMTVAAnisotropicDetected(double EN, int n, double Y[], double X[],double EnergyLevels, double APOP,double RAT,double CONST):
    cdef double A, B, X1, X2, EFAC, temp,ADIP,FWD,BCK,XMT,ELF
    cdef int J
    if EN <= X[n-1]:
        for J in range(1, n):
            if EN <= X[J]:
                break
        A = (Y[J] - Y[J - 1]) / (X[J] - X[J - 1])
        B = (X[J - 1] * Y[J] - X[J] * Y[J - 1]) / (X[J - 1] - X[J])
        temp =  A * EN + B
    else:
        temp = Y[n-1]*(X[n-1]/EN)**2
    EFAC = sqrt(1.0 - (EnergyLevels/EN))
    ADIP = CONST * log((EFAC + 1.0) / (1.0-EFAC)) / EN
    ELF = EN -EnergyLevels
    FWD = log((EN+ELF)/(EN+ELF-2.0*sqrt(EN*ELF)))
    BCK = log((EN+ELF+2.0*sqrt(EN*ELF))/(EN+ELF))

    XMT = ((1.5-FWD/(FWD+BCK))*ADIP+RAT*temp)*APOP*1.0e-16
    temp = (temp+ADIP)*APOP*1.0e-16
    return XMT
