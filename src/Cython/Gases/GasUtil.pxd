cdef double QLSCALE(double EN,int n,double Y[], double X[])

cdef double CALPQ3(double EN,int n,double Y[], double X[])

cdef double CALQIONX(double EN,int n, double Y[],double X[],double BETA2,double CONST1,double CONST,double DEN,double C,double AM2)

cdef double CALQION(double EN,int n, double Y[],double X[])

cdef double CALQINP(double EN, int n, double Y[],double X[],double pow)

cdef double CALQIN(double EN, int n, double Y[],double X[])

cdef double CALQINBEF(double EN, int n, double Y[], double X[], double BETA2, double GAMMA2, double EMASS2, double DEN,
                      double BBCONST, double EIN, double E, double SCA)

cdef double CALQINVISO(double EN, int n, double Y[], double X[],double APOP,double EIN2,double DEG,double EIN1,double CONST)

cdef double CALXMTVANISO(double EN, int n, double Y[], double X[],double EIN, double APOP,double RAT,double CONST)

cdef double CALQINVANISO(double EN, int n, double Y[], double X[],double EIN, double APOP,double RAT,double CONST)

cdef double CALQINVISELA(double EN, int n, double Y[], double X[], double APOP, double EIN2, double DEG, double EIN1,
                    double CONST,int EFACFLAG)

cdef double CALQIONREG(double EN, int n, double Y[], double X[])
