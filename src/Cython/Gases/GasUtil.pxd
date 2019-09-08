cdef double QLSCALE(double EN,int n,double Y[], double X[])

cdef double CALPQ3(double EN,int n,double Y[], double X[])

cdef double CALIonizationCrossSectionX(double EN,int n, double Y[],double X[],double BETA2,double CONST1,double CONST,double DEN,double C,double AM2)

cdef double CALIonizationCrossSection(double EN,int n, double Y[],double X[])

cdef double CALInelasticCrossSectionPerGasP(double EN, int n, double Y[],double X[],double pow)

cdef double CALInelasticCrossSectionPerGas(double EN, int n, double Y[],double X[])

cdef double CALInelasticCrossSectionPerGasBEF(double EN, double ENP, int n, double Y[], double X[], double BETA2, double GAMMA2, double ElectronMass2, double DEN,
                      double BBCONST, double EnergyLevels, double E, double SCA)

cdef double CALInelasticCrossSectionPerGasVISO(double EN, int n, double Y[], double X[],double APOP,double EnergyLevels2,double DEG,double EnergyLevels1,double CONST)

cdef double CALXMTVAAnisotropicDetected(double EN, int n, double Y[], double X[],double EnergyLevels, double APOP,double RAT,double CONST)

cdef double CALInelasticCrossSectionPerGasVAAnisotropicDetected(double EN, int n, double Y[], double X[],double EnergyLevels, double APOP,double RAT,double CONST)

cdef double CALInelasticCrossSectionPerGasVISELA(double EN, int n, double Y[], double X[], double APOP, double EnergyLevels2, double DEG, double EnergyLevels1,
                    double CONST,int EFACFLAG)

cdef double CALIonizationCrossSectionREG(double EN, int n, double Y[], double X[])
