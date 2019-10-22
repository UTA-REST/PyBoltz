class MonteGpu {

public:
  double PElectronEnergyStep, PMaxCollisionFreqTotal, PEField,  PCONST1, PCONST2, PCONST3;
  double Ppi, PISIZE, PNumMomCrossSectionPoints, PMaxCollisionFreq,  * PVTMB,  PAngleFromZ,  PAngleFromX;
  double PInitialElectronEnergy, ** PCollisionFrequency,  *PTotalCollisionFrequency,  ** PRGAS,  ** PEnergyLevels;
  double ** PAngleCut, ** PScatteringParameter,  * PINDEX,  * PIPN, * output;

  MonteGpu(){}
  ~MonteGpu(){}

  void setPElectronEnergyStep(double EnergyStep);
  void MonteTGpu();
};
