class MonteGpu {

public:
  double PElectronEnergyStep, PMaxCollisionFreqTotal, PEField,  PCONST1, PCONST2, PCONST3;
  double Ppi, PNumMomCrossSectionPoints, PMaxCollisionFreq, PAngleFromZ,  PAngleFromX;
  double PInitialElectronEnergy;
  void  ** PRGAS,** PCollisionFrequency,  *PTotalCollisionFrequency,  ** PEnergyLevels,* PVTMB,* PISIZE;
  void ** PAngleCut, ** PScatteringParameter,  * PINDEX,  * PIPN, * output;

  MonteGpu(){}
  ~MonteGpu(){}
  void MonteTGpu();
};
