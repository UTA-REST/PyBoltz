// host Object, this is the object on the cpu.
class MonteGpu {

public:
  double ElectronEnergyStep, MaxCollisionFreqTotal, EField,  CONST1, CONST2, CONST3;
  double pi,   AngleFromZ,  AngleFromX;
  double InitialElectronEnergy;
  double  *MaxCollisionFreq,*NumMomCrossSectionPoints,*RGAS,* CollisionFrequency,  *TotalCollisionFrequency,  * EnergyLevels,* VTMB,* ISIZE;
  double *AngleCut, * ScatteringParameter,  * INDEX,  * IPN, * output;

  MonteGpu(){}
  ~MonteGpu(){}
  void MonteTGpu();
};


// Device object, this is the object on the cpu.
class MonteGpuDevice {
public:
  double * ElectronEnergyStep, *MaxCollisionFreqTotal, *EField,  *CONST1, *CONST2, *CONST3;
  double *pi,   *AngleFromZ,  *AngleFromX,*BP,*F1,*F2,*Sqrt2M,*TwoM,*TwoPi;
  double *InitialElectronEnergy;
  double * MaxCollisionFreq,*NumMomCrossSectionPoints,* RGAS,* CollisionFrequency,  *TotalCollisionFrequency,  * EnergyLevels,* VTMB,* ISIZE;
  double * AngleCut, * ScatteringParameter,  * INDEX,  * IPN, * Output,*X,*Y,*Z,*TimeSum,*DirCosineZ1,*DirCosineX1,*DirCosineY1;
  double * EBefore,*iEnergyBins,*COMEnergy,*VelocityX,*VelocityY,*VelocityZ,*GasVelX,*GasVelY,*GasVelZ,*T,*AP;
};
