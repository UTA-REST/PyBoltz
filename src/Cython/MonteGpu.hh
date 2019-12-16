// Device object, this is the object used on the gpu.
class MonteGpuDevice {
public:
  double * ElectronEnergyStep, *MaxCollisionFreqTotal, *EField,  *CONST1, *CONST2, *CONST3;
  double *pi,   *AngleFromZ,  *AngleFromX,*BP,*F1,*F2,*Sqrt2M,*TwoM,*TwoPi;
  double *InitialElectronEnergy;
  double * MaxCollisionFreq,*NumMomCrossSectionPoints,* RGAS,* CollisionFrequency,  *TotalCollisionFrequency,  * EnergyLevels,* VTMB,* ISIZE;
  double * AngleCut, * ScatteringParameter,  * INDEX,  * IPN, *XOutput,*YOutput,*ZOutput,*TimeSumOutput,*EOutput,*X,*Y,*Z,*TimeSum,*DirCosineZ1,*DirCosineX1,*DirCosineY1;
  double * EBefore,*iEnergyBins,*COMEnergy,*VelocityX,*VelocityY,*VelocityZ,*GasVelX,*GasVelY,*GasVelZ,*T,*AP,*NumberOfGases,*MaxCollisionFreqTotalG;
  long long *NumColls,*GasIndex;
  long long * SeedsGpu;
};
// host Object, this is the object on the cpu.
class MonteGpu {

public:
  double ElectronEnergyStep, MaxCollisionFreqTotal, EField,  CONST1, CONST2, CONST3;
  double pi,   AngleFromZ,  AngleFromX;
  double InitialElectronEnergy;
  double  *MaxCollisionFreq,*NumMomCrossSectionPoints,*RGAS,* CollisionFrequency,  *TotalCollisionFrequency,  * EnergyLevels,* VTMB,* ISIZE;
  double *AngleCut, * ScatteringParameter,  * INDEX,  * IPN;
  double *XOutput,*YOutput,*ZOutput,*TimeSumOutput,*EOutput,NumberOfGases,*MaxCollisionFreqTotalG;
  long long * SeedsGpu;
  long long numElectrons,NumColls;
  int threads,blocks;
  MonteGpuDevice *  DeviceParameters;
  MonteGpu(){};
  void Setup();
  ~MonteGpu();
  void MonteRunGpu();
};
