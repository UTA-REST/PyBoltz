#include "MonteGpu.hh"
#include <stdio.h>


// cudamalloc functions

double * SetupAndCopyDouble(double * data,int s){
  double * pointer;
  cudaMalloc((void **)&pointer,s*sizeof(double));
  cudaMemcpy(pointer,data,s*sizeof(double),cudaMemcpyHostToDevice);
  return pointer;
}

long long *SetupAndCopyllong(long long * data,int s){
  long long * pointer;
  cudaMalloc((void **)&pointer,s*sizeof(long long));
  cudaMemcpy(pointer,data,s*sizeof(long long),cudaMemcpyHostToDevice);
  return pointer;
}

double * SetupArrayOneVal(double val,int s){
  double * pointer;
  cudaMalloc((void **)&pointer,s*sizeof(double));
  double * temp = (double *)malloc(s*sizeof(double));
  for(int i=0;i<s;++i){
    temp[i] = val;
  }
  cudaMemcpy(pointer,temp,s*sizeof(double),cudaMemcpyHostToDevice);
  free(temp);
  return pointer;
}

void MonteGpu::Setup(){
  DeviceParameters = new MonteGpuDevice();
  DeviceParameters->EnergyLevels = SetupAndCopyDouble(EnergyLevels,6*290);
  // Copying constants into device
  DeviceParameters->ElectronEnergyStep = SetupAndCopyDouble(&(ElectronEnergyStep),1);
  DeviceParameters->MaxCollisionFreqTotal = SetupAndCopyDouble(&(MaxCollisionFreqTotal),1);
  double bp = EField*EField*CONST1;
  DeviceParameters->BP = SetupAndCopyDouble(&(bp),1);
  double f1 = EField*CONST2;
  DeviceParameters->F1 = SetupAndCopyDouble(&(f1),1);
  double f2 = EField*CONST3;
  DeviceParameters->F2 = SetupAndCopyDouble(&(f2),1);
  double sqrt2m = CONST3*0.01;
  DeviceParameters->Sqrt2M = SetupAndCopyDouble(&(sqrt2m),1);
  double twom = sqrt2m*sqrt2m;
  DeviceParameters->TwoM = SetupAndCopyDouble(&(twom),1);
  double twpi = pi*2;
  DeviceParameters->TwoPi = SetupAndCopyDouble(&(twpi),1);
  DeviceParameters->SeedsGpu = SetupAndCopyllong(SeedsGpu,numElectrons);
  DeviceParameters->NumColls = SetupAndCopyllong(&NumColls,1);
  DeviceParameters->TwoPi = SetupAndCopyDouble(&(twpi),1);
  DeviceParameters->ISIZE = SetupAndCopyDouble(ISIZE,6);
  DeviceParameters->NumMomCrossSectionPoints = SetupAndCopyDouble(NumMomCrossSectionPoints,6);
  DeviceParameters->MaxCollisionFreq = SetupAndCopyDouble(MaxCollisionFreq,6);
  //Copying arrays to device
  DeviceParameters->VTMB = SetupAndCopyDouble((VTMB),6);
  DeviceParameters->X = SetupArrayOneVal(0,1000);
  DeviceParameters->Y = SetupArrayOneVal(0,1000);
  DeviceParameters->Z = SetupArrayOneVal(0,1000);
  DeviceParameters->TimeSum = SetupArrayOneVal(0,1000);
  DeviceParameters->DirCosineZ1 = SetupArrayOneVal(cos(AngleFromZ),1000);
  DeviceParameters->DirCosineX1 = SetupArrayOneVal(sin(AngleFromZ) * cos(AngleFromX),1000);
  DeviceParameters->DirCosineY1 = SetupArrayOneVal(sin(AngleFromZ) * sin(AngleFromX),1000);
  DeviceParameters->EBefore = SetupArrayOneVal(InitialElectronEnergy,1000);
  DeviceParameters->iEnergyBins = SetupArrayOneVal(0,1000);
  DeviceParameters->COMEnergy = SetupArrayOneVal(0,1000);
  DeviceParameters->VelocityX = SetupArrayOneVal(0,1000);
  DeviceParameters->VelocityY = SetupArrayOneVal(0,1000);
  DeviceParameters->VelocityZ = SetupArrayOneVal(0,1000);
  DeviceParameters->GasVelX = SetupArrayOneVal(0,1000);
  DeviceParameters->GasVelY = SetupArrayOneVal(0,1000);
  DeviceParameters->GasVelZ = SetupArrayOneVal(0,1000);
  DeviceParameters->T = SetupArrayOneVal(0,1000);
  DeviceParameters->AP = SetupArrayOneVal(0,1000);
  DeviceParameters->AngleFromZ = SetupArrayOneVal(AngleFromZ,1000);
  DeviceParameters->CollisionFrequency = SetupAndCopyDouble(CollisionFrequency,6*4000*290);
  DeviceParameters->AngleCut = SetupAndCopyDouble(AngleCut,6*4000*290);
  DeviceParameters->ScatteringParameter = SetupAndCopyDouble(ScatteringParameter,6*4000*290);
  DeviceParameters->INDEX = SetupAndCopyDouble(INDEX,6*290);
  DeviceParameters->IPN = SetupAndCopyDouble(IPN,6*290);
  DeviceParameters->RGAS = SetupAndCopyDouble(RGAS,6*290);
  DeviceParameters->TotalCollisionFrequency = SetupAndCopyDouble(TotalCollisionFrequency,6*4000);
  DeviceParameters->Output = SetupArrayOneVal(0,400000);
}

MonteGpu::~MonteGpu(){
  cudaFree(DeviceParameters->ElectronEnergyStep);
  cudaFree(DeviceParameters->MaxCollisionFreqTotal);
  cudaFree(DeviceParameters->BP);
  cudaFree(DeviceParameters->F1);
  cudaFree(DeviceParameters->F2);
  cudaFree(DeviceParameters->Sqrt2M);
  cudaFree(DeviceParameters->TwoM);
  cudaFree(DeviceParameters->TwoPi);
  cudaFree(DeviceParameters->ISIZE);
  cudaFree(DeviceParameters->NumMomCrossSectionPoints);
  cudaFree(DeviceParameters->MaxCollisionFreq);
  cudaFree(DeviceParameters->VTMB);
  cudaFree(DeviceParameters->SeedsGpu);
  cudaFree(DeviceParameters->NumColls);
  cudaFree(DeviceParameters->X);
  cudaFree(DeviceParameters->Y);
  cudaFree(DeviceParameters->Z);
  cudaFree(DeviceParameters->TimeSum);
  cudaFree(DeviceParameters->DirCosineX1);
  cudaFree(DeviceParameters->DirCosineY1);
  cudaFree(DeviceParameters->DirCosineZ1);
  cudaFree(DeviceParameters->EBefore);
  cudaFree(DeviceParameters->iEnergyBins);
  cudaFree(DeviceParameters->COMEnergy);
  cudaFree(DeviceParameters->VelocityZ);
  cudaFree(DeviceParameters->VelocityY);
  cudaFree(DeviceParameters->VelocityX);
  cudaFree(DeviceParameters->T);
  cudaFree(DeviceParameters->GasVelX);
  cudaFree(DeviceParameters->GasVelY);
  cudaFree(DeviceParameters->GasVelZ);
  cudaFree(DeviceParameters->AP);
  cudaFree(DeviceParameters->AngleFromZ);
  cudaFree(DeviceParameters->CollisionFrequency);
  cudaFree(DeviceParameters->RGAS);
  cudaFree(DeviceParameters->EnergyLevels);
  cudaFree(DeviceParameters->AngleCut);
  cudaFree(DeviceParameters->ScatteringParameter);
  cudaFree(DeviceParameters->INDEX);
  cudaFree(DeviceParameters->IPN);
  cudaFree(DeviceParameters->Output);
}
