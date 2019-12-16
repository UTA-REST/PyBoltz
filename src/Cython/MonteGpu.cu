#include "MonteGpu.hh"
#include <stdio.h>

extern __global__ void MonteTRun(MonteGpuDevice * DP);

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

long long * SetupArrayOneValLL(long long val,int s){
  long long * pointer;
  cudaMalloc((void **)&pointer,s*sizeof(long long));
  long long * temp = (long long *)malloc(s*sizeof(long long));
  for(int i=0;i<s;++i){
    temp[i] = val;
  }
  cudaMemcpy(pointer,temp,s*sizeof(long long),cudaMemcpyHostToDevice);
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
  DeviceParameters->XOutput = SetupArrayOneVal(0,100000);
  DeviceParameters->YOutput = SetupArrayOneVal(0,100000);
  DeviceParameters->ZOutput = SetupArrayOneVal(0,100000);
  DeviceParameters->TimeSumOutput = SetupArrayOneVal(0,100000);
  DeviceParameters->GasIndex = SetupArrayOneValLL(0,1000);
  DeviceParameters->MaxCollisionFreqTotalG = SetupAndCopyDouble(MaxCollisionFreqTotalG,6);
  DeviceParameters->NumberOfGases =  SetupAndCopyDouble(&(NumberOfGases),1);
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
  cudaFree(DeviceParameters->XOutput);
  cudaFree(DeviceParameters->YOutput);
  cudaFree(DeviceParameters->ZOutput);
  cudaFree(DeviceParameters->TimeSumOutput);
  cudaFree(DeviceParameters->GasIndex);
  cudaFree(DeviceParameters->NumberOfGases);
  cudaFree(DeviceParameters->MaxCollisionFreqTotalG);
}

// function that will be called from the PyBoltz_Gpu classoutput
void MonteGpu::MonteRunGpu(){
  MonteGpuDevice * DeviceParametersPointer;

  cudaMalloc((void **)&DeviceParametersPointer,sizeof(MonteGpuDevice));
  cudaMemcpy(DeviceParametersPointer,DeviceParameters,sizeof(MonteGpuDevice),cudaMemcpyHostToDevice);
  printf("%d %d ....\n",numElectrons,NumColls);
  MonteTRun<<<blocks,threads>>>(DeviceParametersPointer);
  //Test<<<threads,blocks>>>(DeviceParametersPointer,DeviceParameters->RGAS);
  cudaDeviceSynchronize();
  cudaMemcpy(XOutput,DeviceParameters->XOutput,100000*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(YOutput,DeviceParameters->YOutput,100000*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(ZOutput,DeviceParameters->ZOutput,100000*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(TimeSumOutput,DeviceParameters->TimeSumOutput,100000*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(RGAS,DeviceParameters->RGAS,6*290*sizeof(double),cudaMemcpyDeviceToHost);
  //FreeRM48GensCuda<<<int(1000),1>>>(pointer);
  cudaFree(DeviceParametersPointer);
}
