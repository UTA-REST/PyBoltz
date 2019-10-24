#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include<curand_kernel.h>
#include "MonteGpu.hh"
#define min(a,b) a<b?a:b
#define max(a,b) a>b?a:b

// cudamalloc functions

double * SetupAndCopyDouble(double * data,int s){
  double * pointer;
  cudaMalloc((void **)&pointer,s*sizeof(double));
  cudaMemcpy(pointer,data,s*sizeof(double),cudaMemcpyHostToDevice);
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

__device__ int MBSortT(double RandomNum,double iEnergyBin,double * CF,double ISIZE,double NumPoints){
  int ISTEP,INCR,I;
  ISTEP = ISIZE;
  INCR = 0;
  for(int K = 0;K<12;++K){
      I = INCR;
      if(ISTEP==2){
        if(I==0){
          return I;
        }
        return I - 1;
      }
      I = INCR + ISTEP/2;
      if (I<= NumPoints){
        if(CF[(int)iEnergyBin*290+I]<RandomNum){
          INCR +=ISTEP;
        }
      }
      ISTEP = ISTEP/2;
  }
  if(I==0){
    return I;
  }
  return I - 1;
}


__device__ extern void GetCollisions(double *ElectronEnergyStep, double* MaxCollisionFreqTotal,double* BP,double*  F1,
  double*  F2,double* Sqrt2M,double* TwoM,double* TwoPi,double* MaxCollisionFreq,double * VTMB,double * TimeSum,
  double * DirCosineZ1,double * DirCosineX1,double * DirCosineY1,double * EBefore,double * iEnergyBins,
  double * COMEnergy,double * VelocityX,double * VelocityY,double * VelocityZ,double * GasVelX,double * GasVelY,double * GasVelZ,
  double * T,double * AP,double * TotalCollisionFrequency,int i,curandState* globalState){

  // function start
  int MaxBoltzNumsUsed = 0;
  curand_uniform(globalState);
  //R = curand_uniform( &state );
  double RNMX[6]={0,0,0,0,0,0};
  double TDash = 0.0,R1,R2,RandomNum,TEST;

  for(int j=0;j<5;j+=2){
    R1 = curand_uniform(globalState);
    R2 = curand_uniform(globalState);
    RNMX[j] = sqrt(-1*log(R1))*cos(R2*((*TwoPi)));
    RNMX[j+1] = sqrt(-1*log(R1))*sin(R2*((*TwoPi)));
  }

  double EAfter = 0.0,VelocityRatio,DCosineZ2,DCosineX2,DCosineY2;

  while(1){
    RandomNum = curand_uniform(globalState);
    T[i] = -1 * log(RandomNum)/(*MaxCollisionFreqTotal)+TDash;
    TDash = T[i];
    AP[i] = DirCosineZ1[i]*(*F2)*sqrt(EBefore[i]);
    EAfter = EBefore[i]+(AP[i]+(*BP)*T[i])*T[i];
    VelocityRatio = sqrt(EBefore[i]/EAfter);
    DCosineZ2 = DirCosineZ1[i] * VelocityRatio + T[i] * (*F2) / (2.0 * sqrt(EAfter));
    DCosineX2 = DirCosineX1[i] * VelocityRatio;
    DCosineY2 = DirCosineY1[i] * VelocityRatio;
    RandomNum = 0;
    MaxBoltzNumsUsed += 1;

    if(MaxBoltzNumsUsed>6){
      for(int j=0;j<5;j+=2){
        R1 = curand_uniform(globalState);
        R2 = curand_uniform(globalState);
        RNMX[j] = sqrt(-1*log(R1))*cos(R2*((*TwoPi)));
        RNMX[j+1] = sqrt(-1*log(R1))*sin(R2*((*TwoPi)));
      }
      MaxBoltzNumsUsed = 1;
    }
    GasVelX[i] = VTMB[0] * RNMX[MaxBoltzNumsUsed - 1];
    MaxBoltzNumsUsed += 1;
    GasVelY[i] = VTMB[0] * RNMX[MaxBoltzNumsUsed - 1];
    MaxBoltzNumsUsed += 1;
    GasVelZ[i] = VTMB[0] * RNMX[MaxBoltzNumsUsed - 1];
    VelocityX[i] = DCosineX2 * (*Sqrt2M) * sqrt(EAfter);
    VelocityY[i] = DCosineY2 * (*Sqrt2M) * sqrt(EAfter);
    VelocityZ[i] = DCosineZ2 * (*Sqrt2M) * sqrt(EAfter);
    COMEnergy[i] = (pow((VelocityX[i] - GasVelX[i]), 2) + pow((VelocityY[i] - GasVelY[i]), 2) + pow(
        (VelocityZ[i] - GasVelZ[i]),
        2)) / (*TwoM);
    iEnergyBins[i] = COMEnergy[i] / (*ElectronEnergyStep);
    iEnergyBins[i] = min(iEnergyBins[i], 3999);
    RandomNum = curand_uniform(globalState);

    TEST = TotalCollisionFrequency[(int)iEnergyBins[i]] / (*MaxCollisionFreq);
    if (RandomNum < TEST){
      TimeSum[i]+=T[i];
      return;
    }
  }
}

__device__ void ProcessCollisions(double *COMEnergy,double * VelocityX,double * VelocityY,double * VelocityZ,double * GasVelX,double * GasVelY,double * GasVelZ,
  double * AP, double * X,double * Y,double * Z, double *DirCosineX1,double *DirCosineY1,double *DirCosineZ1,double * iEnergyBin,double * CF,double * RGAS,double * EnergyLevels,
  double * INDEX, double * ANGCT, double * SCA, double * IPN, double * AngleFromZ, double * TwoPi, double * EBefore, double * Sqrt2M,
  double * TwoM,double *T,double * BP,double * F1,double * ISIZE,double * NumPoints,int i,curandState* globalState )
  {
    int I;
    double VelocityInCOM,DXCOM,DYCOM,DZCOM,T2,A,B,VelocityBefore,RandomNum;
    double S1,S2,EI,EXTRA,RandomNum2,CosTheta,EpsilonOkhr,Theta,Phi,SinPhi,CosPhi;
    double ARG1,D,U,Q,CosZAngle,SinZAngle,ARGZ,CONST12;
    double VXLab,VYLab,VZLab;
    VelocityInCOM  =  ((*Sqrt2M) * sqrt(COMEnergy[i]));

    DXCOM = (VelocityX[i] - GasVelX[i]) / VelocityInCOM;
    DYCOM = (VelocityY[i] - GasVelY[i]) / VelocityInCOM;
    DZCOM = (VelocityZ[i] - GasVelZ[i]) / VelocityInCOM;

    T2 = T[i]*T[i];
    A = AP[i]*T[i];
    B = (*BP) * T2;

    VelocityBefore = (*Sqrt2M) * sqrt(EBefore[i]);

    A = T[i] * VelocityBefore;
    X[i] += DirCosineX1[i] * A;
    Y[i] += DirCosineY1[i] * A;
    Z[i] += DirCosineZ1[i] * A + T2 * (*F1);
    RandomNum = curand_uniform(globalState);

    I = MBSortT(RandomNum,iEnergyBin[i], CF,(*ISIZE),(*NumPoints));
    while(CF[(int)iEnergyBin[i]*290+I]<RandomNum) I+=1;


    S1 = RGAS[I];
    EI = EnergyLevels[I];

    if(IPN[I]>0){
      RandomNum = curand_uniform(globalState);
      EXTRA = RandomNum * (COMEnergy[i]-EI);
      EI = EXTRA + EI;
    }

    if(COMEnergy[i]<EI){
      EI = COMEnergy[i]-0.0001;
    }

    S2 = (S1*S1)/(S1 - 1.0);
    RandomNum = curand_uniform(globalState);

    if(INDEX[I] == 1){
      RandomNum2 = curand_uniform(globalState);
      CosTheta = 1.0-RandomNum*ANGCT[(int)iEnergyBin[i]*290 + I];
      if(RandomNum2>SCA[(int)iEnergyBin[i]*290 + I]){
        CosTheta = -1.0 * CosTheta;
      }
    }else if(INDEX[I]==2){
      EpsilonOkhr = SCA[(int)iEnergyBin[i]*290 + I];
      CosTheta = 1.0 - (2.0 * RandomNum * (1.0 - EpsilonOkhr) / (1.0 + EpsilonOkhr * (1.0 - 2.0 * RandomNum)));
    }else{
      CosTheta = 1.0 - 2.0*RandomNum;
    }

    Theta = acos(CosTheta);
    RandomNum = curand_uniform(globalState);
    Phi = (*TwoPi) * RandomNum;
    SinPhi = sin(Phi);
    CosPhi = cos(Phi);


    ARG1 = max(1.0 - S1*EI/COMEnergy[i],1E-20);

    D = 1.0 - CosTheta * sqrt(ARG1);
    U = (S1 - 1.0)*(S1-1.0)/ARG1;

    EBefore[i] = max(COMEnergy[i] * (1.0 - EI / (S1 * COMEnergy[i]) - 2.0 * D / S2), 1E-20);

    Q = min(sqrt((COMEnergy[i] / EBefore[i]) * ARG1) / S1,1.0);

    AngleFromZ[i] = asin(Q * sin(Theta));
    CosZAngle = cos(AngleFromZ[i]);

    if(CosTheta<0 && CosTheta*CosTheta>U){
      CosZAngle = -1 * CosZAngle;
    }
    SinZAngle = sin(AngleFromZ[i]);
    DZCOM = min(DZCOM,1.0);
    ARGZ = sqrt(DXCOM*DXCOM + DYCOM*DYCOM);
    if (ARGZ ==0){
      DirCosineZ1[i] = CosZAngle;
      DirCosineX1[i] = CosPhi * SinZAngle;
      DirCosineY1[i] = SinPhi * SinZAngle;
    }else{
      DirCosineZ1[i] = DZCOM * CosZAngle + ARGZ * SinZAngle * SinPhi;
      DirCosineY1[i] = DYCOM * CosZAngle + (SinZAngle / ARGZ) * (DXCOM * CosPhi - DYCOM * DZCOM * SinPhi);
      DirCosineX1[i] = DXCOM * CosZAngle - (SinZAngle / ARGZ) * (DYCOM * CosPhi + DXCOM * DZCOM * SinPhi);
    }

    CONST12 = (*Sqrt2M) * sqrt(EBefore[i]);
    VXLab = DirCosineX1[i] * CONST12 + GasVelX[i];
    VYLab = DirCosineY1[i] * CONST12 + GasVelY[i];
    VZLab = DirCosineZ1[i] * CONST12 + GasVelZ[i];

    EBefore[i] = (VXLab * VXLab + VYLab * VYLab + VZLab * VZLab) / (*TwoM);
    VelocityInCOM = ((*Sqrt2M) * sqrt(EBefore[i]));
    DirCosineX1[i] = VXLab / VelocityInCOM;
    DirCosineY1[i] = VYLab / VelocityInCOM;
    DirCosineZ1[i] = VZLab / VelocityInCOM;
}

// Copying constants into device
//Copying arrays to device


__global__ void MonteRun(double * EIN,double * ElectronEnergyStep,double *MaxCollisionFreqTotal,double * BP,double * F1,double * F2,double * Sqrt2M,
double * TwoM,double * TwoPi,double * ISize,double * NumPoints,double * MaxCollisionFreq,double * VTMB,double * X,double * Y,double * Z,double * TimeSum,
double * DirCosineZ1,double * DirCosineY1,double * DirCosineX1,double * EBefore,double * iEnergyBins,double * COMEnergy,double * VelocityX,double * VelocityY,
double * VelocityZ,double * GasVelX,double * GasVelY,double * GasVelZ,double * T,double * AP,double * AngleFromZ,double * CF,double * ANGCT,double * SCA,
double * INDEX,double * IPN,double * RGAS,double * TotalCollisionFrequency,long long  * seeds, double * output
){
  int i = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  curandState state;
  curand_init(seeds[i], i, 0, &state);
  __syncthreads();

  int f = 0;

  for(int iColl=0;iColl<1000000;++iColl){
    GetCollisions(ElectronEnergyStep, MaxCollisionFreqTotal, BP,F1,
      F2,Sqrt2M,TwoM,TwoPi,MaxCollisionFreq, VTMB,TimeSum,
      DirCosineZ1, DirCosineX1, DirCosineY1, EBefore, iEnergyBins,
      COMEnergy, VelocityX, VelocityY,VelocityZ, GasVelX, GasVelY, GasVelZ,
      T, AP, TotalCollisionFrequency,i,&state);
      __syncthreads();

      ProcessCollisions(COMEnergy,VelocityX,VelocityY, VelocityZ, GasVelX,GasVelY, GasVelZ,
    AP, X, Y, Z, DirCosineX1,DirCosineY1,DirCosineZ1,iEnergyBins, CF, RGAS,EIN,
      INDEX,ANGCT, SCA, IPN, AngleFromZ,  TwoPi,  EBefore, Sqrt2M, TwoM,T,BP,F1,ISize,NumPoints,i,&state);
      if(((iColl)%(1000000/100))==0){

        output[0*100000+f*1000+i]=X[i];
        output[1*100000+f*1000+i]=Y[i];
        output[2*100000+f*1000+i]=Z[i];
        output[3*100000+f*1000+i]=TimeSum[i];
          f+=1;
      }
      __syncthreads();
  }
}

__global__ void Test(MonteGpuDevice * M){
  //printf("THIS IS IT %f\n",M->EnergyLevels[0]);
  M->Output[0] = 1000;

}
// function that will be called from the PyBoltz_Gpu class
void MonteGpu::MonteTGpu(){
  printf("HEREEEE %.10f\n", RGAS[10]);
  MonteGpuDevice * DeviceParameters = new MonteGpuDevice();


  MonteGpuDevice * DeviceParametersPointer;

  //DeviceParameters->EnergyLevels = LinearizeAndCopy2D(EnergyLevels,6,290);
  cudaMemcpy(DeviceParameters->EnergyLevels,EnergyLevels,6*290*sizeof(double),cudaMemcpyHostToDevice);

  printf("HERE\n");

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
  printf("HERE\n");

  DeviceParameters->TwoPi = SetupAndCopyDouble(&(twpi),1);
  DeviceParameters->ISIZE = SetupAndCopyDouble(ISIZE,6);
  printf("HERE\n");
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
  printf("HERE\n");

  DeviceParameters->T = SetupArrayOneVal(0,1000);
  DeviceParameters->AP = SetupArrayOneVal(0,1000);
  cudaMemcpy(DeviceParameters->TotalCollisionFrequency,TotalCollisionFrequency,6*4000*sizeof(double),cudaMemcpyHostToDevice);

  DeviceParameters->AngleFromZ = SetupArrayOneVal(AngleFromZ,1000);
  cudaMemcpy(DeviceParameters->CollisionFrequency,CollisionFrequency,6*4000*290*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceParameters->AngleCut,AngleCut,6*4000*290*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceParameters->ScatteringParameter,ScatteringParameter,6*4000*290*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceParameters->INDEX,INDEX,6*290*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceParameters->IPN,IPN,6*290*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceParameters->RGAS,RGAS,6*290*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceParameters->TotalCollisionFrequency,TotalCollisionFrequency,6*4000*sizeof(double),cudaMemcpyHostToDevice);
  DeviceParameters->Output = SetupArrayOneVal(0,400000);
  printf("%.20f\n",sqrt2m*InitialElectronEnergy );

  srand(3);
  //RM48 stuff
  //struct RM48Gen* gen =(struct RM48Gen *)malloc(1000*sizeof(struct RM48Gen));
  long long * Seeds = (long long *)malloc(1000*sizeof(long long));
  for (int i=0;i<1000;i++){
    Seeds[i] = (rand()%100000000);
  }

  int f = 0;
  //printf("%d\n",gen[0].IJKLIN);

  long long * pointer;
  cudaMalloc((void **)&pointer,1000*sizeof(long long));
  cudaMemcpy(pointer,Seeds,1000*sizeof(long long),cudaMemcpyHostToDevice);
  double * TT = (double *)malloc(1000*sizeof(double));

  cudaMemcpy(DeviceParametersPointer,DeviceParameters,sizeof(MonteGpuDevice),cudaMemcpyHostToDevice);

/*  MonteRun<<<25,40>>>(EIN, ElectronEnergyStep,MaxCollisionFreqTotal, BP, F1, F2, Sqrt2M,
   TwoM,TwoPi, ISize, NumPoints, MaxCollisionFreq, VTMB, X, Y, Z, TimeSum,
  DirCosineZ1, DirCosineY1, DirCosineX1, EBefore, iEnergyBins, COMEnergy, VelocityX, VelocityY,
   VelocityZ, GasVelX, GasVelY, GasVelZ, T, AP, AngleFromZ, CF, ANGCT, SCA,
   INDEX, IPN, RGAS, TotalCollisionFrequency, pointer, Output);*/
   Test<<<1,1>>>(DeviceParametersPointer);
   printf("HERE\n");
cudaDeviceSynchronize();
   cudaMemcpy(output,DeviceParameters->Output,400000*sizeof(double),cudaMemcpyDeviceToHost);

   printf("THIS IS IT  %f\n",output[0]);

  //FreeRM48GensCuda<<<int(1000),1>>>(pointer);
  cudaFree(DeviceParameters->Output);
  cudaFree(pointer);
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
  cudaFree(DeviceParametersPointer);
}
