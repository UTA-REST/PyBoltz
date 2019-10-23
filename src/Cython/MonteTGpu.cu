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

double * LinearizeAndCopy(double** arr,int h,int w){

  double * pointer;

  double * temp = (double *)malloc(h*w*sizeof(double));

  for(int i=0;i<h;++i){

    for(int j = 0;j<w;++j){
      temp[i*w+j] =arr[i][j];
    }
  }

  cudaMalloc((void **)&pointer,h*w*sizeof(double));
  cudaMemcpy(pointer,temp,h*w*sizeof(double),cudaMemcpyHostToDevice);
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


// function that will be called from the PyBoltz_Gpu class
void MonteGpu::MonteTGpu(){
  printf("HEREEEE %f\n", PElectronEnergyStep);

  double * EIN = LinearizeAndCopy(PEnergyLevels,6,290);
  // Copying constants into device
  double * ElectronEnergyStep = SetupAndCopyDouble(&(PElectronEnergyStep),1);
  double * MaxCollisionFreqTotal = SetupAndCopyDouble(&(PMaxCollisionFreqTotal),1);
  double bp = PEField*PEField*PCONST1;
  double * BP = SetupAndCopyDouble(&(bp),1);
  double f1 = PEField*PCONST2;
  double * F1 = SetupAndCopyDouble(&(f1),1);
  double f2 = PEField*PCONST3;
  double * F2 = SetupAndCopyDouble(&(f2),1);
  double sqrt2m = PCONST3*0.01;
  double * Sqrt2M = SetupAndCopyDouble(&(sqrt2m),1);
  double twom = sqrt2m*sqrt2m;
  double * TwoM = SetupAndCopyDouble(&(twom),1);
  double twpi = Ppi*2;
  double * TwoPi = SetupAndCopyDouble(&(twpi),1);
  double * ISize = SetupAndCopyDouble(&(PISIZE[0]),1);
  double * NumPoints = SetupAndCopyDouble(&(PNumMomCrossSectionPoints),1);
  double * MaxCollisionFreq = SetupAndCopyDouble(&(PMaxCollisionFreq),1);

  //Copying arrays to device
  double * VTMB = SetupAndCopyDouble((PVTMB),6);
  double * X = SetupArrayOneVal(0,1000);
  double * Y = SetupArrayOneVal(0,1000);
  double * Z = SetupArrayOneVal(0,1000);
  double * TimeSum = SetupArrayOneVal(0,1000);
  double * DirCosineZ1 = SetupArrayOneVal(cos(PAngleFromZ),1000);
  double * DirCosineX1 = SetupArrayOneVal(sin(PAngleFromZ) * cos(PAngleFromX),1000);
  double * DirCosineY1 = SetupArrayOneVal(sin(PAngleFromZ) * sin(PAngleFromX),1000);
  double * EBefore = SetupArrayOneVal(PInitialElectronEnergy,1000);
  double * iEnergyBins = SetupArrayOneVal(0,1000);
  double * COMEnergy = SetupArrayOneVal(0,1000);
  double * VelocityX = SetupArrayOneVal(0,1000);
  double * VelocityY = SetupArrayOneVal(0,1000);
  double * VelocityZ = SetupArrayOneVal(0,1000);
  double * GasVelX = SetupArrayOneVal(0,1000);
  double * GasVelY = SetupArrayOneVal(0,1000);
  double * GasVelZ = SetupArrayOneVal(0,1000);
  double * T = SetupArrayOneVal(0,1000);
  double * AP = SetupArrayOneVal(0,1000);
  double * AngleFromZ = SetupArrayOneVal(PAngleFromZ,1000);
  double * CF = LinearizeAndCopy((double **)PCollisionFrequency,4000,290);
  double * ANGCT = LinearizeAndCopy((double **)PAngleCut,4000,290);
  double * SCA = LinearizeAndCopy((double **)PScatteringParameter,4000,290);
  double * INDEX = SetupAndCopyDouble((PINDEX),290);
  double * IPN = SetupAndCopyDouble((PIPN),290);
  double * RGAS = LinearizeAndCopy((double **)PRGAS,6,290);
  double * Output = SetupArrayOneVal(0,400000);
  double * TotalCollisionFrequency = SetupAndCopyDouble(PTotalCollisionFrequency,4000);
  printf("%.20f\n",sqrt2m*PInitialElectronEnergy );

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

  MonteRun<<<25,40>>>(EIN, ElectronEnergyStep,MaxCollisionFreqTotal, BP, F1, F2, Sqrt2M,
   TwoM,TwoPi, ISize, NumPoints, MaxCollisionFreq, VTMB, X, Y, Z, TimeSum,
  DirCosineZ1, DirCosineY1, DirCosineX1, EBefore, iEnergyBins, COMEnergy, VelocityX, VelocityY,
   VelocityZ, GasVelX, GasVelY, GasVelZ, T, AP, AngleFromZ, CF, ANGCT, SCA,
   INDEX, IPN, RGAS, TotalCollisionFrequency, pointer, Output);
   cudaMemcpy(output,Output,400000*sizeof(double),cudaMemcpyDeviceToHost);
  //FreeRM48GensCuda<<<int(1000),1>>>(pointer);
  cudaFree(Output);
  cudaFree(pointer);
  cudaFree(ElectronEnergyStep);
  cudaFree(MaxCollisionFreqTotal);
  cudaFree(BP);
  cudaFree(F1);
  cudaFree(F2);
  cudaFree(Sqrt2M);
  cudaFree(TwoM);
  cudaFree(TwoPi);
  cudaFree(ISize);
  cudaFree(NumPoints);
  cudaFree(MaxCollisionFreq);
  cudaFree(VTMB);
  cudaFree(X);
  cudaFree(Y);
  cudaFree(Z);
  cudaFree(TimeSum);
  cudaFree(DirCosineX1);
  cudaFree(DirCosineY1);
  cudaFree(DirCosineZ1);
  cudaFree(EBefore);
  cudaFree(iEnergyBins);
  cudaFree(COMEnergy);
  cudaFree(VelocityZ);
  cudaFree(VelocityY);
  cudaFree(VelocityX);
  cudaFree(T);
  cudaFree(GasVelX);
  cudaFree(GasVelY);
  cudaFree(GasVelZ);
  cudaFree(AP);
  cudaFree(AngleFromZ);
  cudaFree(CF);
  cudaFree(RGAS);
  cudaFree(EIN);
  cudaFree(ANGCT);
  cudaFree(SCA);
  cudaFree(INDEX);
  cudaFree(IPN);
}
