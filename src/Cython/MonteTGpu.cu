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


__device__ extern void GetCollisions(MonteGpuDevice * DP,int i,curandState* globalState){

  // function start
  int MaxBoltzNumsUsed = 0;
  double RNMX[6]={0,0,0,0,0,0};
  double TDash = 0.0,R1,R2,RandomNum,TEST;

  for(int j=0;j<5;j+=2){
    R1 = curand_uniform(globalState);
    R2 = curand_uniform(globalState);
    RNMX[j] = sqrt(-1*log(R1))*cos(R2*((*DP->TwoPi)));
    RNMX[j+1] = sqrt(-1*log(R1))*sin(R2*((*DP->TwoPi)));
  }

  double EAfter = 0.0,VelocityRatio,DCosineZ2,DCosineX2,DCosineY2;

  while(1){
    RandomNum = curand_uniform(globalState);
    T[i] = -1 * log(RandomNum)/(*DP->MaxCollisionFreqTotal)+TDash;
    TDash = DP->T[i];
    AP[i] = DP->DirCosineZ1[i]*(*DP->F2)*sqrt(DP->EBefore[i]);
    EAfter = DP->EBefore[i]+(DP->AP[i]+(*DP->BP)*DP->T[i])*DP->T[i];
    VelocityRatio = sqrt(DP->EBefore[i]/EAfter);
    DCosineZ2 = DP->DirCosineZ1[i] * VelocityRatio + DP->T[i] * (*DP->F2) / (2.0 * sqrt(EAfter));
    DCosineX2 = DP->DirCosineX1[i] * VelocityRatio;
    DCosineY2 = DP->DirCosineY1[i] * VelocityRatio;
    RandomNum = 0;
    MaxBoltzNumsUsed += 1;

    if(MaxBoltzNumsUsed>6){
      for(int j=0;j<5;j+=2){

        R1 = curand_uniform(globalState);
        R2 = curand_uniform(globalState);
        RNMX[j] = sqrt(-1*log(R1))*cos(R2*((*DP->TwoPi)));
        RNMX[j+1] = sqrt(-1*log(R1))*sin(R2*((*DP->TwoPi)));
      }
      MaxBoltzNumsUsed = 1;
    }
    DP->GasVelX[i] = DP->VTMB[0] * RNMX[MaxBoltzNumsUsed - 1];
    MaxBoltzNumsUsed += 1;
    DP->GasVelY[i] = DP->VTMB[0] * RNMX[MaxBoltzNumsUsed - 1];
    MaxBoltzNumsUsed += 1;
    DP->GasVelZ[i] = DP->VTMB[0] * RNMX[MaxBoltzNumsUsed - 1];
    DP->VelocityX[i] = DCosineX2 * (*DP->Sqrt2M) * sqrt(EAfter);
    DP->VelocityY[i] = DCosineY2 * (*DP->Sqrt2M) * sqrt(EAfter);
    DP->VelocityZ[i] = DCosineZ2 * (*DP->Sqrt2M) * sqrt(EAfter);
    DP->COMEnergy[i] = (pow((DP->VelocityX[i] - DP->GasVelX[i]), 2) + pow((DP->VelocityY[i] - DP->GasVelY[i]), 2) + pow(
        (DP->VelocityZ[i] - DP->GasVelZ[i]),
        2)) / (*DP->TwoM);
    DP->iEnergyBins[i] = DP->COMEnergy[i] / (*DP->ElectronEnergyStep);
    DP->iEnergyBins[i] = min(DP->iEnergyBins[i], 3999);
    RandomNum = curand_uniform(globalState);

    TEST = DP->TotalCollisionFrequency[(int)DP->iEnergyBins[i]] / (*DP->MaxCollisionFreq);
    if (RandomNum < TEST){
      DP->TimeSum[i]+=DP->T[i];
      return;
    }
  }
}

__device__ void ProcessCollisions(MonteGpuDevice * DP,int i,curandState* globalState )
  {
    int I;
    double VelocityInCOM,DXCOM,DYCOM,DZCOM,T2,A,B,VelocityBefore,RandomNum;
    double S1,S2,EI,EXTRA,RandomNum2,CosTheta,EpsilonOkhr,Theta,Phi,SinPhi,CosPhi;
    double ARG1,D,U,Q,CosZAngle,SinZAngle,ARGZ,CONST12;
    double VXLab,VYLab,VZLab;
    VelocityInCOM  =  ((*DP->Sqrt2M) * sqrt(DP->COMEnergy[i]));

    DXCOM = (DP->VelocityX[i] - DP->GasVelX[i]) / VelocityInCOM;
    DYCOM = (DP->VelocityY[i] - DP->GasVelY[i]) / VelocityInCOM;
    DZCOM = (DP->VelocityZ[i] - DP->GasVelZ[i]) / VelocityInCOM;

    T2 = DP->T[i]*DP->T[i];
    A = DP->AP[i]*DP->T[i];
    B = (*DP->BP) * T2;

    VelocityBefore = (*DP->Sqrt2M) * sqrt(DP->EBefore[i]);

    A = DP->T[i] * VelocityBefore;
    DP->X[i] += DP->DirCosineX1[i] * A;
    DP->Y[i] += DP->DirCosineY1[i] * A;
    DP->Z[i] += DP->DirCosineZ1[i] * A + T2 * (*DP->F1);
    RandomNum = curand_uniform(globalState);

    I = MBSortT(RandomNum,DP->iEnergyBin[i], DP->CollisionFrequency,(* DP->ISIZE),(*DP->NumMomCrossSectionPoints));
    while(DP->CollisionFrequency[(int)DP->iEnergyBin[i]*290+I]<RandomNum) I+=1;


    S1 = DP->RGAS[I];
    EI = DP->EnergyLevels[I];

    if(DP->IPN[I]>0){
      RandomNum = curand_uniform(globalState);
      EXTRA = RandomNum * (DP->COMEnergy[i]-EI);
      EI = EXTRA + EI;
    }

    if(DP->COMEnergy[i]<EI){
      EI = DP->COMEnergy[i]-0.0001;
    }

    S2 = (S1*S1)/(S1 - 1.0);
    RandomNum = curand_uniform(globalState);

    if(INDEX[I] == 1){
      RandomNum2 = curand_uniform(globalState);
      CosTheta = 1.0-RandomNum*DP->AngleCut[(int)iEnergyBin[i]*290 + I];
      if(RandomNum2>DP->ScatteringParameter[(int)DP->iEnergyBin[i]*290 + I]){
        CosTheta = -1.0 * CosTheta;
      }
    }else if(INDEX[I]==2){
      EpsilonOkhr = DP->ScatteringParameter[(int)DP->iEnergyBin[i]*290 + I];
      CosTheta = 1.0 - (2.0 * RandomNum * (1.0 - EpsilonOkhr) / (1.0 + EpsilonOkhr * (1.0 - 2.0 * RandomNum)));
    }else{
      CosTheta = 1.0 - 2.0*RandomNum;
    }

    Theta = acos(CosTheta);
    RandomNum = curand_uniform(globalState);
    Phi = (*DP->TwoPi) * RandomNum;
    SinPhi = sin(Phi);
    CosPhi = cos(Phi);


    ARG1 = max(1.0 - S1*EI/DP->COMEnergy[i],1E-20);

    D = 1.0 - CosTheta * sqrt(ARG1);
    U = (S1 - 1.0)*(S1-1.0)/ARG1;

    DP->EBefore[i] = max(DP->COMEnergy[i] * (1.0 - EI / (S1 * DP->COMEnergy[i]) - 2.0 * D / S2), 1E-20);

    Q = min(sqrt((DP->COMEnergy[i] / DP->EBefore[i]) * ARG1) / S1,1.0);

    DP->AngleFromZ[i] = asin(Q * sin(Theta));
    CosZAngle = cos(DP->AngleFromZ[i]);

    if(CosTheta<0 && CosTheta*CosTheta>U){
      CosZAngle = -1 * CosZAngle;
    }
    SinZAngle = sin(DP->AngleFromZ[i]);
    DZCOM = min(DZCOM,1.0);
    ARGZ = sqrt(DXCOM*DXCOM + DYCOM*DYCOM);
    if (ARGZ ==0){
      DP->DirCosineZ1[i] = CosZAngle;
      DP->DirCosineX1[i] = CosPhi * SinZAngle;
      DP->DirCosineY1[i] = SinPhi * SinZAngle;
    }else{
      DP->DirCosineZ1[i] = DZCOM * CosZAngle + ARGZ * SinZAngle * SinPhi;
      DP->DirCosineY1[i] = DYCOM * CosZAngle + (SinZAngle / ARGZ) * (DXCOM * CosPhi - DYCOM * DZCOM * SinPhi);
      DP->DirCosineX1[i] = DXCOM * CosZAngle - (SinZAngle / ARGZ) * (DYCOM * CosPhi + DXCOM * DZCOM * SinPhi);
    }

    CONST12 = (*DP->Sqrt2M) * sqrt(DP->EBefore[i]);
    VXLab = DP->DirCosineX1[i] * CONST12 + DP->GasVelX[i];
    VYLab = DP->DirCosineY1[i] * CONST12 + DP->GasVelY[i];
    VZLab = DP->DirCosineZ1[i] * CONST12 + DP->GasVelZ[i];

    DP->EBefore[i] = (VXLab * VXLab + VYLab * VYLab + VZLab * VZLab) / (*TwoM);
    VelocityInCOM = ((*DP->Sqrt2M) * sqrt(DP->EBefore[i]));
    DP->DirCosineX1[i] = VXLab / VelocityInCOM;
    DP->DirCosineY1[i] = VYLab / VelocityInCOM;
    DP->DirCosineZ1[i] = VZLab / VelocityInCOM;
}

// Copying constants into deviceDeviceParametersPointer
//Copying arrays to devicePointe


__global__ void MonteTRun(MonteGpuDevice * DP){
  int i = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  curandState state;
  curand_init(DP->SeedsGpu[i], i, 0, &state);
  __syncthreads();

  int f = 0;
  for(int iColl=0;iColl<DP->NumColls;++iColl){
    GetCollisions(DP,i,&state);
      __syncthreads();

      ProcessCollisions(DP,i,&state);
      if(((iColl)%(DP->NumColls/100))==0){
        DP->output[0*100000+f*1000+i]=DP->X[i];
        DP->output[1*100000+f*1000+i]=DP->Y[i];
        DP->output[2*100000+f*1000+i]=DP->Z[i];
        DP->output[3*100000+f*1000+i]=DP->TimeSum[i];
        f+=1;
      }
      __syncthreads();
  }
}

__global__ void Test(MonteGpuDevice * M,double * RGAS){

  M->Output[0] = 100;
  M->RGAS[10] = 100;
  __syncthreads();

}
// function that will be called from the PyBoltz_Gpu classoutput
void MonteGpu::MonteTGpu(){
  MonteGpuDevice * DeviceParametersPointer;

  cudaMalloc((void **)&DeviceParametersPointer,sizeof(MonteGpuDevice));
  cudaMemcpy(DeviceParametersPointer,DeviceParameters,sizeof(MonteGpuDevice),cudaMemcpyHostToDevice);

  MonteTRun<<<threads,blocks>>>(DeviceParametersPointer);
  //Test<<<threads,blocks>>>(DeviceParametersPointer,DeviceParameters->RGAS);
  printf("HERE\n");
  cudaDeviceSynchronize();
  cudaMemcpy(DeviceParameters,DeviceParametersPointer,sizeof(MonteGpuDevice),cudaMemcpyDeviceToHost);
  printf("%p\n", DeviceParameters->RGAS);
  cudaMemcpy(output,DeviceParameters->Output,400000*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(RGAS,DeviceParameters->RGAS,6*290*sizeof(double),cudaMemcpyDeviceToHost);
  //FreeRM48GensCuda<<<int(1000),1>>>(pointer);
  cudaFree(DeviceParametersPointer);
}
