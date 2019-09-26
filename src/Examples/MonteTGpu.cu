#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
 #endif
 #ifndef min
  #define min(a,b) (((a) < (b)) ? (a) : (b))
  #endif

#define maxx(x, y) (((x) > (y)) ? (x) : (y))
#define minn(x, y) (((x) < (y)) ? (x) : (y))
int main()
{
    return 0;
}
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
struct RM48Gen {
  double RVEC[1001],U[98];
  int IVEC = 0;
  int NVEC = 1000;
  int  I97, J97;
  double C;
  int IJKLIN = 54217137,KALLED;
  double NTOT2N = 0,NTOTIN =0 ,NTOT=-1, NTOT2=0;
};

// RM48 functions
__device__ double dmod(double x, double y) {
    return x - (int)(x/y) * y;
}



void SetupRM48Gens(struct RM48Gen* gen,int s,long long * seeds){
    for(int i=0;i<s;++i){
      gen[i].IJKLIN = seeds[i];
      gen[i].NTOTIN = 0;
      gen[i].NTOT2N = 0;
      gen[i].KALLED = 1;
      gen[i].NTOT = -1;
      gen[i].NVEC = 1000;
      gen[i].IVEC = 0;
    }
}


__device__ double MOD(double A,double B){
  return dmod(A,B);
}


__device__ void RM48(struct RM48Gen *RM48gen,double LENV){
  long MODCNS = 1000000000;
  double T,S,HALF,UNI;
  long long NTOT2N,I,J,K,L,M,NOW,IJ,KL;
  static double CD, CM, TWOM24,TWOM49 ,ONE, ZERO;
  static long long IJKL=0;
  int II,JJ,I24,LOOP2,IDUM;


  if(RM48gen->NTOT>=0) goto L50;
  IJKL = RM48gen->IJKLIN;
  RM48gen->NTOT = RM48gen->NTOTIN;
  RM48gen->NTOT2 = RM48gen->NTOT2N;

  IJ = IJKL/30082;
  KL = IJKL - 30082*IJ;
  I = MOD(IJ/177, 177) + 2;
  J = MOD(IJ, 177)     + 2;
  K = MOD(KL/169, 178) + 1;
  L = MOD(KL, 169);
  ONE = 1.;
  HALF = 0.5;
  ZERO = 0.;
  for( II= 1;II<= 97;++II){
  S = 0.;
  T = HALF;
  for(JJ= 1;JJ<= 48;++JJ){
    M = MOD(MOD(I*J,179)*K, 179);
    I = J;
    J = K;
    K = M;
    L = MOD(53*L+1, 169);
     if (MOD(L*M,64) >= 32)  S = S+T;
     T = HALF*T;
   }
 RM48gen->U[II] = S;
}
TWOM49 = T;
TWOM24 = ONE;
for(I24= 1;I24<= 24;++I24){
 TWOM24 = HALF*TWOM24;
}
RM48gen->C  =   362436.*TWOM24;
CD =  7654321.*TWOM24;
CM = 16777213.*TWOM24;
RM48gen->I97 = 97;
RM48gen->J97 = 33;

for(LOOP2 = 1;LOOP2<=RM48gen->NTOT2+1;++LOOP2){
  NOW = MODCNS;
  if (LOOP2 == RM48gen->NTOT2+1)  NOW=RM48gen->NTOT;
  if (NOW > 0)  {
      for(IDUM = 1;IDUM<= RM48gen->NTOT;++IDUM){
      UNI = RM48gen->U[RM48gen->I97]-RM48gen->U[RM48gen->J97];
      if (UNI < ZERO)  UNI=UNI+ONE;
      RM48gen->U[RM48gen->I97] = UNI;
      RM48gen->I97 = RM48gen->I97-1;
      if (RM48gen->I97== 0)  RM48gen->I97=97;
      RM48gen->J97 = RM48gen->J97-1;
      if (RM48gen->J97 == 0)  RM48gen->J97=97;
     RM48gen->C =RM48gen->C - CD;
      if (RM48gen->C < ZERO) RM48gen->C=RM48gen->C+CM;
      }
  }
}

  if (RM48gen->KALLED == 1) {
    RM48gen->KALLED = 0;
  return;

  }
  L50:
  for( RM48gen->IVEC= 1;RM48gen->IVEC<=LENV;RM48gen->IVEC+=1){
  UNI = RM48gen->U[RM48gen->I97]-RM48gen->U[RM48gen->J97];
  if (UNI < ZERO)  UNI=UNI+ONE;
  RM48gen->U[RM48gen->I97] = UNI;
  RM48gen->I97 = RM48gen->I97-1;
  if (RM48gen->I97 == 0)  RM48gen->I97=97;
  RM48gen->J97 = RM48gen->J97-1;
  if (RM48gen->J97== 0)  RM48gen->J97=97;
 RM48gen->C =RM48gen->C - CD;
  if (RM48gen->C < ZERO) RM48gen->C=RM48gen->C+CM;
  UNI = UNI-RM48gen->C;
  if (UNI < ZERO) UNI=UNI+ONE;
  RM48gen->RVEC[RM48gen->IVEC] = UNI;
//             Replace exact zeros by 2**-49
     if (UNI == ZERO){
        RM48gen->RVEC[RM48gen->IVEC] = TWOM49;
     }
  }
  RM48gen->NTOT = RM48gen->NTOT + LENV;
     if (RM48gen->NTOT >= MODCNS) {
     RM48gen->NTOT2 = RM48gen->NTOT2 + 1;
     RM48gen->NTOT = RM48gen->NTOT - MODCNS;
   }

   return;
}
__device__ double DRAND48(struct RM48Gen *RM48gen,double dummy){
  if (RM48gen->IVEC ==0 || RM48gen->IVEC>=RM48gen->NVEC){
    RM48(RM48gen,RM48gen->NVEC);
    RM48gen->IVEC = 1;
  }else{
    RM48gen->IVEC+=1;
  }
  return RM48gen->RVEC[RM48gen->IVEC];
}


__global__ void SetupRM48GensCuda(struct RM48Gen * gen){
  int i = threadIdx.x+blockDim.x*blockIdx.x;
  /*gen[i].RVEC =(double*) malloc(1001*sizeof(double));
  gen[i].U =(double*) malloc(98*sizeof(double));
*/
  RM48(&(gen[i]),gen[i].NVEC);
}

/*__global__ void FreeRM48GensCuda(struct RM48Gen * gen){
  int i = threadIdx.x+blockDim.x*blockIdx.x;
  free(gen[i].RVEC);
  free(gen[i].U);

}*/

__global__ extern void GetCollisions(double *ElectronEnergyStep, double* MaxCollisionFreqTotal,double* BP,double*  F1,
  double*  F2,double* Sqrt2M,double* TwoM,double* TwoPi,double* MaxCollisionFreq,double * VTMB,double * TimeSum,
  double * DirCosineZ1,double * DirCosineX1,double * DirCosineY1,double * EBefore,double * iEnergyBins,
  double * COMEnergy,double * VelocityX,double * VelocityY,double * VelocityZ,double * GasVelX,double * GasVelY,double * GasVelZ,
  double * T,double * AP,double * TotalCollisionFrequency,struct RM48Gen * gen){

  // function start
  int i = threadIdx.x+blockDim.x*blockIdx.x;
  int MaxBoltzNumsUsed = 1;
  DRAND48(&(gen[i]),0.5);
  //R = curand_uniform( &state );
  double RNMX[6]={0,0,0,0,0,0};
  double TDash = 0.0,R1,R2,RandomNum,TEST;

  for(int j=0;j<5;j+=2){
    R1 = DRAND48(&(gen[i]),0.5);
    R2 = DRAND48(&(gen[i]),0.5);

    RNMX[j] = sqrt((-1*log((double)R1)))*cos((double)(R2*(*TwoPi)));
    RNMX[j+1] = sqrt(-1*log((double)R1))*sin((double)R2*(*TwoPi));

  }

  double EAfter = 0.0,VelocityRatio,DCosineZ2,DCosineX2,DCosineY2;

  while(1){
    RandomNum = DRAND48(&(gen[i]),0.5);
    T[i] = -1 * log(RandomNum)/(*MaxCollisionFreqTotal)+TDash;
    TDash = T[i];
    AP[i] = DirCosineZ1[i]*(*F2)*sqrt(EBefore[i]);
    EAfter = EBefore[i]+(AP[i]+(*BP)*T[i])*T[i];
    VelocityRatio = sqrt(EBefore[i]/EAfter);
    DCosineZ2 = DirCosineZ1[i] * VelocityRatio + T[i] * (*F2) / (2 * sqrt(EAfter));
    DCosineX2 = DirCosineX1[i] * VelocityRatio;
    DCosineY2 = DirCosineY1[i] * VelocityRatio;
    RandomNum = 0;
    MaxBoltzNumsUsed += 1;

    if(MaxBoltzNumsUsed>6){
      for(int j=0;j<5;j+=2){
        R1 = DRAND48(&(gen[i]),0.5);
        R2 = DRAND48(&(gen[i]),0.5);
        RNMX[j] = sqrt((-1*log(R1)))*cos((R2*(*TwoPi)));
        RNMX[j+1] = sqrt(-1*log(R1))*sin(R2*(*TwoPi));
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
    iEnergyBins[i] = minn(iEnergyBins[i], 3999);
    RandomNum = DRAND48(&(gen[i]),0.5);

    TEST = TotalCollisionFrequency[(int)iEnergyBins[i]] / (*MaxCollisionFreq);
    if (RandomNum < TEST){
      TimeSum[i] +=T[i];
      return;
    }
  }
}



// function that will be called from the PyBoltz_Gpu class
extern "C" void MonteTGpu(double PElectronEnergyStep,double PMaxCollisionFreqTotal,double PEField, double PCONST1,double PCONST2,double PCONST3
, double Ppi,double PISIZE,double PNumMomCrossSectionPoints,double PMaxCollisionFreq, double * PVTMB, double PAngleFromZ, double PAngleFromX,
double PInitialElectronEnergy, double** PCollisionFrequency, double *PTotalCollisionFrequency, double ** PRGAS, double ** PEnergyLevels,
double ** PAngleCut,double ** PScatteringParameter, double * PINDEX, double * PIPN
){
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
  double * ISize = SetupAndCopyDouble(&(PISIZE),1);
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
  double * TotalCollisionFrequency = SetupAndCopyDouble(PTotalCollisionFrequency,4000);



  //RM48 stuff
  //struct RM48Gen* gen =(struct RM48Gen *)malloc(1000*sizeof(struct RM48Gen));
  long long * Seeds = (long long *)malloc(1000*sizeof(long long));
  for (int i=0;i<1000;i++){
    Seeds[i] = (i*54217137)%100000000;
  }

  //printf("%d\n",gen[0].IJKLIN);
  struct  RM48Gen* gens = (struct  RM48Gen*)malloc(1000*sizeof(struct  RM48Gen));
  SetupRM48Gens(gens,1000,Seeds);

  struct  RM48Gen* pointer;
  cudaMalloc((void **)&pointer,1000*sizeof(struct RM48Gen));
  cudaMemcpy(pointer,gens,1000*sizeof(struct  RM48Gen),cudaMemcpyHostToDevice);
  double * TT = (double *)malloc(1000*sizeof(double));
  SetupRM48GensCuda<<<int(1000),1>>>(pointer);
  for(int i=0;i<1000;++i){
/*
    double ElectronEnergyStep, double MaxCollisionFreqTotal,double BP,double  F1,
      double  F2,double Sqrt2M,double TwoM,double TwoPi,double MaxCollisionFreq,double * VTMB,double * TimeSum,
      double * DirCosineZ1,double * DirCosineX1,double * DirCosineY1,double * EBefore,double * iEnergyBins,
      double * COMEnergy,double * VelocityX,double * VelocityY,double * VelocityZ,double * GasVelX,double * GasVelY,double * GasVelZ,
      double * T,double * AP,double * TotalCollisionFrequency,struct RM48Gen * gen
*/
    GetCollisions<<<int(1000),1>>>(ElectronEnergyStep, MaxCollisionFreqTotal, BP,F1,
      F2,Sqrt2M,TwoM,TwoPi,MaxCollisionFreq, VTMB,TimeSum,
      DirCosineZ1, DirCosineX1, DirCosineY1, EBefore, iEnergyBins,
      COMEnergy, VelocityX, VelocityY,VelocityZ, GasVelX, GasVelY, GasVelZ,
      T, AP, TotalCollisionFrequency, pointer);
      printf("HERE\n");
      cudaMemcpy(TT,T,1000*sizeof(double),cudaMemcpyDeviceToHost);
      printf("%d %f\n",i,TT[0]);
  }
  //FreeRM48GensCuda<<<int(1000),1>>>(pointer);
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
