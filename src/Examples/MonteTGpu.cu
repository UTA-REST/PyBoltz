#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
  double RVEC[1001];
  int IVEC = 0;
  int NVEC = 1000;
  int  I97, J97;
  double U[98],C;
  int IJKLIN = 54217137,KALLED;
  double NTOT2N = 0,NTOTIN =0 ,NTOT=-1, NTOT2=0;
};

// RM48 functions
__device__ double dmod(double x, double y) {
    return x - (int)(x/y) * y;
}

__device__ double MOD(double A,double B){
  return dmod(A,B);
}


void SetupRM48Gens(struct RM48Gen* gen,int s,long long * seeds){
    for(int i=0;i<s;++i){
      gen[i].IJKLIN = seeds[i];
    }
}

__global__ extern void GetCollisions(double ElectronEnergyStep, double MaxCollisionFreqTotal,double BP,double  F1,
  double  F2,double Sqrt2M,double TwoM,double TwoPi,double MaxCollisionFreq,double * VTMB,double * TimeSum,
  double * DirCosineZ1,double * DirCosineX1,double * DirCosineY1,double * EBefore,double * iEnergyBins,
  double * COMEnergy,double * VelocityX,double * VelocityY,double * VelocityZ,double * GasVelX,double * GasVelY,double * GasVelZ,
  double * T,double * AP,double * TotalCollisionFrequency,long long * seed){

  // function start
  int i = threadIdx.x+blockDim.x*blockIdx.x;
  int MaxBoltzNumsUsed = 1;
  //R = curand_uniform( &state );
  double RNMX[6]={0,0,0,0,0,0};
  double TDash = 0.0,R1,R2,RandomNum,TEST;
  for(int j=0;j<5;j+=2){
    R1 = 0;
    R2 = 0;
    RNMX[(int)j] = sqrt((double)(-1*log((double)R1)))*cos((double)(R2*TwoPi));
    RNMX[j+1] = sqrt(-1*log(R1))*sin(R2*TwoPi);
  }
  double EAfter = 0.0,VelocityRatio,DCosineZ2,DCosineX2,DCosineY2;
  while(1){
    RandomNum = 0;
    T[(int)i] = -1 * log((double)RandomNum)/MaxCollisionFreqTotal+TDash;
    TDash = T[i];
    AP[i] = DirCosineZ1[i]*F2*sqrt(EBefore[i]);
    EAfter = EBefore[i]+(AP[i]+BP*T[i])*T[i];
    VelocityRatio = sqrt(EBefore[i]/EAfter);
    DCosineZ2 = DirCosineZ1[i] * VelocityRatio + T[i] * F2 / (2 * sqrt(EAfter));
    DCosineX2 = DirCosineX1[i] * VelocityRatio;
    DCosineY2 = DirCosineY1[i] * VelocityRatio;
    RandomNum = 0;
    MaxBoltzNumsUsed += 1;
    if(MaxBoltzNumsUsed>6){
      for(int j=0;j<5;j+=2){
        R1 = 0;
        R2 = 0;
        RNMX[j] = sqrt(-1*log(R1))*cos(R2*TwoPi);
        RNMX[j+1] = sqrt(-1*log(R1))*sin(R2*TwoPi);
      }
      MaxBoltzNumsUsed = 1;
    }
    GasVelX[i] = VTMB[0] * RNMX[MaxBoltzNumsUsed - 1];
    MaxBoltzNumsUsed += 1;
    GasVelY[i] = VTMB[0] * RNMX[MaxBoltzNumsUsed - 1];
    MaxBoltzNumsUsed += 1;
    GasVelZ[i] = VTMB[0] * RNMX[MaxBoltzNumsUsed - 1];
    VelocityX[i] = DCosineX2 * Sqrt2M * sqrt(EAfter);
    VelocityY[i] = DCosineY2 * Sqrt2M * sqrt(EAfter);
    VelocityZ[i] = DCosineZ2 * Sqrt2M * sqrt(EAfter);
    COMEnergy[i] = (pow((VelocityX[i] - GasVelX[i]), 2) + pow((VelocityY[i] - GasVelY[i]), 2) + pow(
        (VelocityZ[i] - GasVelZ[i]),
        2)) / TwoM;
    iEnergyBins[i] = (int)COMEnergy[i] / ElectronEnergyStep;
    iEnergyBins[i] = minn(iEnergyBins[i], 3999);
    RandomNum = 0;
    TEST = TotalCollisionFrequency[(int)iEnergyBins[i]] / MaxCollisionFreq;
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
    Seeds[i] = (i*25348)%100000;
  }

  //printf("%d\n",gen[0].IJKLIN);
  printf("HERE\n");

  long long * pointer;
  cudaMalloc((void **)&pointer,1000*sizeof(long long));
  cudaMemcpy(pointer,Seeds,1000*sizeof(long long),cudaMemcpyHostToDevice);
  double * TT = (double *)malloc(1000*sizeof(double));
  for(int i=0;i<1000;++i){
    GetCollisions<<<int(1000),1>>>(*ElectronEnergyStep, *MaxCollisionFreqTotal, *BP,*F1,
      *F2,*Sqrt2M,*TwoM,*TwoPi,*MaxCollisionFreq, VTMB,TimeSum,
      DirCosineZ1, DirCosineX1, DirCosineY1, EBefore, iEnergyBins,
      COMEnergy, VelocityX, VelocityY,VelocityZ, GasVelX, GasVelY, GasVelZ,
      T, AP, TotalCollisionFrequency, pointer);
      cudaMemcpy(TT,T,1000*sizeof(double),cudaMemcpyDeviceToHost);
      printf("%d %f\n",i,TT[0]);
  }


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
