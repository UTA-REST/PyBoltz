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
    iEnergyBins[i] = min(iEnergyBins[i], 3999);
    RandomNum = DRAND48(&(gen[i]),0.5);

    TEST = TotalCollisionFrequency[(int)iEnergyBins[i]] / (*MaxCollisionFreq);
    if (RandomNum < TEST){
      TimeSum[i] +=T[i];
      return;
    }
  }
}

__global__ void ProcessCollisions(double *COMEnergy,double * VelocityX,double * VelocityY,double * VelocityZ,double * GasVelX,double * GasVelY,double * GasVelZ,
  double * AP, double * X,double * Y,double * Z, double *DirCosineX1,double *DirCosineY1,double *DirCosineZ1,double * iEnergyBin,double * CF,double * RGAS,double * EnergyLevels,
  double * INDEX, double * ANGCT, double * SCA, double * IPN, double * AngleFromZ, double * TwoPi, double * EBefore, double * Sqrt2M,
  double * TwoM,double *T,double * BP,double * F1,double * ISIZE,double * NumPoints,struct RM48Gen * gen)
  {
    int i = threadIdx.x+blockDim.x*blockIdx.x;

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
    RandomNum = DRAND48(&(gen[i]),0.5);

    I = MBSortT(RandomNum,iEnergyBin[i], CF,(*ISIZE),(*NumPoints));
    while(CF[(int)iEnergyBin[i]*290+I]<RandomNum) I+=1;


    S1 = RGAS[I];
    EI = EnergyLevels[I];

    if(IPN[I]>0){
      RandomNum = DRAND48(&(gen[i]),0.5);
      EXTRA = RandomNum * (COMEnergy[i]-EI);
      EI = EXTRA + EI;
    }

    if(COMEnergy[i]<EI){
      EI = COMEnergy[i]-0.0001;
    }

    S2 = (S1*S1)/(S1 - 1.0);
    RandomNum = DRAND48(&(gen[i]),0.5);

    if(INDEX[I] == 1){
      RandomNum2 = DRAND48(&(gen[i]),0.5);
      CosTheta = 1.0 *RandomNum*ANGCT[(int)iEnergyBin[i]*290 + I];
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
    RandomNum = DRAND48(&(gen[i]),0.5);
    Phi = (*TwoPi) * RandomNum;
    SinPhi = sin(Phi);
    CosPhi = cos(Phi);


    ARG1 = max(1.0 - S1*EI/COMEnergy[i],1e-20);

    D = 1.0 - CosTheta * sqrt(ARG1);
    U = (S1 - 1)*(S1-1)/ARG1;

    EBefore[i] = max(COMEnergy[i] * (1.0 - EI / (S1 * COMEnergy[i]) - 2.0 * D / S2), 1e-20);

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

// function that will be called from the PyBoltz_Gpu class
extern "C" double* MonteTGpu(double PElectronEnergyStep,double PMaxCollisionFreqTotal,double PEField, double PCONST1,double PCONST2,double PCONST3
, double Ppi,double PISIZE,double PNumMomCrossSectionPoints,double PMaxCollisionFreq, double * PVTMB, double PAngleFromZ, double PAngleFromX,
double PInitialElectronEnergy, double** PCollisionFrequency, double *PTotalCollisionFrequency, double ** PRGAS, double ** PEnergyLevels,
double ** PAngleCut,double ** PScatteringParameter, double * PINDEX, double * PIPN,double * output
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
  printf("%f\n",PCollisionFrequency[0][0] );


  //RM48 stuff
  //struct RM48Gen* gen =(struct RM48Gen *)malloc(1000*sizeof(struct RM48Gen));
  long long * Seeds = (long long *)malloc(1000*sizeof(long long));
  for (int i=0;i<1000;i++){
    Seeds[i] = (i*54217137)%100000000;
  }
  int f = 0;
  //printf("%d\n",gen[0].IJKLIN);
  struct  RM48Gen* gens = (struct  RM48Gen*)malloc(1000*sizeof(struct  RM48Gen));
  SetupRM48Gens(gens,1000,Seeds);

  struct  RM48Gen* pointer;
  cudaMalloc((void **)&pointer,1000*sizeof(struct RM48Gen));
  cudaMemcpy(pointer,gens,1000*sizeof(struct  RM48Gen),cudaMemcpyHostToDevice);
  double * TT = (double *)malloc(1000*sizeof(double));
  SetupRM48GensCuda<<<int(1000),1>>>(pointer);
  char * str;
  for(int i=0;i<10000;++i){
    GetCollisions<<<int(1000),1>>>(ElectronEnergyStep, MaxCollisionFreqTotal, BP,F1,
      F2,Sqrt2M,TwoM,TwoPi,MaxCollisionFreq, VTMB,TimeSum,
      DirCosineZ1, DirCosineX1, DirCosineY1, EBefore, iEnergyBins,
      COMEnergy, VelocityX, VelocityY,VelocityZ, GasVelX, GasVelY, GasVelZ,
      T, AP, TotalCollisionFrequency, pointer);
      ProcessCollisions<<<int(1000),1>>>(COMEnergy,VelocityX,VelocityY, VelocityZ, GasVelX,GasVelY, GasVelZ,
    AP, X, Y, Z, DirCosineX1,DirCosineY1,DirCosineZ1,iEnergyBins, CF, RGAS,EIN,
      INDEX,ANGCT, SCA, IPN, AngleFromZ,  TwoPi,  EBefore, Sqrt2M, TwoM,T,BP,F1,ISize,NumPoints,pointer);

      if(((i)%(10000/100))==0){
        cudaMemcpy(&output[0*100000+f*1000],X,1000*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(&output[1*100000+f*1000],Y,1000*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(&output[2*100000+f*1000],Z,1000*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(&output[3*100000+f*1000],TimeSum,1000*sizeof(double),cudaMemcpyDeviceToHost);
        f+=1;
      }
      if(i!=0&& double(int(log2(i)))==log2(i)){
        printf("%d analyzed collisions\n", i );
      }
  }
  printf("HERE\n");

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
  return output;
}
