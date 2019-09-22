#include <stdio.h>

#include <stdlib.h>
double dmod(double x, double y) {
    return x - (int)(x/y) * y;
}
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
 #endif
 #ifndef min
  #define min(a,b) (((a) < (b)) ? (a) : (b))
  #endif

double MOD(double A,double B){
  return dmod(A,B);
}
double RVEC[1001];
int IVEC = 0;
int NVEC = 1000;
int  I97, J97;
double U[98],C;  int IJKLIN = 54217137,KALLED;


double NTOT2N = 0,NTOTIN =0 ,NTOT=-1, NTOT2=0;

extern void RM48(double LENV){
  long MODCNS = 1000000000;
  double T,S,HALF,UNI;
  long long I,J,K,L,M,NOW,IJ,KL;
  static double CD, CM, TWOM24,TWOM49 ,ONE, ZERO;
  static long long IJKL=0;
  int II,JJ,I24,LOOP2,IDUM;


  if(NTOT>=0) goto L50;
  IJKL = IJKLIN;
  NTOT = NTOTIN;
  NTOT2 = NTOT2N;

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
 U[II] = S;
}
TWOM49 = T;
TWOM24 = ONE;
for(I24= 1;I24<= 24;++I24){
 TWOM24 = HALF*TWOM24;
}
C  =   362436.*TWOM24;
CD =  7654321.*TWOM24;
CM = 16777213.*TWOM24;
I97 = 97;
J97 = 33;

for(LOOP2 = 1;LOOP2<=NTOT2+1;++LOOP2){
  NOW = MODCNS;
  if (LOOP2 == NTOT2+1)  NOW=NTOT;
  if (NOW > 0)  {
      for(IDUM = 1;IDUM<= NTOT;++IDUM){
      UNI = U[I97]-U[J97];
      if (UNI < ZERO)  UNI=UNI+ONE;
      U[I97] = UNI;
      I97 = I97-1;
      if (I97== 0)  I97=97;
      J97 = J97-1;
      if (J97 == 0)  J97=97;
      C = C - CD;
      if (C < ZERO)  C=C+CM;
      }
  }
}

  if (KALLED == 1) {
    KALLED = 0;
  return;

  }
  L50:
  for( IVEC= 1;IVEC<=LENV;++IVEC){
  UNI = U[I97]-U[J97];
  if (UNI < ZERO)  UNI=UNI+ONE;
  U[I97] = UNI;
  I97 = I97-1;
  if (I97 == 0)  I97=97;
  J97 = J97-1;
  if (J97== 0)  J97=97;
  C = C - CD;
  if (C < ZERO)  C=C+CM;
  UNI = UNI-C;
  if (UNI < ZERO) UNI=UNI+ONE;
  RVEC[IVEC] = UNI;
//             Replace exact zeros by 2**-49
     if (UNI == ZERO){
        RVEC[IVEC] = TWOM49;
     }
  }
  NTOT = NTOT + LENV;
     if (NTOT >= MODCNS) {
     NTOT2 = NTOT2 + 1;
     NTOT = NTOT - MODCNS;
   }

   return;
}
extern double DRAND48(double dummy){
  if (IVEC ==0 || IVEC>=NVEC){
    RM48(NVEC);
    IVEC = 1;
  }else{
    IVEC+=1;
  }
  return RVEC[IVEC];
}

extern void RM48IN(int IJKLIN1, int NTOTIN1, int NTOT2N1){
    IJKLIN = IJKLIN1;
    NTOTIN = NTOTIN1;
    NTOT2N = NTOT2N1;
    KALLED = 1;
    NTOT = -1;
    RM48(NVEC);
}
int main(){
return 0;
}
