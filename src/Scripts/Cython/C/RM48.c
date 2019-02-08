
#include <stdio.h>
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
 #endif
 #ifndef min
  #define min(a,b) (((a) < (b)) ? (a) : (b))
  #endif

int MOD(int A,int B){
  return A%B;
}
extern double RVEC[1001];
extern double U[98],C;
extern long I97, J97,IVEC = 0,KALLED,NTOTIN,NTOT2N,IJKL,NTOT, NTOT2,I,J,K,L,S,M;
extern long MODCNS = 1000000000;
extern int NVEC = 1000;
extern double CD, CM, TWOM24,TWOM49, ONE, ZERO;
extern double IJKLIN;
extern double IJ,KL,HALF,T,NOW,UNI;
void RM48IN(int IJKLIN,int NTOTIN,int NTOT2N){
  IJKL = IJKLIN;
  NTOT = max(NTOTIN,0);
  NTOT2 = max(NTOT2N,0);
  KALLED = 1;
  return;
}
void RM48UT(){
  printf("RM48 status: %f %f %f\n",IJKL,NTOT,NTOT2 );
}
double DRAND48(double dummy){
  if (IVEC ==0 || IVEC>=NVEC){
    RM48(RVEC,NVEC);
    IVEC = 1;
  }else{
    IVEC+=1;
  }
  return RVEC[IVEC];
}

void RM48(double RVEC[],double LENV){
  NTOT = -1;NTOT2 = 0;IJKL = 0;
  if(NTOT>=0) goto L50;
  IJKL = 54217137;
  NTOT = 0;
  NTOT2 = 0;
  KALLED = 0;

  IJ = IJKL/30082;
  KL = IJKL - 30082*IJ;
  I = MOD(IJ/177, 177) + 2;
  J = MOD(IJ, 177)     + 2;
  K = MOD(KL/169, 178) + 1;
  L = MOD(KL, 169);
  ONE = 1.;
  HALF = 0.5;
  ZERO = 0.;
  for(int II= 1;II<= 97;++II){
  S = 0.;
  T = HALF;
  for(int JJ= 1;JJ<= 48;++JJ){
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
for(int I24= 1;I24<= 24;++I24){
 TWOM24 = HALF*TWOM24;
}
C  =   362436.*TWOM24;
CD =  7654321.*TWOM24;
CM = 16777213.*TWOM24;
I97 = 97;
J97 = 33;

for(int LOOP2 = 1;LOOP2<=NTOT2+1;++LOOP2){
  NOW = MODCNS;
  if (LOOP2 == NTOT2+1)  NOW=NTOT;
  if (NOW > 0)  {
      for(int IDUM = 1;IDUM<= NTOT;++IDUM){
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

  if (KALLED == 1)  return;
  L50:
  for(int IVEC= 1;IVEC<=LENV;++IVEC){
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
