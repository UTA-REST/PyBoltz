from libc.math cimport sin, cos, acos, asin, log, sqrt, exp, pow
cimport libc.math
import numpy as np
cimport numpy as np
import sys
from Gas cimport Gas
from cython.parallel import prange
cimport GasUtil

sys.path.append('../hdf5_python')
import cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.fast_getattr(True)
cdef void Gas9(Gas*object):
    """
    This function is used to calculate the needed momentum cross sections for Ethane gas.
    """
    gd = np.load('gases.npy').item()
    cdef double XEN[164],YMT[164],YEL[164],YEPS[164],XATT1[11],YATT1[11],XATT2[9],YATT2[9],XVIB1[29],YVIB1[29],XVIB2[28]
    cdef double YVIB2[28],XVIB3[28],YVIB3[28],XVIB4[46],YVIB4[46],XVIB5[16],YVIB5[16],XTR1[12],YTR1[12],XTR2[11],YTR2[11]
    cdef double XTR3[11],YTR3[11],XNUL1[25],YNUL1[25],XNUL2[13],YNUL2[13],XNUL3[14],YNUL3[14]
    cdef double XION1[31],YION1[31],XION2[31],YION2[31],XION3[31],YION3[31],XION4[30],YION4[30],XION5[29],YION5[29],XION6[29]
    cdef double YION6[29],XION7[26],YION7[26],XION8[26],YION8[26],XION9[25],YION9[25],XION10[24],YION10[24],XION11[24]
    cdef double YION12[24],XION13[23],YION13[23],XION14[21],YION14[21],YION11[24],XION12[24]
    cdef double XION15[21],YION15[21],XION16[83],YION16[83],XION[50],YIONG[50],YIONC[50],Z1T[25],Z6T[25],EBRM[25]
    cdef int IOFFION[16], IOFFN[250]

    XEN=gd['gas9/XEN']
    YMT=gd['gas9/YMT']
    YEL=gd['gas9/YEL']
    YEPS=gd['gas9/YEPS']
    XATT1=gd['gas9/XATT1']
    YATT1=gd['gas9/YATT1']
    XATT2=gd['gas9/XATT2']
    YATT2=gd['gas9/YATT2']
    XVIB1=gd['gas9/XVIB1']
    YVIB1=gd['gas9/YVIB1']
    XVIB2=gd['gas9/XVIB2']
    YVIB2=gd['gas9/YVIB2']
    XVIB3=gd['gas9/XVIB3']
    YVIB3=gd['gas9/YVIB3']
    XVIB4=gd['gas9/XVIB4']
    YVIB4=gd['gas9/YVIB4']
    XVIB5=gd['gas9/XVIB5']
    YVIB5=gd['gas9/YVIB5']
    XTR1=gd['gas9/XTR1']
    YTR1=gd['gas9/YTR1']
    XTR2=gd['gas9/XTR2']
    YTR2=gd['gas9/YTR2']
    XTR3=gd['gas9/XTR3']
    YTR3=gd['gas9/YTR3']
    XNUL1=gd['gas9/XNUL1']
    YNUL1=gd['gas9/YNUL1']
    XNUL2=gd['gas9/XNUL2']
    YNUL2=gd['gas9/YNUL2']
    XNUL3=gd['gas9/XNUL3']
    YNUL3=gd['gas9/YNUL3']
    XION1=gd['gas9/XION1']
    YION1=gd['gas9/YION1']
    XION2=gd['gas9/XION2']
    YION2=gd['gas9/YION2']
    XION3=gd['gas9/XION3']
    YION3=gd['gas9/YION3']
    XION4=gd['gas9/XION4']
    YION4=gd['gas9/YION4']
    XION5=gd['gas9/XION5']
    YION5=gd['gas9/YION5']
    XION6=gd['gas9/XION6']
    YION6=gd['gas9/YION6']
    XION7=gd['gas9/XION7']
    YION7=gd['gas9/YION7']
    XION8=gd['gas9/XION8']
    YION8=gd['gas9/YION8']
    XION9=gd['gas9/XION9']
    YION9=gd['gas9/YION9']
    XION10=gd['gas9/XION10']
    YION10=gd['gas9/YION10']
    XION11=gd['gas9/XION11']
    YION11=gd['gas9/YION11']
    XION12=gd['gas9/XION12']
    YION12=gd['gas9/YION12']
    XION13=gd['gas9/XION13']
    YION13=gd['gas9/YION13']
    XION14=gd['gas9/XION14']
    YION14=gd['gas9/YION14']
    XION15=gd['gas9/XION15']
    YION15=gd['gas9/YION15']
    XION16=gd['gas9/XION16']
    YION16=gd['gas9/YION16']
    XION=gd['gas9/XION']
    YIONG=gd['gas9/YIONG']
    YIONC=gd['gas9/YIONC']
    Z1T=gd['gas9/Z1T']
    Z6T=gd['gas9/Z6T']
    EBRM=gd['gas9/EBRM']

    cdef double A0,RY,CONST,EMASS2,API,BBCONST,AM2,C,
    cdef int NBREM,NASIZE,NDATA,NIOND,NION1,NION2,NION3,NION4,NION5,NION6,NION7,NION8,NION9,NION10,NION11,NION12,NION13
    cdef int NION14,NION15,NION16,NATT1,NATT2,NVIB1,NVIB2,NVIB3,NVIB4,NVIB5,NTR1,NTR2,NTR3,NUL1,NUL2,NUL3,i,j,I,J,NL

    # BORN-BETHE CONSTANTS
    A0=0.52917720859e-08
    RY=13.60569193
    CONST=1.8738843-20
    EMASS2=1021997.804
    API=acos(-1.0)
    BBCONST=16.0*API*A0*A0*RY*RY/EMASS2

    # BORN BETHE FOR IONISATION
    AM2=7.21
    C=70.5
    # ARRAY SIZE
    NASIZE=4000
    object.NION=16
    object.NATT=2
    object.NIN=55
    object.NNULL=3
    NBREM=25

    for i in range(6):
        object.KEL[i] = object.WhichAngularModel
    for i in range(10):
        object.KIN[i] = 0
    for i in range(10,object.NIN):
        object.KIN[i] = 2
    NDATA=164

    NIOND=50
    NION1=31
    NION2=31
    NION3=31
    NION4=30
    NION5=29
    NION6=29
    NION7=26
    NION8=26
    NION9=25
    NION10=24
    NION11=24
    NION12=24
    NION13=23
    NION14=21
    NION15=21
    NION16=83
    NATT1=11
    NATT2=9
    NVIB1=29
    NVIB2=28
    NVIB3=28
    NVIB4=46
    NVIB5=16
    NTR1=12
    NTR2=11
    NTR3=11
    NUL1=25
    NUL2=13
    NUL3=14
    
    object.SCLN[0:3]=[1.0,10,10]
    cdef double EMASS = 9.10938291e-31
    cdef double AMU = 1.660538921e-27, EOBY[16],SCLOBY,APOP1,APOP2,APOP3,APOP4,QCOUNT=0.0

    object.E = [0.0, 1.0, 11.52, 0.0, 0.0, 0.0]
    object.E[1] = 2.0 * EMASS / (30.06964 * AMU)

    object.EION[0:16]=[11.52,12.05,12.65,13.65,14.8,14.8,20.5,21.5,25.8,26.2,32.0,32.5,36.0,37.0,37.0,285.0]

    # OPAL BEATY
    SCLOBY = 1.0

    for J in range(object.NION):
        EOBY[J] = object.EION[J]*SCLOBY
    EOBY[object.NION-1]=object.EION[object.NION-1]*0.63

    for J in range(15):
        object.NC0[J] =0
        object.EC0[J]=0.0
        object.WK[J]=0.0
        object.EFL[J]=0.0
        object.NG1[J]=0
        object.EG1[J]=0.0
        object.NG2[J]=0
        object.EG2[J] = 0.0
    #DOUBLE CHARGE , ++ ION STATES ( EXTRA ELECTRON )
    object.NC0[10]=1
    object.EC0[10]=6.0
    #FLUORESCENCE DATA  (KSHELL)
    object.NC0[15]=2
    object.EC0[15]=253
    object.WK[15]=0.0026
    object.EFL[15]=273
    object.NG1[15]=1
    object.EG1[15]=253
    object.NG2[15]=2
    object.EG2[15]=5.0

    #OFFSET ENERGY FOR IONISATION ELECTRON ANGULAR DISTRIBUTION
    for j in range(0, object.NION):
        for i in range(0, 4000):
            if (object.EG[i] > object.EION[j]):
                IOFFION[j] = i - 1
                break

    object.EIN = gd['gas9/EIN']
    #OFFSET ENERGY FOR EXCITATION LEVELS ANGULAR DISTRIBUTION
    
    for NL in range(object.NIN):
        for i in range(4000):
            if object.EG[i] > object.EIN[NL]:
                IOFFN[NL] = i -1
                break

    for i in range(object.NIN):
        for j in range(3):
            object.PenningFraction[j][i]=0.0


    # CALC LEVEL POPULATIONS
    APOP1=exp(object.EIN[0]/object.AKT)
    APOP2=exp(object.EIN[2]/object.AKT)
    APOP3=exp(object.EIN[4]/object.AKT)
    APOP4=exp(object.EIN[6]/object.AKT)
    for J in range(NBREM):
        EBRM[J] = exp(EBRM[J])
    cdef double EN,ENLG,GAMMA1,GAMMA2,BETA,BETA2,QMT,QEL,PQ[3],X1,X2,QBB=0.0,QSUM,EFAC,F[42]
    cdef int FI
    F=[0.000136,0.00174,0.008187,0.006312,0.011877,0.020856,0.031444,0.39549,0.042350,0.041113,0.038256,0.036556,0.096232,
       .083738,.043456,.047436,.047800,.048914,.054353,.061019,.244430,.284790,.095973,.090728,0.071357,.074875,.054542,
       .022479,.008585,.004524,.004982,.010130,.013320,.013310,.010760,.009797,.009198,.008312,.007139,.004715,.002137,.000662]

    object.EnergySteps=4000
    for I in range(object.EnergySteps):
        EN = object.EG[I]
        ENLG = log(EN)
        GAMMA1 = (EMASS2 + 2 * EN) / EMASS2
        GAMMA2 = GAMMA1 * GAMMA1
        BETA = sqrt(1.0 - 1.0 / GAMMA2)
        BETA2 = BETA * BETA
        
        if EN<=10:
            QMT = GasUtil.CALPQ3(EN, NDATA,YMT, XEN)*1e-16
            QEL = GasUtil.CALPQ3(EN, NDATA,YEL, XEN)*1e-16
            PQ[2] = GasUtil.CALPQ3(EN, NDATA,YEPS,XEN)
        else:
            QEL = GasUtil.QLSCALE(EN, NDATA, YEL, XEN)
            QMT = GasUtil.QLSCALE(EN, NDATA, YMT, XEN)
            PQ[2] = GasUtil.QLSCALE(EN, NDATA, YEPS, XEN)*1e16
        PQ[2]=1-PQ[2]
        PQ[1]=0.5+(QEL-QMT)/QEL
        PQ[0]=0.5
        object.PEQEL[1][I] = PQ[object.WhichAngularModel]

        object.Q[1][I] = QEL

        if object.WhichAngularModel==0:
            object.Q[1][I] = QMT

        # IONISATION

        for i in range(object.NION):
            object.QION[i][I] = 0.0
            object.PEQION[i][I]=0.5
            if object.WhichAngularModel==2:
                object.PEQION[i][I]=0.0

        # C2H6+
        if EN>object.EION[0]:
            object.QION[0][I] = GasUtil.CALQION(EN,NION1, YION1, XION1)
            if object.QION[0][I]==0:
                if EN<=XION[NIOND-1]:
                    QCOUNT = GasUtil.QLSCALE(EN, NIOND, YIONC, XION)
                    # fraction of QCOUNT
                    object.QION[0][I] = QCOUNT * 0.1378
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    X2 = 1 / BETA2
                    X1 = X2 * log(BETA2 / (1 - BETA2)) - 1
                    QBB = CONST * (AM2 * (X1 - object.DEN[I] / 2) + C * X2)
                    object.QION[0][I] = QBB*0.1378
            if EN >=2*object.EION[0]:
                object.PEQION[0][I] = object.PEQEL[1][I-IOFFION[0]]

        # C2H4+
        if EN>object.EION[1]:
            object.QION[1][I] = GasUtil.CALQION(EN,NION2, YION2, XION2)
            if object.QION[1][I]==0:
                if EN<=XION[NIOND-1]:
                    # fraction of QCOUNT
                    object.QION[1][I] = QCOUNT * 0.4481
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[1][I] = QBB*0.4481
            if EN >=2*object.EION[1]:
                object.PEQION[1][I] = object.PEQEL[1][I-IOFFION[1]]

        # C2H5+
        if EN>object.EION[2]:
            object.QION[2][I] = GasUtil.CALQION(EN,NION3, YION3, XION3)
            if object.QION[2][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[2][I] = QCOUNT * 0.1104
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[2][I] = QBB*0.1104
            if EN >=2*object.EION[2]:
                object.PEQION[2][I] = object.PEQEL[1][I-IOFFION[2]]


        # CH3+
        if EN>object.EION[3]:
            object.QION[3][I] = GasUtil.CALQION(EN,NION4, YION4, XION4)
            if object.QION[3][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[3][I] = QCOUNT * 0.01718
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[3][I] = QBB*0.01718
            if EN >=2*object.EION[3]:
                object.PEQION[3][I] = object.PEQEL[1][I-IOFFION[3]]

        # C2H3+
        if EN>object.EION[4]:
            object.QION[4][I] = GasUtil.CALQION(EN,NION5, YION5, XION5)
            if object.QION[4][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[4][I] = QCOUNT * 0.1283
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[4][I] = QBB*0.1283
            if EN >=2*object.EION[4]:
                object.PEQION[4][I] = object.PEQEL[1][I-IOFFION[4]]

        # C2H2+
        if EN>object.EION[5]:
            object.QION[5][I] = GasUtil.CALQION(EN,NION6, YION6, XION6)
            if object.QION[5][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[5][I] = QCOUNT * 0.07
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[5][I] = QBB*0.07
            if EN >=2*object.EION[5]:
                object.PEQION[5][I] = object.PEQEL[1][I-IOFFION[5]]
                

        # H+
        if EN>object.EION[6]:
            object.QION[6][I] = GasUtil.CALQION(EN,NION7, YION7, XION7)
            if object.QION[6][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[6][I] = QCOUNT * 0.000011
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[6][I] = QBB*0.000011
            if EN >=2*object.EION[6]:
                object.PEQION[6][I] = object.PEQEL[1][I-IOFFION[6]]

        # H2+
        if EN>object.EION[7]:
            object.QION[7][I] = GasUtil.CALQION(EN,NION8, YION8, XION8)
            if object.QION[7][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[7][I] = QCOUNT * 0.00036
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[7][I] = QBB*0.00036
            if EN >=2*object.EION[7]:
                object.PEQION[7][I] = object.PEQEL[1][I-IOFFION[7]]

        # CH2+
        if EN>object.EION[8]:
            object.QION[8][I] = GasUtil.CALQION(EN,NION9, YION9, XION9)
            if object.QION[8][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[8][I] = QCOUNT * 0.0066
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[8][I] = QBB*0.0066
            if EN >=2*object.EION[8]:
                object.PEQION[8][I] = object.PEQEL[1][I-IOFFION[8]]

        # C2H+
        if EN>object.EION[9]:
            object.QION[9][I] = GasUtil.CALQION(EN,NION10, YION10, XION10)
            if object.QION[9][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[9][I] = QCOUNT * 0.0062
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[9][I] = QBB*0.0062
            if EN >=2*object.EION[9]:
                object.PEQION[9][I] = object.PEQEL[1][I-IOFFION[9]]

        # C2H6++
        if EN>object.EION[10]:
            object.QION[10][I] = GasUtil.CALQION(EN,NION11, YION11, XION11)
            if object.QION[10][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[10][I] = QCOUNT * 0.0745
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[10][I] = QBB*0.0745
            if EN >=2*object.EION[10]:
                object.PEQION[10][I] = object.PEQEL[1][I-IOFFION[10]]


        # H3+
        if EN>object.EION[11]:
            object.QION[11][I] = GasUtil.CALQION(EN,NION12, YION12, XION12)
            if object.QION[11][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[11][I] = QCOUNT * 0.0000055
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[11][I] = QBB*0.0000055
            if EN >=2*object.EION[11]:
                object.PEQION[11][I] = object.PEQEL[1][I-IOFFION[11]]

        # CH+
        if EN>object.EION[12]:
            object.QION[12][I] = GasUtil.CALQION(EN,NION13, YION13, XION13)
            if object.QION[12][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[12][I] = QCOUNT * 0.00037
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[12][I] = QBB*0.00037
            if EN >=2*object.EION[12]:
                object.PEQION[12][I] = object.PEQEL[1][I-IOFFION[12]]


        # C2+
        if EN>object.EION[13]:
            object.QION[13][I] = GasUtil.CALQION(EN,NION14, YION14, XION14)
            if object.QION[13][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[13][I] = QCOUNT * 0.000022
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[13][I] = QBB*0.000022
            if EN >=2*object.EION[13]:
                object.PEQION[13][I] = object.PEQEL[1][I-IOFFION[13]]

        # C+
        if EN>object.EION[14]:
            object.QION[14][I] = GasUtil.CALQION(EN,NION15, YION15, XION15)
            if object.QION[14][I]==0:
                if EN<=XION[NIOND-1]:
                    #USE FRACTION OF COUNTING IONISATION ABOVE 600 EV
                    # fraction of QCOUNT
                    object.QION[14][I] = QCOUNT * 0.00011
                else:
                    #USE BORN-BETHE X-SECTION ABOVE X[n] EV
                    object.QION[14][I] = QBB*0.00011
            if EN >=2*object.EION[14]:
                object.PEQION[14][I] = object.PEQEL[1][I-IOFFION[14]]
        
        #CARBON K-SHELL
        if EN > object.EION[15]:
            object.QION[15][I] = GasUtil.CALQIONREG(EN, NION16,YION16, XION16)*2
        if EN >=2*object.EION[15]:
                object.PEQION[15][I] = object.PEQEL[1][I-IOFFION[15]]

        QSUM = 0.0
        for J in range(15):
            QSUM +=object.QION[J][I]
        if QSUM!=0.0:
                for J in range(15):
                    object.QION[J][I]*=(QSUM-object.QION[15][I])/QSUM

        object.Q[3][I] = 0.0
        
        object.QATT[0][I] = 0.0
        # ATTACHMENT to H-
        if EN>=XATT1[0]:
            object.QATT[0][I] = GasUtil.CALQION(EN, NATT1,YATT1, XATT1)

        object.QATT[1][I] = 0.0
        # ATTACHMENT to CH2-
        if EN>=XATT2[0]:
            object.QATT[1][I] = GasUtil.CALQION(EN, NATT2,YATT2, XATT2)

        object.Q[4][I] = 0.0
        object.Q[5][I] = 0.0

        # set ZEROS

        for J in range(object.NIN):
            object.QIN[J][I] = 0.0
            object.PEQIN[J][I] = 0.0

        # SUPERELASTIC VIBRATION-TORSION         AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN>0.0:
            EFAC = sqrt(1.0-(object.EIN[0]/EN))
            object.QIN[0][I]=0.0045*log((EFAC+1.0)/(EFAC-1.0))/EN
            object.QIN[0][I]*=APOP1/(1.0+APOP1)*1.e-16
        if EN>10:
            object.PEQIN[0][I] = object.PEQEL[1][I-IOFFN[0]]


        #VIBRATION-TORSION                      AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN>object.EIN[1]:
            EFAC = sqrt(1.0-(object.EIN[1]/EN))
            object.QIN[1][I]=0.0045*log((EFAC+1.0)/(1.0-EFAC))/EN
            object.QIN[1][I]*=1.0/(1.0+APOP1)*1.e-16
        if EN>10:
            object.PEQIN[1][I] = object.PEQEL[1][I-IOFFN[1]]

        #SUPERELASTIC VIB1                     AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN>0.0:
            object.QIN[2][I] = GasUtil.CALQINVISO(EN,NVIB1, YVIB1, XVIB1, APOP2/(1+APOP2), object.EIN[3],1, -1*5*EN,0)
        if EN>10:
            object.PEQIN[2][I] = object.PEQEL[1][I-IOFFN[2]]


        #VIB1                           AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN >object.EIN[3]:
            object.QIN[3][I] = GasUtil.CALQINVISO(EN,NVIB1, YVIB1, XVIB1, 1/(1+APOP2), 0,1, -1*5*EN,0)
        if EN>10:
            object.PEQIN[3][I] = object.PEQEL[1][I-IOFFN[3]]

        #SUPERELASTIC VIB2                     AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN>0.0:
            object.QIN[4][I] = GasUtil.CALQINVISO(EN,NVIB2, YVIB2, XVIB2, APOP3/(1+APOP3), object.EIN[5],1, -1*5*EN,0)
        if EN>10:
            object.PEQIN[4][I] = object.PEQEL[1][I-IOFFN[4]]

        #VIB2                           AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN >object.EIN[5]:
            object.QIN[5][I] = GasUtil.CALQINVISO(EN,NVIB2, YVIB2, XVIB2, 1/(1+APOP3), 0,1, -1*5*EN,0)
        if EN>10:
            object.PEQIN[5][I] = object.PEQEL[1][I-IOFFN[5]]
        
        #SUPERELASTIC VIB3                     AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN>0.0:
            object.QIN[6][I] = GasUtil.CALQINVISO(EN,NVIB3, YVIB3, XVIB3, APOP4/(1+APOP4), object.EIN[7],1, -1*5*EN,0)
        if EN>10:
            object.PEQIN[6][I] = object.PEQEL[1][I-IOFFN[6]]


        #VIB3                           AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN >object.EIN[7]:
            object.QIN[7][I] = GasUtil.CALQINVISO(EN,NVIB3, YVIB3, XVIB3, 1/(1+APOP4), 0,1, -1*5*EN,0)
        if EN>10:
            object.PEQIN[7][I] = object.PEQEL[1][I-IOFFN[7]]

        #VIB4                           AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN >object.EIN[8]:
            object.QIN[8][I] = GasUtil.CALQINVISO(EN,NVIB4, YVIB4, XVIB4, 1, 0,1, -1*5*EN,0)
        if EN>10:
            object.PEQIN[8][I] = object.PEQEL[1][I-IOFFN[8]]
    
        #VIB HARMONICS                  AAnisotropicDetectedTROPIC ABOVE 10 EV
        if EN >object.EIN[9]:
            object.QIN[9][I] = GasUtil.CALQINVISO(EN,NVIB5, YVIB5, XVIB5, 1, 0,1, -1*5*EN,0)
        if EN>10:
            object.PEQIN[9][I] = object.PEQEL[1][I-IOFFN[9]]

        # EXCITATION TO TRIPLET AND SINGLET LEVELS
        #
        # FIRST TRIPLET AT  6.85 EV
        
        if EN>object.EIN[10]:
            object.QIN[10][I]= GasUtil.CALQINP(EN, NTR1, YTR1, XTR1,2)*100
        if EN>3*object.EIN[10]:
            object.PEQIN[10][I] = object.PEQEL[1][I-IOFFN[10]]

        #SINGLET DISSOCIATION AT  7.93  EV     BEF SCALING F[FI]
        FI = 0
        J=11        

        if EN > object.EIN[J]:
            object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                        log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                    I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2]) 
            if object.QIN[J][I]<0.0:
                object.QIN[J][I] = 0.0
        if EN > 3 * object.EIN[J]:
            object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
        FI+=1
        

        # SECOND TRIPLET AT  8.00 EV
        if EN>object.EIN[12]:
            object.QIN[12][I]= GasUtil.CALQINP(EN, NTR2, YTR2, XTR2,2)*100
        if EN>3*object.EIN[12]:
            object.PEQIN[12][I] = object.PEQEL[1][I-IOFFN[12]]


        for J in range(13,24):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                            log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                        I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
                if object.QIN[J][I]<0.0:
                    object.QIN[J][I] = 0.0
            if EN > 3 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI+=1
        
        # THIRD TRIPLET AT  10.00 EV
        if EN>object.EIN[24]:
            object.QIN[24][I]= GasUtil.CALQINP(EN, NTR3, YTR3, XTR3,2)*100
        if EN>3*object.EIN[24]:
            object.PEQIN[24][I] = object.PEQEL[1][I-IOFFN[24]]

        for J in range(25,55):
            if EN > object.EIN[J]:
                object.QIN[J][I] = F[FI] / (object.EIN[J] * BETA2) * (
                            log(BETA2 * GAMMA2 * EMASS2 / (4.0 * object.EIN[J])) - BETA2 - object.DEN[
                        I] / 2.0) * BBCONST * EN / (EN + object.EIN[J] + object.E[2])
                if object.QIN[J][I]<0.0:
                    object.QIN[J][I] = 0.0
            if EN > 3 * object.EIN[J]:
                object.PEQIN[J][I] = object.PEQEL[1][I - IOFFN[J]]
            FI+=1

        #LOAD BREMSSTRAHLUNG X-SECTION
        object.QIN[55][I] = 0.0
        object.QIN[56][I] = 0.0
        if EN>1000:
            object.QIN[55][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z6T, EBRM)*2e-8
            object.QIN[56][I] = GasUtil.QLSCALE(exp(EN), NBREM, Z1T, EBRM)*6e-8

        # LOAD NULL COLLISIONS
        object.QNULL[0][I]=0.0
        if EN>XNUL1[0]:
            object.QNULL[0][I] = GasUtil.CALQINP(EN, NUL1, YNUL1, XNUL1,1)*100 *0.9*object.SCLN[0]

        # LIGHT EMISSION FROM H ALPHA   
        #   MOHLMANN AND DE HEER  CHEM.PHYS.19(1979)233 
        object.QNULL[1][I]=0.0
        if EN>XNUL2[0]:
            object.QNULL[1][I] = GasUtil.CALQINP(EN, NUL2, YNUL2, XNUL2,1)*100 *object.SCLN[1]

        # LIGHT EMISSION FROM CH2(A2DELTA - X2PI)
        #  MOHLMANN AND DE HEER  CHEM.PHYS.19(1979)233 
        object.QNULL[2][I]=0.0
        if EN>XNUL3[0]:
            object.QNULL[2][I] = GasUtil.CALQINP(EN, NUL3, YNUL3, XNUL3,1)*100 *object.SCLN[2]



    for J in range(object.NIN):
        if object.EFINAL <= object.EIN[J]:
            object.NIN = J
            break
    return
