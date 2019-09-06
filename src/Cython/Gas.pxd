ctypedef struct Gas:
    int NGS, NIN, EnergySteps, WhichAngularModel, NATT, NNULL, NION, EnablePenning, NumberOfGases
    double PRESSURE, TemperatureC, RhydbergConst, EnergyStep, AKT, EFINAL, DENS,PIR2
    double Q[6][4000],QIN[250][4000],E[6],EIN[250],KIN[250],QION[30][4000],PEQION[30][4000],EION[30],EB[30],PEQEL[6][4000]
    double PEQIN[250][4000],KEL[6],PenningFraction[3][290],NC0[30],EC0[30],WK[30],EFL[30],NG1[30],EG1[30],NG2[30],EG2[30]
    double QATT[8][4000],QNULL[10][4000],SCLN[10],EG[4000],EROOT[4000],QT1[4000],QT2[4000],QT3[4000],QT4[4000],DEN[4000]
    double EMT[182], ET[153],EAT[182]
