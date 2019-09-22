ctypedef struct Gas:
    int GasNumber, N_Inelastic, EnergySteps, WhichAngularModel, N_Attachment, N_Null, N_Ionization, EnablePenning, NumberOfGases
    double Pressure, TemperatureC, RhydbergConst, EnergyStep, ThermalEnergy, FinalEnergy, DENS,PIR2
    double Q[6][4000],InelasticCrossSectionPerGas[250][4000],E[6],EnergyLevels[250],KIN[250],IonizationCrossSection[30][4000],PEIonizationCrossSection[30][4000],IonizationEnergy[30],PEElasticCrossSection[6][4000]
    double PEInelasticCrossSectionPerGas[250][4000],AngularModel[6],PenningFraction[3][290],NC0[30],EC0[30],WK[30],EFL[30],NG1[30],EG1[30],NG2[30],EG2[30]
    double AttachmentCrossSection[8][4000],NullCrossSection[10][4000],ScaleNull[10],EG[4000],SqrtEnergy[4000],QT1[4000],QT2[4000],QT3[4000],QT4[4000],DEN[4000]
    double EMT[182], ET[153],EAT[182]
