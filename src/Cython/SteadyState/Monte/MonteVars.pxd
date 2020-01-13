ctypedef struct MonteVars:
    # Number of collisions used to estimate starting values, else use number specified by user
    int NumberOfMaxColli = 80000000
    int ID = 0, I100=0,NumberOfCollision = 0,NumberOfNullCollision = 0,NumberOfElectron = 0
    int NumberOfElectronIon = 0,NTPMFlag =0,NMXADD = 0, NPONT = 0,NCLUS = 0,J1 = 1
    int PrintN,IPrint=0,Iterator=0,IPlane=0,IPrimary=0, iEnergyBin
    double StartingEnergy,SpaceZStart = 0.0,TimeSumStart = 0.0, AbsFakeIoniz = 0.0,TimeStop =0.0
    double FakeIonisationsTime[8],FakeIonisationsSpace[8], TDash = 0.0,T = 0.0,AP,BP
    double DirCosineZ1,DirCosineX1,DirCosineY1,Energy
    double Energy100, DirCosineZ100,DirCosineY100,DirCosineX100,F1,F2,TwoPi,TwoM,Sqrt2M
    double Attachment, Ionisation