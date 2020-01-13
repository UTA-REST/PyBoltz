ctypedef struct MonteVars:
    # Number of collisions used to estimate starting values, else use number specified by user
    int NumberOfMaxColli
    int ID, I100,NumberOfCollision,NumberOfNullCollision,NumberOfElectron
    int NumberOfElectronIon,NTPMFlag,NMXADD, NPONT,NCLUS,J1
    int PrintN,IPrint,Iterator,IPlane, iEnergyBin
    double StartingEnergy,SpaceZStart,TimeSumStart, AbsFakeIoniz,TimeStop
    double FakeIonisationsTime[9],FakeIonisationsSpace[9], TDash,T,AP,BP
    double DirCosineZ1,DirCosineX1,DirCosineY1,Energy
    double Energy100, DirCosineZ100,DirCosineY100,DirCosineX100,F1,F2,TwoPi,TwoM,Sqrt2M
    double Attachment, Ionisation