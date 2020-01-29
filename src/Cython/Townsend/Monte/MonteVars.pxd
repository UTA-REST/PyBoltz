ctypedef struct MonteVars:
    # Number of collisions used to estimate starting values, else use number specified by user
    int NumberOfMaxColli
    int ID, I100,NumberOfCollision,NumberOfNullCollision,NumberOfElectron
    int NumberOfElectronAtt,NTPMFlag,NMXADD, NPONT,NCLUS,J1
    int PrintN,IPrint,Iterator,IPlane, iEnergyBin,FFFlag
    double StartingEnergy,SpaceZStart,TimeSumStart, AbsFakeIoniz,TimeStop,TimeStop1,TimeStop2,ZPlanes[10]
    double FakeIonisationsTime[9],FakeIonisationsSpace[11], TDash,T,AP,BP
    double DirCosineZ1,DirCosineX1,DirCosineY1,Energy
    double Energy100, DirCosineZ100,DirCosineY100,DirCosineX100,F1,F2,TwoPi,TwoM,Sqrt2M
    double Attachment, Ionisation, TOld,MaxSpaceZ1
    double XS[2001], YS[2001], ZS[2001], TS[2001], ES[2001], DirCosineX[2001], DirCosineY[2001], DirCosineZ[2001]
    int IPlaneS[2001],ISolution,FFlag
