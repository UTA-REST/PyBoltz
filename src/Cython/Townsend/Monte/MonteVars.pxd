ctypedef struct MonteVars:
    # Number of collisions used to estimate starting values, else use number specified by user
    int NumberOfMaxColli
    #'''The total number of collisions to be simulated'''
    int ID
    int I100
    #'''Iterator used to tell the monte carlo functions that it has been 100 collisions since our energy decorrelation
    #happened.'''
    int NumberOfCollision
    #'''Current number of collision in the simulation.'''
    int NumberOfNullCollision
    #'''Current number of null collisions in the simulation.'''
    int NumberOfElectron
    #'''Current number of simulated electrons.'''
    int NumberOfElectronAtt
    #'''Current number of electrons that got attached'''
    int NTPMFlag, NMXADD, NPONT, NCLUS, J1
    int PrintN, IPrint
    int Iterator
    #'''An iterator to indicate whether the simulation reached the maximum number of collisions.'''
    int IPlane
    #'''The number of the plane that the current electron is in. This could be indicating the number of
    #the current time or space plane.'''
    int iEnergyBin
    #'''An indexing variables that indicates what energy bin the current electron's energy falls in.'''
    int FFFlag
    double StartingEnergy
    #'''The strarting energy of the electron in the simulation. By starting this is the electron's energy before it
    #gets into a collision, or the starting energy of the simulated electron'''
    double SpaceZStart
    #'''The start of the Z value for the current plane.'''
    double TimeSumStart
    #'''The starting time value for the current electron (the time when the electron was first simulated).'''
    double AbsFakeIoniz, TimeStop, TimeStop1, TimeStop2, ZPlanes[10]
    double FakeIonisationsTime[9], FakeIonisationsSpace[11], TDash, T, AP, BP
    double DirCosineZ1, DirCosineX1, DirCosineY1, Energy
    double Energy100, DirCosineZ100, DirCosineY100, DirCosineX100, F1, F2, TwoPi, TwoM, Sqrt2M
    double Attachment, Ionisation, TOld, MaxSpaceZ1
    double XS[2001], YS[2001], ZS[2001], TS[2001], ES[2001], DirCosineX[2001], DirCosineY[2001], DirCosineZ[2001]
    int IPlaneS[2001], ISolution, FFlag
