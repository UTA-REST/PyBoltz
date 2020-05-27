ctypedef struct MonteVars:
    # Number of collisions used to estimate starting values, else use number specified by user
    int NumberOfMaxColli
    #'''The total number of collisions to be simulated'''
    int ID
    int I100
    #'''Iterator used to tell the monte carlo functions that it has been 100 collisions since our energy decorrelation
    # happened.'''
    int NumberOfCollision
    #'''Current number of collision in the simulation.'''
    int NumberOfNullCollision
    #'''Current number of null collisions in the simulation.'''
    int NumberOfElectron
    #'''Current number of simulated electrons.'''
    int NumberOfElectronAtt
    #'''Current number of electrons that got attached'''
    int SecondaryElectronFlag
    # Variable used to indicate whether there is a need to calculate the angles for a secondary electron that was generated.
    int ElectronStorageTop
    # Number that is always pointing at where the top of the electron storage stack.
    int TotalNumberOfElectrons
    # Number of simulated electrons + number of unstimulated electrons in the stack. If we cross this number,
    # this means we need a new primary electron in the simulation
    int Iterator
    #'''An iterator to indicate whether the simulation reached the maximum number of collisions.'''
    int IPlane
    #'''The number of the plane that the current electron is in. This could be indicating the number of
    #the current time or space plane.'''
    int iEnergyBin
    #'''An indexing variables that indicates what energy bin the current electron's energy falls in.'''
    int NewTimeFlag
    # A flag used to indicate whether there is a need to calculate a new time for the current electron and collision.
    # This is different from TimeStop.
    double StartingEnergy
    #'''The strarting energy of the electron in the simulation. By starting this is the electron's energy before it
    #gets into a collision, or the starting energy of the simulated electron'''
    double SpaceZStart
    #'''The start of the Z value for the current plane.'''
    double TimeSumStart
    #'''The starting time value for the current electron (the time when the electron was first simulated).'''
    double AbsFakeIoniz
    double TimeStop
    # Time at plane, if used in the pulsed townsend simulation. Else this is used as the time for when
    # the electron reaches space plane I, this happens in the steady state simulation.
    double TimeStop1
    # A temporary variable used to hold the other solutions for the TimeStop in the Steady State simulation.
    # This is mainly used as the equation used is a quadratic equation.
    double TimeStop2
    # Another temporary variable that is used to hole another solution for the TimeStop in the Steady State simulation.
    double ZPlanes[10]
    # This array holds the Z value for each space plane I.
    double FakeIonisationsTime[9]
    # This array will save the fake ionisation time binning for each time plane I. This is used in the Pulsed townsend
    # simulation.
    double FakeIonisationsSpace[11]
    # This array will save the fake ionisation space binning for each space plane I. This is used in the Steady State
    # simulation.
    double TDash
    # This value is used to store the total time for the current collision in the simulation, this doesn't include the
    # current T.
    double T
    # This  value is used to store the current time for the current collision, this includes the TDash value.
    double AP
    double BP
    double DirCosineZ1
    # This value stores the cosine of the current electron. This is calculated from the angle of the z axis.
    double DirCosineX1
    # This value stores the cosine of the current electron. This is calculated from the angle of the x axis.
    double DirCosineY1
    # This value stores the cosine of the current electron. This is calculated from the angle of the y axis.
    double Energy
    # This is the energy of the electron after the collision.
    double Energy100
    # This energy value is used to store the energy of every 200th electron. This is mainly used to give each new primary
    # electron a different starting energy.
    double DirCosineZ100
    # This cosine value stores the cosine value of the z axis for every 200th electron.
    double DirCosineY100
    # This cosine value stores the cosine value of the y axis for every 200th electron.
    double DirCosineX100
    # This cosine value stores the cosine value of the x axis for every 200th electron.
    double F1
    double F2
    double TwoPi
    # This value simply stores 2 * Pi.
    double TwoM
    double Sqrt2M
    double TOld
    double XS[2001]
    # This array is used to store the x position of the electrons in the storage stack.
    double YS[2001]
    # This array is used to store the y position of the electrons in the storage stack.
    double ZS[2001]
    # This array is used to store the z position of the electrons in the storage stack.
    double TS[2001]
    # This array is used to store the time of the electrons in the storage stack.
    double ES[2001]
    # This array is used to store the energy of the electrons in the storage stack.
    double DirCosineX[2001]
    # This array is used to store the cosine of the x axis of electrons in the storage stack.
    double DirCosineY[2001]
    # This array is used to store the cosine of the y axis of electrons in the storage stack.
    double DirCosineZ[2001]
    # This array is used to store the cosine of the z axis of electrons in the storage stack.
    int IPlaneS[2001]
    # This array is used to store the plane of each electron in the storage stack.
    int ISolution
    # This variable iss used to store whether the TimeStop calculation has two solutions or 1.
    int TimeCalculationFlag
    # Flag used to indicate whether there is a need to try and calculate a new TimeStop value.


