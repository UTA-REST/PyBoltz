def MBSort(I, R2, IE, Object):
    """
    Select collision type from collision array by binary step sampling.
    Reduces sampling range to within 4 positions in array.
    Returns a new index.
    """
    iStep = int(Object.ISIZENT)
    Increment = 0
    for K in range(12):
        I = Increment
        if iStep == 2:
            if I == 0:
                return I
            return I - 1
        I = Increment + iStep
        if I <= Object.NumMomCrossSectionPointsNT:
            if Object.CollisionFrequencyNT[IE][I - 1] < R2:
                Increment = Increment + iStep
        iStep = int(iStep / 2)
    if I == 0:
        return I
    return I - 1
