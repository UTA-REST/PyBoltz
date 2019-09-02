from PyBoltz cimport PyBoltz
from libc.math cimport acos, sqrt
cimport numpy as np
import  numpy as np
import cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Setup(PyBoltz object):
    """
    This function sets up the given PyBoltz object. It fills the values of the main constants. 
        
    The object parameter is the PyBoltz object to be setup.
    """
    TwoPi = 2.0 * np.pi
    object.RhydbergConst = <float>(13.60569253)
    object.PIR2 = 8.7973554297e-17
    ElectronCharge = 1.602176565e-19
    ElectronMass = 9.10938291e-31
    AMU = 1.660538921e-27
    BoltzmannConst_eV = 8.6173324e-5
    BoltzmannConst_eVJ = 1.3806488e-23
    MassOverChargeDivTen = 1.758820088e10
    ALOSCH = 2.6867805e19
    EOVM = sqrt(2.0 * ElectronCharge / ElectronMass) * 100.0
    ZeroCelcius = 273.15
    OneAtmosphere = 760.0
    object.CONST1 = MassOverChargeDivTen / 2.0 * 1.0e-19
    object.CONST2 = object.CONST1* 1.0e-02
    object.CONST3 = sqrt(0.2 * MassOverChargeDivTen) * 1.0e-9
    object.PresTempCor = ZeroCelcius * object.PressureTorr / (OneAtmosphere * (ZeroCelcius + object.TemperatureCentigrade) * 100.0)

    # Set long decorrelation length and step
    object.Decor_NCOLM = 2000000
    object.Decor_NCORLN = 500000
    object.Decor_NCORST = 2
    FRACM = 0.0
    MXEKR = 0

    # Set short decorrelation length and step for mixtures with more than 3% inelastic/molecular component
    for IH in range(object.NumberOfGases):
        if object.GasIDs[IH] != 2 and object.GasIDs[IH] != 6 and object.GasIDs[IH] != 7 and object.GasIDs[IH] != 3 and \
                object.GasIDs[IH] != 4 and object.GasIDs[IH] != 5:
            # Molecular gas sum total fraction
            FRACM += object.GasFractions[IH]

    # If greater than 3% molecular/inelastic fraction, or large electric field use short decorrelation length.
    if object.EField > (10 / object.PresTempCor) or FRACM>3:
            object.Decor_NCOLM = 400000
            object.Decor_NCORLN = 50000
            object.Decor_NCORST = 4
    TOTFRAC = 0.0
    if object.NumberOfGases == 0 or object.NumberOfGases > 6:
        raise ValueError("Error in Gas Input")

    for J in range(object.NumberOfGases):
        if object.GasIDs[J] == 0 or object.GasFractions[J] == 0:
            raise ValueError("Error in Gas Input")
        TOTFRAC += object.GasFractions[J]
    if abs(TOTFRAC - 100) >= 1e-6:
        raise ValueError("Error in Gas Input")
    object.MaxNumberOfCollisions = object.MaxNumberOfCollisions 

    if object.MaxNumberOfCollisions < 0:
        raise ValueError("MaxNumberOfCollisions value is too large - overflow")
    object.EnergySteps = 4000
    object.AngleFromZ = 0.785
    object.AngleFromX = 0.1

    object.VelocityX = 0.0
    object.VelocityY = 0.0
    object.VelocityZ = 0.0

    object.InitialElectronEnergy = object.FinalElectronEnergy / 50.0

    for i in range(6):
        object.ANN[i] = object.GasFractions[i] * object.PresTempCor * ALOSCH
    for i in range(6):
        object.VANN[i] = object.GasFractions[i] * object.PresTempCor * object.CONST3 * ALOSCH 

    # Radians per picosecond
    object.AngularSpeedOfRotation = MassOverChargeDivTen * object.BFieldMag * 1e-12

    if object.BFieldMag == 0:
        return

    # Metres per picosecond
    object.EFieldOverBField = object.EField * 1e-9 / object.BFieldMag
    return


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef SetupT(PyBoltz object):
    """
    This function sets up the given PyBoltz object. It fills the values of the main constants. 
    
    The object parameter is the PyBoltz object to be setup.
    """
    cdef double TwoPi,  ElectronCharge, ElectronMass, AMU, BoltzmannConst_eV, BoltzmannConst_eVJ, MassOverChargeDivTen, ALOSCH, EOVM, ZeroCelcius, OneAtmosphere, TOTFRAC
    cdef long long MXEKR, IH,  i

    TwoPi = 2.0 * np.pi
    object.RhydbergConst = <float>(13.60569253)
    object.PIR2 = 8.7973554297e-17
    ElectronCharge = 1.602176565e-19
    ElectronMass = 9.10938291e-31
    AMU = 1.660538921e-27
    BoltzmannConst_eV = 8.6173324e-5
    BoltzmannConst_eVJ = 1.3806488e-23
    MassOverChargeDivTen = 1.758820088e10
    ALOSCH = 2.6867805e19
    EOVM = sqrt(2.0 * ElectronCharge / ElectronMass) * 100.0
    ZeroCelcius = 273.15
    OneAtmosphere = 760.0
    object.CONST1 = MassOverChargeDivTen / 2.0 * 1.0e-19
    object.CONST2 = object.CONST1 * 1.0e-02
    object.CONST3 = sqrt(0.2 * MassOverChargeDivTen) * 1.0e-9
    object.PresTempCor = ZeroCelcius * object.PressureTorr / (OneAtmosphere * (ZeroCelcius + object.TemperatureCentigrade) * 100.0)

    # Set long decorrelation length and step
    object.Decor_NCOLM = 2000000
    object.Decor_NCORLN = 500000
    object.Decor_NCORST = 2

    # Set short decorrelation length and step for mixtures with more than 3% inelastic/molecular component
    cdef double FRACM = 0.0
    MXEKR = 0
    for IH in range(object.NumberOfGases):
        if object.GasIDs[IH] != 2 and object.GasIDs[IH] != 6 and object.GasIDs[IH] != 7 and object.GasIDs[IH] != 3 and \
                object.GasIDs[IH] != 4 and object.GasIDs[IH] != 5:
            # Molecular gas sum total fraction
            FRACM += object.GasFractions[IH]
    # If greater than 3% molecular/inelastic fraction, or large electric field use short decorrelation length.
    if (object.EField > (10.0 / object.PresTempCor)) or (FRACM > 3):
            object.Decor_NCOLM = 400000
            object.Decor_NCORLN = 50000
            object.Decor_NCORST = 4
    TOTFRAC = 0.0

    if object.NumberOfGases == 0 or object.NumberOfGases > 6:
        raise ValueError("Error in Gas Input")
    for J in range(object.NumberOfGases):
        if object.GasIDs[J] == 0 or object.GasFractions[J] == 0:
            raise ValueError("Error in Gas Input")
        TOTFRAC += object.GasFractions[J]
    if abs(TOTFRAC - 100) >= 1e-6:
        raise ValueError("Error in Gas Input")


    object.MaxNumberOfCollisions = object.MaxNumberOfCollisions

    if object.MaxNumberOfCollisions < 0:
        raise ValueError("NMAX value is too large - overflow")
    object.EnergySteps = 4000
    object.AngleFromZ = 0.785
    object.AngleFromX = 0.1
    object.InitialElectronEnergy = object.FinalElectronEnergy / 50.0
    object.PresTempCor = ZeroCelcius * object.PressureTorr / (OneAtmosphere * (ZeroCelcius + object.TemperatureCentigrade) * 100.0)

    object.ThermalEnergy = (ZeroCelcius + object.TemperatureCentigrade) * BoltzmannConst_eV
    for i in range(6):
        object.ANN[i] = object.GasFractions[i] * object.PresTempCor * ALOSCH
    object.AN = 100.0 * object.PresTempCor * ALOSCH
    for i in range(6):
        object.VANN[i] = object.GasFractions[i] * object.PresTempCor * object.CONST3 * ALOSCH
    object.VAN = 100.0 * object.PresTempCor * object.CONST3 * ALOSCH

    # Radians per picosecond
    object.AngularSpeedOfRotation = MassOverChargeDivTen * object.BFieldMag * 1e-12


    if object.BFieldMag == 0:
        return
    # Metres per picosecond
    object.EFieldOverBField = object.EField * 1e-9 / object.BFieldMag
    return
