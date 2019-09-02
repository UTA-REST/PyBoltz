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
    TWOPI = 2.0 * np.pi
    object.ARY = <float>(13.60569253)
    object.PIR2 = 8.7973554297e-17
    ECHARG = 1.602176565e-19
    EMASS = 9.10938291e-31
    AMU = 1.660538921e-27
    BOLTZ = 8.6173324e-5
    BOLTZJ = 1.3806488e-23
    AWB = 1.758820088e10
    ALOSCH = 2.6867805e19
    EOVM = sqrt(2.0 * ECHARG / EMASS) * 100.0
    ABZERO = 273.15
    ATMOS = 760.0
    object.CONST1 = AWB / 2.0 * 1.0e-19
    object.CONST2 = object.CONST1* 1.0e-02
    object.CONST3 = sqrt(0.2 * AWB) * 1.0e-9
    object.CONST4 = object.CONST3 * ALOSCH * 1.0e-15
    object.CONST5 = object.CONST3 / 2.0
    object.CORR = ABZERO * object.PressureTorr / (ATMOS * (ABZERO + object.TemperatureCentigrade) * 100.0)

    # Set long decorrelation length and step
    object.Decor_NCOLM = 2000000
    object.Decor_NCORLN = 500000
    object.Decor_NCORST = 2
    FRACM = 0.0
    MXEKR = 0

    # Set short decorrelation length and step for mixtures with more than 3% inelastic/molecular component
    for IH in range(object.NumberOfGases):
        if object.NumberOfGasesN[IH] != 2 and object.NumberOfGasesN[IH] != 6 and object.NumberOfGasesN[IH] != 7 and object.NumberOfGasesN[IH] != 3 and \
                object.NumberOfGasesN[IH] != 4 and object.NumberOfGasesN[IH] != 5:
            # Molecular gas sum total fraction
            FRACM += object.FRAC[IH]

    # If greater than 3% molecular/inelastic fraction, or large electric field use short decorrelation length.
    if object.EField > (10 / object.CORR) or FRACM>3:
            object.Decor_NCOLM = 400000
            object.Decor_NCORLN = 50000
            object.Decor_NCORST = 4
    TOTFRAC = 0.0
    if object.NumberOfGases == 0 or object.NumberOfGases > 6:
        raise ValueError("Error in Gas Input")

    for J in range(object.NumberOfGases):
        if object.NumberOfGasesN[J] == 0 or object.FRAC[J] == 0:
            raise ValueError("Error in Gas Input")
        TOTFRAC += object.FRAC[J]
    if abs(TOTFRAC - 100) >= 1e-6:
        raise ValueError("Error in Gas Input")
    NSCALE = 40000000
    object.MaxNumberOfCollisions = object.MaxNumberOfCollisions * NSCALE

    if object.MaxNumberOfCollisions < 0:
        raise ValueError("MaxNumberOfCollisions value is too large - overflow")
    object.NSTEP = 4000
    object.AngleFromZ = 0.785
    object.AngleFromX = 0.1

    object.VelocityX = 0.0
    object.VelocityY = 0.0
    object.VelocityZ = 0.0

    object.InitialElectronEnergy = object.FinalElectronEnergy / 50.0

    for i in range(6):
        object.ANN[i] = object.FRAC[i] * object.CORR * ALOSCH
    for i in range(6):
        object.VANN[i] = object.FRAC[i] * object.CORR * object.CONST4 * 1e15

    # Radians per picosecond
    object.AngularSpeedOfRotation = AWB * object.BFieldMag * 1e-12

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
    cdef double TWOPI,  ECHARG, EMASS, AMU, BOLTZ, BOLTZJ, AWB, ALOSCH, EOVM, ABZERO, ATMOS, TOTFRAC
    cdef long long MXEKR, IH, NSCALE, i

    TWOPI = 2.0 * np.pi
    object.ARY = <float>(13.60569253)
    object.PIR2 = 8.7973554297e-17
    ECHARG = 1.602176565e-19
    EMASS = 9.10938291e-31
    AMU = 1.660538921e-27
    BOLTZ = 8.6173324e-5
    BOLTZJ = 1.3806488e-23
    AWB = 1.758820088e10
    ALOSCH = 2.6867805e19
    EOVM = sqrt(2.0 * ECHARG / EMASS) * 100.0
    ABZERO = 273.15
    ATMOS = 760.0
    object.CONST1 = AWB / 2.0 * 1.0e-19
    object.CONST2 = object.CONST1 * 1.0e-02
    object.CONST3 = sqrt(0.2 * AWB) * 1.0e-9
    object.CONST4 = object.CONST3 * ALOSCH * 1.0e-15
    object.CONST5 = object.CONST3 / 2.0
    object.CORR = ABZERO * object.PressureTorr / (ATMOS * (ABZERO + object.TemperatureCentigrade) * 100.0)

    # Set long decorrelation length and step
    object.Decor_NCOLM = 2000000
    object.Decor_NCORLN = 500000
    object.Decor_NCORST = 2

    # Set short decorrelation length and step for mixtures with more than 3% inelastic/molecular component
    cdef double FRACM = 0.0
    MXEKR = 0
    for IH in range(object.NumberOfGases):
        if object.NumberOfGasesN[IH] != 2 and object.NumberOfGasesN[IH] != 6 and object.NumberOfGasesN[IH] != 7 and object.NumberOfGasesN[IH] != 3 and \
                object.NumberOfGasesN[IH] != 4 and object.NumberOfGasesN[IH] != 5:
            # Molecular gas sum total fraction
            FRACM += object.FRAC[IH]
    # If greater than 3% molecular/inelastic fraction, or large electric field use short decorrelation length.
    if (object.EField > (10.0 / object.CORR)) or (FRACM > 3):
            object.Decor_NCOLM = 400000
            object.Decor_NCORLN = 50000
            object.Decor_NCORST = 4
    TOTFRAC = 0.0

    if object.NumberOfGases == 0 or object.NumberOfGases > 6:
        raise ValueError("Error in Gas Input")
    for J in range(object.NumberOfGases):
        if object.NumberOfGasesN[J] == 0 or object.FRAC[J] == 0:
            raise ValueError("Error in Gas Input")
        TOTFRAC += object.FRAC[J]
    if abs(TOTFRAC - 100) >= 1e-6:
        raise ValueError("Error in Gas Input")


    object.NSCALE = 40000000
    object.MaxNumberOfCollisions = object.MaxNumberOfCollisions * object.NSCALE

    if object.MaxNumberOfCollisions < 0:
        raise ValueError("NMAX value is too large - overflow")
    object.NSTEP = 4000
    object.AngleFromZ = 0.785
    object.AngleFromX = 0.1
    object.InitialElectronEnergy = object.FinalElectronEnergy / 50.0
    object.CORR = ABZERO * object.PressureTorr / (ATMOS * (ABZERO + object.TemperatureCentigrade) * 100.0)

    object.ThermalEnergy = (ABZERO + object.TemperatureCentigrade) * BOLTZ
    for i in range(6):
        object.ANN[i] = object.FRAC[i] * object.CORR * ALOSCH
    object.AN = 100.0 * object.CORR * ALOSCH
    for i in range(6):
        object.VANN[i] = object.FRAC[i] * object.CORR * object.CONST4 * 1e15
    object.VAN = 100.0 * object.CORR * object.CONST4 * 1.0e15

    # Radians per picosecond
    object.AngularSpeedOfRotation = AWB * object.BFieldMag * 1e-12


    if object.BFieldMag == 0:
        return
    # Metres per picosecond
    object.EFieldOverBField = object.EField * 1e-9 / object.BFieldMag
    return
