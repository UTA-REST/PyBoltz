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
    object.API = acos(-1.0)
    TWOPI = 2.0 * object.API
    object.ARY = <float>(13.60569253)
    PIR2 = 8.7973554297e-17
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
    object.CORR = ABZERO * object.TORR / (ATMOS * (ABZERO + object.TEMPC) * 100.0)

    # Set long decorrelation length and step
    object.NCOLM = 2000000
    object.NCORLN = 500000
    object.NCORST = 2
    FRACM = 0.0
    MXEKR = 0

    # Set short decorrelation length and step for mixtures with more than 3% inelastic/molecular component
    for IH in range(object.NumberOfGases):
        if object.NumberOfGasesN[IH] != 2 and object.NumberOfGasesN[IH] != 6 and object.NumberOfGasesN[IH] != 7 and object.NumberOfGasesN[IH] != 3 and \
                object.NumberOfGasesN[IH] != 4 and object.NumberOfGasesN[IH] != 5:
            # Molecular gas sum total fraction
            FRACM += object.FRAC[IH]

    # If greater than 3% molecular/inelastic fraction, or large electric field use short decorrelation length.
    if object.EFIELD > (10 / object.CORR) or FRACM>3:
            object.NCOLM = 400000
            object.NCORLN = 50000
            object.NCORST = 4
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
    object.NMAX = object.NMAX * NSCALE

    if object.NMAX < 0:
        raise ValueError("NMAX value is too large - overflow")
    object.NSTEP = 4000
    object.THETA = 0.785
    object.PHI = 0.1

    object.WX = 0.0
    object.WY = 0.0
    object.WZ = 0.0

    object.ESTART = object.EFINAL / 50.0

    for i in range(6):
        object.ANN[i] = object.FRAC[i] * object.CORR * ALOSCH
    object.AN = 100.0 * object.CORR * ALOSCH
    for i in range(6):
        object.VANN[i] = object.FRAC[i] * object.CORR * object.CONST4 * 1e15
    object.VAN = 100.0 * object.CORR * object.CONST4 * 1.0e15

    # Radians per picosecond
    object.WB = AWB * object.BFieldMag * 1e-12

    if object.BFieldMag == 0:
        return

    # Metres per picosecond
    object.EOVB = object.EFIELD * 1e-9 / object.BFieldMag
    return


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef SetupT(PyBoltz object):
    """
    This function sets up the given PyBoltz object. It fills the values of the main constants. 
    
    The object parameter is the PyBoltz object to be setup.
    """
    cdef double TWOPI, PIR2, ECHARG, EMASS, AMU, BOLTZ, BOLTZJ, AWB, ALOSCH, EOVM, ABZERO, ATMOS, TOTFRAC
    cdef long long MXEKR, IH, NSCALE, i

    object.API = acos(-1.0)
    TWOPI = 2.0 * object.API
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
    object.CORR = ABZERO * object.TORR / (ATMOS * (ABZERO + object.TEMPC) * 100.0)

    # Set long decorrelation length and step
    object.NCOLM = 2000000
    object.NCORLN = 500000
    object.NCORST = 2

    # Set short decorrelation length and step for mixtures with more than 3% inelastic/molecular component
    cdef double FRACM = 0.0
    MXEKR = 0
    for IH in range(object.NumberOfGases):
        if object.NumberOfGasesN[IH] != 2 and object.NumberOfGasesN[IH] != 6 and object.NumberOfGasesN[IH] != 7 and object.NumberOfGasesN[IH] != 3 and \
                object.NumberOfGasesN[IH] != 4 and object.NumberOfGasesN[IH] != 5:
            # Molecular gas sum total fraction
            FRACM += object.FRAC[IH]
    # If greater than 3% molecular/inelastic fraction, or large electric field use short decorrelation length.
    if (object.EFIELD > (10.0 / object.CORR)) or (FRACM>3):
            object.NCOLM = 400000
            object.NCORLN = 50000
            object.NCORST = 4
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
    object.NMAX = object.NMAX * object.NSCALE

    if object.NMAX < 0:
        raise ValueError("NMAX value is too large - overflow")
    object.NSTEP = 4000
    object.THETA = 0.785
    object.PHI = 0.1
    object.ESTART = object.EFINAL / 50.0
    object.CORR = ABZERO * object.TORR / (ATMOS * (ABZERO + object.TEMPC) * 100.0)

    object.AKT = (ABZERO + object.TEMPC) * BOLTZ
    for i in range(6):
        object.ANN[i] = object.FRAC[i] * object.CORR * ALOSCH
    object.AN = 100.0 * object.CORR * ALOSCH
    for i in range(6):
        object.VANN[i] = object.FRAC[i] * object.CORR * object.CONST4 * 1e15
    object.VAN = 100.0 * object.CORR * object.CONST4 * 1.0e15

    # Radians per picosecond
    object.WB = AWB * object.BFieldMag * 1e-12


    if object.BFieldMag == 0:
        return
    # Metres per picosecond
    object.EOVB = object.EFIELD * 1e-9 / object.BFieldMag
    return
