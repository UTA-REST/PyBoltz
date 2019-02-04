import math

def SETUPT(Magboltz):
    Magboltz.API = math.acos(-1.0)
    TWOPI = 2.0 * Magboltz.API
    Magboltz.ARY = 13.60569253
    PIR2 = 8.7973554297e-17
    ECHARG = 1.602176565e-19
    EMASS = 9.10938291e-31
    AMU = 1.660538921e-27
    BOLTZ = 8.6173324e-5
    BOLTZJ = 1.3806488e-23
    AWB = 1.758820088e10
    ALOSCH = 2.6867805e19
    EOVM = math.sqrt(2.0 * ECHARG / EMASS) * 100.0
    ABZERO = 273.15
    ATMOS = 760.0
    Magboltz.CONST1 = AWB / 2.0 * 1.0e-19
    Magboltz.CONST2 = Magboltz.CONST1* 1.0e-02
    Magboltz.CONST3 = math.sqrt(0.2 * AWB) * 1.0e-9
    Magboltz.CONST4 = Magboltz.CONST3 * ALOSCH * 1.0e-15
    Magboltz.CONST5 = Magboltz.CONST3 / 2.0
    Magboltz.CORR = ABZERO * Magboltz.TORR / (ATMOS * (ABZERO + Magboltz.TEMPC) * 100.0)
    Magboltz.NANISO =2
    Magboltz.NCOLM = 400000
    Magboltz.NCORLN = 50000
    Magboltz.NCORST = 4
    MXEKR = 0
    for IH in range(Magboltz.NGAS):
        if Magboltz.NGASN[IH] == 2 or Magboltz.NGASN[IH] == 6 or Magboltz.NGASN[IH] == 7:
            MXEKR = IH
        if Magboltz.EFIELD > (10 / Magboltz.CORR):
            MXEKR = -1
    if MXEKR != -1:
        if Magboltz.NGAS == 1:
            Magboltz.NCOLM = 2000000
            Magboltz.NCORLN = 500000
            Magboltz.NCORST = 2
        elif Magboltz.FRAC[MXEKR] > 90.0:
            Magboltz.NCOLM = 2000000
            Magboltz.NCORLN = 500000
            Magboltz.NCORST = 2
    TOTFRAC = 0.0
    if Magboltz.NGAS == 0 or Magboltz.NGAS > 6:
        raise ValueError("Error in Gas Input")

    for J in range(Magboltz.NGAS):
        if Magboltz.NGASN[J] == 0 or Magboltz.FRAC[J] == 0:
            raise ValueError("Error in Gas Input")
        TOTFRAC += Magboltz.FRAC[J]
    if abs(TOTFRAC - 100) >= 1e-6:
        raise ValueError("Error in Gas Input")
    Magboltz.TMAX = 100.0
    NSCALE = 40000000
    Magboltz.NMAX = Magboltz.NMAX * NSCALE

    if Magboltz.NMAX < 0:
        raise ValueError("NMAX value is too large - overflow")
    Magboltz.NSTEP = 4000
    Magboltz.THETA = 0.785
    Magboltz.PHI = 0.1
    Magboltz.ESTART = Magboltz.EFINAL / 50.0
    Magboltz.RSTART=0.666
    Magboltz.CORR = ABZERO * Magboltz.TORR / (ATMOS * (ABZERO + Magboltz.TEMPC) * 100.0)

    Magboltz.AKT = (ABZERO + Magboltz.TEMPC) * BOLTZ
    Magboltz.ANN = [Magboltz.FRAC[i] * Magboltz.CORR * ALOSCH for i in range(6)]
    Magboltz.AN = 100.0 * Magboltz.CORR * ALOSCH
    Magboltz.VANN = [Magboltz.FRAC[i] * Magboltz.CORR * Magboltz.CONST4 * 1e15 for i in range(6)]
    Magboltz.VAN = 100.0 * Magboltz.CORR * Magboltz.CONST4 * 1.0e15

    Magboltz.WB = AWB * Magboltz.BMAG * 1e-12

    if Magboltz.BMAG == 0:
        return
    Magboltz.EOVB = Magboltz.EFIELD *1e-9/Magboltz.BMAG
    return
