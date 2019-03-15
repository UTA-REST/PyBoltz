from SETUPT import SETUPT
import math
from MIXERT import MIXERT
from libc.math cimport sin, cos, acos, asin, log, sqrt, pow
import numpy as np
cimport numpy as np
from ELIMITT import ELIMITT
import numpy as np
from MONTET import MONTET
from libc.string cimport memset
from ALPCALCT import ALPCALCT
cdef extern from "C/RM48.h":
    double DRAND48(double dummy)
    void RM48(double lenv)

cdef double drand48(double dummy):
    return DRAND48(dummy)

cdef class Magboltz:
    def __init__(self):
        self.EOVB = 0.0
        self.WB = 0.0
        self.PIR2 = 0.0
        self.BTHETA = 0.0
        self.BMAG = 0.0
        self.NGAS = 0
        self.NSTEP = 0
        self.NANISO = 0
        self.EFINAL = 0.0
        self.ESTEP = 0
        self.AKT = 0.0
        self.ARY = 0.0
        self.TEMPC = 0.0
        self.TORR = 0.0
        self.IPEN = 0
        self.NSCALE = 0
        self.TMAX = 0.0
        self.SMALL = 0.0
        self.API = 0.0
        memset(self.NESST, 0, 9 * sizeof(double))

        self.ESTART = 0.0
        self.THETA = 0.0
        self.PHI = 0.0
        self.EFIELD = 0.0
        memset(self.DENSY, 0, 4000 * sizeof(double))
        memset(self.SPEC, 0, 4000 * sizeof(double))
        memset(self.TIME, 0, 300 * sizeof(double))
        memset(self.ICOLL, 0, 6 * 5 * sizeof(double))
        memset(self.ICOLNN, 0, 6 * 10 * sizeof(double))
        memset(self.ICOLN, 0, 6 * 290 * sizeof(double))

        self.NMAX = 0.0
        self.ALPHA = 0.0
        memset(self.AMGAS, 0, 6 * sizeof(double))
        memset(self.VTMB, 0, 6 * sizeof(double))
        self.TCFMX = 0.0
        memset(self.TCFMXG, 0, 6 * sizeof(double))
        self.ITHRM = 0.0
        self.ITMAX = 0
        memset(self.NGASN, 0, 6 * sizeof(double))
        self.CORR = 0.0
        memset(self.FRAC, 0, 6 * sizeof(double))
        # common output blocks
        self.WX = 0.0
        self.WY = 0.0
        self.WZ = 0.0
        self.DWX = 0.0
        self.DWY = 0.0
        self.DWZ = 0.0
        self.TTOTS = 0.0
        self.ATT = 0.0
        self.ALPER = 0.0
        self.ATTER = 0.0
        self.DIFLN = 0.0
        self.DIFTR = 0.0
        self.DIFXX = 0.0
        self.DIFYY = 0.0
        self.DIFZZ = 0.0
        self.DIFYZ = 0.0
        self.DIFXY = 0.0
        self.DIFXZ = 0.0
        self.DXXER = 0.0
        self.DYYER = 0.0
        self.DZZER = 0.0
        self.DYZER = 0.0
        self.DXYER = 0.0
        self.DXZER = 0.0
        self.IFAKE = 0
        self.FAKEI = 0.0
        self.RSTART = 0.666
        memset(self.ANN, 0, 6 * sizeof(double))
        memset(self.VANN, 0, 6 * sizeof(double))
        memset(self.RI, 0, 8 * sizeof(double))
        memset(self.EPT, 0, 8 * sizeof(double))
        memset(self.VZPT, 0, 8 * sizeof(double))
        memset(self.TTEST, 0, 8 * sizeof(double))
        self.AN = 0.0
        self.VAN = 0.0
        memset(self.QELM, 0, 4000 * sizeof(double))
        memset(self.QSUM, 0, 4000 * sizeof(double))

        memset(self.QION, 0, 6 * 4000 * sizeof(double))

        memset(self.QIN, 0, 6 * 250 * 4000 * sizeof(double))
        memset(self.E, 0, 4000 * sizeof(double))
        memset(self.EROOT, 0, 4000 * sizeof(double))
        memset(self.QTOT, 0, 4000 * sizeof(double))
        memset(self.QREL, 0, 4000 * sizeof(double))
        memset(self.QINEL, 0, 4000 * sizeof(double))
        memset(self.NIN, 0, 6 * sizeof(double))
        memset(self.LION, 0, 6 * sizeof(double))

        memset(self.LIN, 0, 6 * 250 * sizeof(double))

        memset(self.ALION, 0, 6 * sizeof(double))

        memset(self.ALIN, 0, 6 * 250 * sizeof(double))

        memset(self.CF, 0, 6 * 290 * 4000 * sizeof(double))

        memset(self.TCF, 0, 6 * 4000 * sizeof(double))

        memset(self.EIN, 0, 6 * 290 * sizeof(double))

        memset(self.IARRY, 0, 6 * 290 * sizeof(double))

        memset(self.RGAS, 0, 6 * 290 * sizeof(double))

        memset(self.IPN, 0, 6 * 290 * sizeof(double))

        memset(self.WPL, 0, 6 * 290 * sizeof(double))

        memset(self.IPLAST, 0, 6 * sizeof(double))
        memset(self.ISIZE, 0, 6 * sizeof(double))
        memset(self.PENFRA, 0, 6 * 290 * 3 * sizeof(double))

        memset(self.TCFMAX, 0, 6 * sizeof(double))

        memset(self.CFN, 0, 6 * 10 * 4000 * sizeof(double))

        memset(self.TCFN, 0, 6 * 4000 * sizeof(double))

        memset(self.SCLENUL, 0, 6 * 10 * sizeof(double))

        memset(self.NPLAST, 0, 6 * sizeof(double))

        memset(self.PSCT, 0, 6 * 290 * 4000 * sizeof(double))

        memset(self.ANGCT, 0, 6 * 290 * 4000 * sizeof(double))

        memset(self.INDEX, 0, 6 * 290 * sizeof(double))

        self.NISO = 0
        memset(self.FCION, 0, 4000 * sizeof(double))
        memset(self.FCATT, 0, 4000 * sizeof(double))

        memset(self.NC0, 0, 6 * 290 * sizeof(double))

        memset(self.EC0, 0, 6 * 290 * sizeof(double))

        memset(self.NG1, 0, 6 * 290 * sizeof(double))

        memset(self.EG1, 0, 6 * 290 * sizeof(double))

        memset(self.NG2, 0, 6 * 290 * sizeof(double))

        memset(self.EG2, 0, 6 * 290 * sizeof(double))

        memset(self.WKLM, 0, 6 * 290 * sizeof(double))

        memset(self.EFL, 0, 6 * 290 * sizeof(double))

        memset(self.XSS, 0, 2000 * sizeof(double))
        memset(self.YSS, 0, 2000 * sizeof(double))
        memset(self.ZSS, 0, 2000 * sizeof(double))
        memset(self.TSS, 0, 2000 * sizeof(double))
        memset(self.ESS, 0, 2000 * sizeof(double))
        memset(self.DCXS, 0, 2000 * sizeof(double))
        memset(self.DCYS, 0, 2000 * sizeof(double))
        memset(self.DCZS, 0, 2000 * sizeof(double))
        memset(self.IPLS, 0, 2000 * sizeof(double))
        memset(self.IFAKET, 0, 8 * sizeof(double))
        memset(self.IFAKED, 0, 9 * sizeof(double))
        memset(self.QEL, 0, 4000 * sizeof(double))
        memset(self.QSATT, 0, 4000 * sizeof(double))
        memset(self.RNMX, 0, 6 * sizeof(double))
        memset(self.ES, 0, 4000 * sizeof(double))
        self.ZTOT = 0.0
        self.ZTOTS = 0.0
        self.CONST1 = 0.0
        self.CONST2 = 0.0
        self.CONST3 = 0.0
        self.CONST4 = 0.0
        self.CONST5 = 0.0
        memset(self.LAST, 0, 6 * sizeof(double))
        self.IELOW = 1
        self.NCOLM = 0
        self.NCORLN = 0
        self.NCORST = 0
        self.NNULL = 0
        self.TMAX1 = 0.0
        self.DEN = 0.0
        self.AVE = 0.0
        self.XID = 0.0
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.DFLER = 0.0
        self.DFTER = 0.0
        self.TGAS = 0.0
        self.ALPP = 0.0
        self.ATTP = 0.0
        self.SSTMIN = 0.0
        self.VDOUT = 0.0
        self.VDERR = 0.0
        self.WSOUT = 0.0
        self.WSERR = 0.0
        self.DLOUT = 0.0
        self.DLERR = 0.0
        self.NMAXOLD = 0.0
        self.DTOUT = 0.0
        self.DTERR = 0.0
        self.ALPHSST = 0.0
        memset(self.ETPL, 0, 8 * sizeof(double))
        memset(self.XTPL, 0, 8 * sizeof(double))
        memset(self.YTPL, 0, 8 * sizeof(double))
        memset(self.ZTPL, 0, 8 * sizeof(double))
        memset(self.YZTPL, 0, 8 * sizeof(double))
        memset(self.XZTPL, 0, 8 * sizeof(double))
        memset(self.XYTPL, 0, 8 * sizeof(double))
        memset(self.VYTPL, 0, 8 * sizeof(double))
        memset(self.VXTPL, 0, 8 * sizeof(double))
        memset(self.TTPL, 0, 8 * sizeof(double))
        memset(self.XXTPL, 0, 8 * sizeof(double))
        memset(self.YYTPL, 0, 8 * sizeof(double))
        memset(self.ZZTPL, 0, 8 * sizeof(double))
        memset(self.VZTPL, 0, 8 * sizeof(double))
        memset(self.NETPL, 0, 8 * sizeof(double))
        self.ATTOINT = 0.0
        self.ATTERT = 0.0
        self.AIOERT = 0.0
        self.ALPHERR = 0.0
        self.ATTSST = 0.0
        self.TTOT = 0.0
        self.ATTERR = 0.0
        memset(self.ZPLANE, 0, 8 * sizeof(double))
        self.IZFINAL = 0.0
        self.RALPHA = 0.0
        self.RALPER = 0.0
        self.TODENE = 0.0
        self.TOFENER = 0.0
        self.TOFENE = 0.0
        self.TOFWV = 0.0
        self.TOFWVER = 0.0
        self.TOFDL = 0.0
        self.TOFDLER = 0.0
        self.TOFDT = 0.0
        self.TOFDTER = 0.0
        self.TOFWR = 0.0
        self.TOFWRER = 0.0
        self.RATTOF = 0.0
        self.RATOFER = 0.0
        self.ALPHAST = 0.0
        self.VDST = 0.0
        self.TSTEP = 0.0
        self.ZSTEP = 0.0
        self.TFINAL = 0.0
        self.RATTOFER = 0.0
        self.ZFINAL = 0.0
        self.ITFINAL = 0.0
        self.IPRIM = 0.0
        memset(self.XS, 0, 2000 * sizeof(double))
        memset(self.YS, 0, 2000 * sizeof(double))
        memset(self.ZS, 0, 2000 * sizeof(double))
        memset(self.TS, 0, 2000 * sizeof(double))
        memset(self.DCX, 0, 2000 * sizeof(double))
        memset(self.DCY, 0, 2000 * sizeof(double))
        memset(self.DCZ, 0, 2000 * sizeof(double))
        memset(self.IPL, 0, 2000 * sizeof(double))
        self.ST = 0.0
        memset(self.ESPL, 0, 8 * sizeof(double))
        memset(self.XSPL, 0, 8 * sizeof(double))
        memset(self.TMSPL, 0, 8 * sizeof(double))
        memset(self.TTMSPL, 0, 8 * sizeof(double))
        memset(self.RSPL, 0, 8 * sizeof(double))
        memset(self.RRSPL, 0, 8 * sizeof(double))
        memset(self.RRSPM, 0, 8 * sizeof(double))
        memset(self.YSPL, 0, 8 * sizeof(double))
        memset(self.ZSPL, 0, 8 * sizeof(double))
        memset(self.TSPL, 0, 8 * sizeof(double))
        memset(self.XXSPL, 0, 8 * sizeof(double))
        memset(self.YYSPL, 0, 8 * sizeof(double))
        memset(self.ZZSPL, 0, 8 * sizeof(double))
        memset(self.VZSPL, 0, 8 * sizeof(double))
        memset(self.TSSUM, 0, 8 * sizeof(double))
        memset(self.TSSUM2, 0, 8 * sizeof(double))
        self.TOFWVZ = 0.0
        self.TOFWVZER = 0.0
        self.TOFWVX = 0.0
        self.TOFWVXER = 0.0
        self.TOFWVY = 0.0
        self.TOFWVYER = 0.0
        self.TOFDZZ = 0.0
        self.TOFDZZER = 0.0
        self.TOFDXX = 0.0
        self.TOFDXXER = 0.0
        self.TOFDYY = 0.0
        self.TOFDYYER = 0.0
        self.TOFDYZ = 0.0
        self.TOFDYZER = 0.0
        self.TOFDXZ = 0.0
        self.TOFDXZER = 0.0
        self.TOFDXY = 0.0
        self.TOFDXYER = 0.0
        self.TOFWRZ = 0.0
        self.TOFWRZER = 0.0
        self.TOFWRY = 0.0
        self.TOFWRYER = 0.0
        self.TOFWRX = 0.0
        self.TOFWRXER = 0.0
        self.ATTOION = 0.0
        self.ATTIOER = 0.0
        self.ATTATER = 0.0
        self.EMTX = [0.00, .001, .005, .007, 0.01, .015, 0.02, .025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12,
                     0.14, 0.17,
                     0.20, 0.25, 0.27, 0.30, 0.32, 0.35, 0.37, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.51, 0.52, 0.53,
                     0.54, 0.55,
                     0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, .715,
                     0.73, 0.75,
                     0.77, 0.80, 0.83, 0.85, 0.87, 0.90, 1.00, 1.08, 1.14, 1.20, 1.30, 1.40, 1.50, 1.70, 2.00, 2.50,
                     3.00, 3.50,
                     4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00, 9.00, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0,
                     40.0, 50.0,
                     60.0, 70.0, 80.0, 90.0, 100., 125., 150., 200., 250., 300., 400., 500., 600., 700., 800., 1000.,
                     1500., 2000.,
                     3000., 4000., 5000., 6000., 8000., 1.0e4, 1.5e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4, 6.0e4, 8.0e4, 1.0e5,
                     1.25e5,
                     1.5e5, 1.75e5, 2.0e5, 2.5e5, 3.0e5, 3.5e5, 4.0e5, 4.5e5, 5.0e5, 6.0e5, 7.0e5, 8.0e5, 9.0e5, 1.0e6,
                     1.25e6,
                     1.5e6, 1.75e6, 2.0e6, 2.5e6, 3.0e6, 3.5e6, 4.0e6, 4.5e6, 5.0e6, 6.0e6, 7.0e6, 8.0e6, 9.0e6, 1.0e7,
                     1.25e7,
                     1.5e7, 1.75e7, 2.0e7, 2.5e7, 3.0e7, 3.5e7, 4.0e7, 4.5e7, 5.0e7, 6.0e7, 7.0e7, 8.0e7, 9.0e7, 1.0e8,
                     1.25e8,
                     1.5e8, 1.75e8, 2.0e8, 2.5e8, 3.0e8, 3.5e8, 4.0e8, 4.5e8, 5.0e8, 6.0e8, 7.0e8, 8.0e8, 9.0e8, 1.0e9]
        self.ETX = [0.00, .001, .005, .007, 0.01, .015, 0.02, .025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12,
                    0.14, 0.17,
                    0.20, 0.25, 0.27, 0.30, 0.32, 0.35, 0.37, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.51, 0.52, 0.53,
                    0.54, 0.55,
                    0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.75,
                    0.80, 0.85,
                    0.90, 1.00, 1.20, 1.50, 1.75, 2.00, 2.50, 2.75, 3.00, 3.75, 4.00, 4.50, 5.00, 5.50, 6.00, 6.50,
                    7.00, 8.00,
                    9.00, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100., 125.,
                    150., 200.,
                    250., 300., 400., 500., 600., 700., 800., 1000., 1500., 2000., 3000., 4000., 5000., 6000., 8000.,
                    1.0e4, 1.5e4,
                    2.0e4, 3.0e4, 4.0e4, 5.0e4, 6.0e4, 8.0e4, 1.0e5, 1.25e5, 1.5e5, 1.75e5, 2.0e5, 2.5e5, 3.0e5, 3.5e5,
                    4.0e5,
                    4.5e5, 5.0e5, 6.0e5, 7.0e5, 8.0e5, 9.0e5, 1.0e6, 1.25e6, 1.5e6, 1.75e6, 2.0e6, 2.5e6, 3.0e6, 3.5e6,
                    4.0e6,
                    4.5e6, 5.0e6, 6.0e6, 7.0e6, 8.0e6, 9.0e6, 1.0e7, 1.25e7, 1.5e7, 1.75e7, 2.0e7, 2.5e7, 3.0e7, 3.5e7,
                    7.0e7,
                    1.e9]
        self.EATX = [0.00, .001, .005, .007, .010, .015, .020, .025, .030, .040, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12,
                     0.14, 0.17,
                     0.20, 0.25, 0.27, 0.30, 0.32, 0.35, 0.37, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.51, 0.52, 0.53,
                     0.54, 0.55,
                     0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71,
                     0.72, 0.73,
                     0.75, 0.77, 0.80, 0.83, 0.85, 0.87, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.70, 2.00, 2.50,
                     3.00, 3.50,
                     4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00, 9.00, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0,
                     40.0, 50.0,
                     60.0, 70.0, 80.0, 90.0, 100., 125., 150., 200., 250., 300., 400., 500., 600., 700., 800., 1000.,
                     1500., 2000.,
                     3000., 4000., 5000., 6000., 8000., 10000., 15000., 2.0e4, 3.0e4, 4.0e4, 5.0e4, 6.0e4, 8.0e4, 1.0e5,
                     1.25e5,
                     1.5e5, 1.75e5, 2.0e5, 2.5e5, 3.0e5, 3.5e5, 4.0e5, 4.5e5, 5.0e5, 6.0e5, 7.0e5, 8.0e5, 9.0e5, 1.0e6,
                     1.25e6,
                     1.5e6, 1.75e6, 2.0e6, 2.5e6, 3.0e6, 3.5e6, 4.0e6, 4.5e6, 5.0e6, 6.0e6, 7.0e6, 8.0e6, 9.0e6, 1.0e7,
                     1.25e7,
                     1.5e7, 1.75e7, 2.0e7, 2.5e7, 3.0e7, 3.5e7, 4.0e7, 4.5e7, 5.0e7, 6.0e7, 7.0e7, 8.0e7, 9.0e7, 1.0e8,
                     1.25e8,
                     1.5e8, 1.75e8, 2.0e8, 2.5e8, 3.0e8, 3.5e8, 4.0e8, 4.5e8, 5.0e8, 6.0e8, 7.0e8, 8.0e8, 9.0e8, 1.0e9]

        #default EMT
        self.EMTY = [131., 115., 97.0, 91.1, 83.9, 74.6, 67.3, 61.2, 56.1, 47.9, 41.4, 36.2, 31.8, 28.2, 22.5, 18.1,
                     14.8, 11.1,
                     8.36, 5.33, 4.47, 3.43, 2.88, 2.22, 1.86, 1.43, 1.20, 1.01, .844, .708, .596, .548, .504, .465,
                     .430, .399,
                     .372, .348, .328, .310, .296, .285, .276, .270, .266, .265, .266, .270, .276, .287, .306, .341,
                     .377, .427,
                     .479, .562, .651, .713, .778, .880, 1.26, 1.62, 1.92, 2.25, 2.85, 3.51, 4.22, 5.73, 7.97, 11.8,
                     15.8, 20.4,
                     24.4, 28.0, 30.7, 31.5, 32.3, 31.6, 31.0, 27.8, 23.5, 19.8, 15.0, 10.9, 8.40, 7.25, 5.65, 5.00,
                     4.50, 3.10,
                     2.42, 2.17, 2.00, 1.89, 1.80, 1.73, 1.65, 1.50, 1.39, 1.26, 1.09, 0.94, 0.84, 0.75, 0.68, 0.56,
                     0.38, 0.26,
                     .155, .105, .076, .059, .038, .027, .0148, .0094, .0050, .0031, .0022, .00163, .001024, .000714,
                     .000498,
                     .000372, .000291, .000236, .000166, .000125, 9.90e-5, 8.08e-5, 6.76e-5, 5.77e-5, 4.38e-5, 3.48e-5,
                     2.85e-5,
                     2.39e-5, 2.04e-5, 1.43e-5, 1.08e-5, 8.52e-6, 6.91e-6, 4.85e-6, 3.62e-6, 2.81e-6, 2.25e-6, 1.85e-6,
                     1.55e-6,
                     1.13e-6, 8.67e-7, 6.86e-7, 5.58e-7, 4.63e-7, 3.10e-7, 2.23e-7, 1.68e-7, 1.31e-7, 8.64e-8, 6.11e-8,
                     4.54e-8,
                     3.51e-8, 2.78e-8, 2.26e-8, 1.57e-8, 1.15e-8, 8.79e-9, 6.93e-9, 5.60e-9, 3.57e-9, 2.47e-9, 1.81e-9,
                     1.38e-9,
                     8.82e-10, 6.11e-10, 4.48e-10, 3.43e-10, 2.71e-10, 2.19e-10, 1.52e-10, 1.12e-10, 8.55e-11, 6.75e-11,
                     5.47e-11]
        # default ELASTIC TOTAL
        self.ETY = [131., 117., 101., 95.4, 88.8, 80.1, 73.3, 67.5, 62.6, 54.7, 48.4, 43.2, 38.8, 35.2, 29.4, 24.7,
                    21.2, 17.1,
                    14.0, 10.3, 9.10, 7.75, 6.94, 5.95, 5.40, 4.50, 4.25, 3.95, 3.65, 3.45, 3.20, 3.11, 3.00, 2.90,
                    2.79, 2.69,
                    2.59, 2.48, 2.37, 2.25, 2.14, 2.02, 1.92, 1.80, 1.69, 1.58, 1.48, 1.40, 1.32, 1.28, 1.26, 1.24,
                    1.30, 1.45,
                    1.50, 1.87, 2.80, 4.76, 6.68, 8.85, 13.7, 16.3, 18.7, 24.5, 29.0, 32.7, 36.8, 39.3, 41.7, 41.7,
                    41.8, 41.8,
                    41.0, 40.0, 37.4, 34.2, 32.4, 30.8, 21.9, 14.1, 8.58, 6.78, 5.97, 5.49, 5.29, 5.21, 5.10, 4.66,
                    4.58, 4.67,
                    4.53, 4.35, 4.12, 3.77, 3.58, 3.30, 3.12, 2.80, 2.36, 2.07, 1.72, 1.52, 1.34, 1.13, .937, .817,
                    .632, .523,
                    .397, .326, .279, .246, .203, .175, .152, .136, .124, .116, .103, .0946, .0886, .0841, .0807, .0779,
                    .0739,
                    .0711, .0690, .0674, .0662, .0640, .0627, .0618, .0612, .0604, .0599, .0596, .0594, .0593, .0592,
                    .0590, .0589,
                    .0589, .0588, .0588, .0587, .0587, .0587, .0587, .0587, .05867, .05866, .05865, .05865]
        # default ELASTIC ANGULAR DISTRUBUTION
        self.EATY = [1., .9744, .9406, .9325, .9173, .8972, .8776, .8606, .845, .8148, .7851, .7598, .7333, .7069,
                     .6565, .6117,
                     .5651, .5015, .4373, .3459, .3177, .2684, .2421, .2044, .1803, .1590, .1326, .1141, .0981, .0822,
                     .0714,
                     .0659, .0615, .0575, .0544, .0515, .0492, .0476, .0467, .0464, .0467, .0480, .0493, .0523, .0560,
                     .0614,
                     .0678, .0751, .0845, .0937, .1056, .1183, .1321, .1471, .1802, .2095, .2585, .2944, .3183, .3588,
                     .4247,
                     .5332, .6259, .7104, .7648, .8047, .8308, .8717, .8515, .7938, .7699, .7682, .7647, .7864, .7544,
                     .7075,
                     .6695, .6461, .6238, .5219, .4088, .3217, .2293, .1597, .1165, .1011, .1156, .1887, .3535, .2829,
                     .2332,
                     .2240, .2087, .1956, .1873, .2028, .1934, .1617, .1506, .1379, .1201, .1098, .1003, .0956, .0899,
                     .0792,
                     .0579, .0408, .0259, .0182, .0140, .0126, .0091, .0070, .0046, .00331, .00215, .00154, .00123,
                     .00100,
                     7.30e-4, 5.69e-4, 4.42e-4, 3.59e-4, 3.00e-4, 2.56e-4, 1.96e-4, 1.57e-4, 1.29e-4, 1.09e-4, 9.34e-5,
                     8.12e-5,
                     6.34e-5, 5.12e-5, 4.23e-5, 3.563e-5, 3.048e-5, 2.134e-5, 1.607e-5, 1.255e-5, 1.009e-5, 6.944e-6,
                     5.077e-6,
                     3.875e-6, 3.056e-6, 2.473e-6, 2.043e-6, 1.461e-6, 1.097e-6, 8.531e-7, 6.826e-7, 5.583e-7, 3.635e-7,
                     2.550e-7,
                     1.885e-7, 1.447e-7, 9.266e-8, 6.407e-8, 4.674e-8, 3.548e-8, 2.775e-8, 2.224e-8, 1.512e-8, 1.089e-8,
                     8.19e-9,
                     6.36e-9, 5.08e-9, 3.16e-8, 2.14e-9, 1.54e-9, 1.163e-9, 7.25e-10, 4.93e-10, 3.56e-10, 2.69e-10,
                     2.10e-10,
                     1.68e-10, 1.15e-10, 8.3e-11, 6.3e-11, 4.9e-11, 3.9e-11]

        self.A = 0.0
        self.D = 0.0
        self.F = 0.0
        self.A1 = 0.0
        self.Lambda = 0.0
        self.EV0 = 0.0
        self.DTOVMB = 0.0
        self.DTMN = 0.0
        self.DFTER1 = 0.0
        self.DLOVMB = 0.0
        self.DLMN = 0.0
        self.DFLER1 = 0.0
    def end(self):
        if self.WZ != 0:
            self.DTOVMB = self.DIFTR * self.EFIELD / self.WZ
            self.DTMN = sqrt(2.0 * self.DIFTR / self.WZ) * 10000.0
            self.DFTER1 = math.sqrt(self.DFTER ** 2 + self.DWZ ** 2)
            self.DFTER1 = self.DFTER1 / 2.0

            self.DLOVMB = self.DIFLN * self.EFIELD / self.WZ
            self.DLMN = sqrt(2.0 * self.DIFLN / self.WZ) * 10000.0
            self.DFLER1 = sqrt(self.DFLER ** 2 + self.DWZ ** 2)
            self.DFLER1 = self.DFLER1 / 2.0

    def MERT(self, epsilon, A, D, F, A1):
        a0 = 1  # 5.29e-11  # in m
        hbar = 1  # 197.32697*1e-9 # in eV m
        m = 1  # 511e3     # eV/c**2
        alpha = 27.292 * a0 ** 3
        k = np.sqrt((epsilon) / (13.605 * a0 ** 2))

        eta0 = -A * k * (1 + (4 * alpha) / (3 * a0) * k ** 2 * np.log(k * a0)) \
               - (np.pi * alpha) / (3 * a0) * k ** 2 + D * k ** 3 + F * k ** 4

        eta1 = (np.pi) / (15 * a0) * alpha * k ** 2 - A1 * k ** 3

        Qm = (4 * np.pi * a0 ** 2) / (k ** 2) * (np.sin(np.arctan(eta0) - np.arctan(eta1))) ** 2

        Qt = (4 * np.pi * a0 ** 2) / (k ** 2) * (np.sin(np.arctan(eta0))) ** 2

        return Qm * (5.29e-11) ** 2 * 1e20, Qt * (5.29e-11) ** 2 * 1e20

    def WEIGHT_Q(self, eV, Qm, BashBoltzQm, Lamda, eV0):
        WeightQm = (1 - np.tanh(Lamda * (eV - eV0))) / 2
        WeightBB = (1 + np.tanh(Lamda * (eV - eV0))) / 2

        NewBashQm = BashBoltzQm * WeightBB
        NewMERTQm = Qm * WeightQm
        NewQm = NewBashQm + NewMERTQm
        return NewQm

    def HYBRID_X_SECTIONS(self, MB_EMTx, MB_EMTy, MB_ETx, MB_ETy, A, D, F, A1, Lambda, eV0):
        Qm_MERT, Qt_MERT = self.MERT(MB_EMTx, A, D, F, A1)
        New_Qm = self.WEIGHT_Q(MB_EMTx, Qm_MERT, MB_EMTy, Lambda, eV0)
        Qm_MERT, Qt_MERT = self.MERT(MB_ETx, A, D, F, A1)
        New_Qt = self.WEIGHT_Q(MB_ETx, Qt_MERT, MB_ETy, Lambda, eV0)

        return MB_EMTx, New_Qm, MB_ETx, New_Qt
    def Start(self):
        cdef double EOB
        cdef int i = 0
        cdef double firstnotnanEMTY;
        cdef double temp[182]
        memset(temp, 0, 182 * sizeof(double))

        if self.A != 0 and self.F != 0 and self.D != 0 and self.A1 != 0 and self.Lambda != 0 and self.EV0 != 0:
            for i in range(182):
                self.EMTX[i], self.EMTY[i], self.EATX[i], temp[i] = self.HYBRID_X_SECTIONS(self.EMTX[i],
                                                                                                self.EMTY[i],
                                                                                                self.EATX[i],
                                                                                                temp[i], self.A,
                                                                                                self.D, self.F, self.A1,
                                                                                                self.Lambda, self.EV0)

            for i in range(153):
                self.EMTX[i], temp[i], self.ETX[i], self.ETY[i] = self.HYBRID_X_SECTIONS(self.EMTX[i],
                                                                                temp[i],
                                                                                self.ETX[i],
                                                                                self.ETY[i], self.A,
                                                                                self.D, self.F, self.A1,
                                                                                self.Lambda, self.EV0)
            # getting rid of nan
            self.EMTY[0]=131
            self.ETY[0]=131


        if self.ITHRM != 0:
            SETUPT(self)
            if self.EFINAL == 0.0:
                self.EFINAL = 0.5
                EOB = self.EFIELD * (self.TEMPC + 273.15) / (self.TORR * 293.15)
                if EOB > 15:
                    self.EFINAL = 8.0
                self.ESTART = self.EFINAL / 50.0
                while self.IELOW == 1:
                    print("MIXERT")
                    MIXERT(self)
                    if self.BMAG == 0 or self.BTHETA == 0 or abs(self.BTHETA) == 180:
                        print("ELIMITT")
                        ELIMITT(self)
                    elif self.BTHETA == 90:
                        print("")
                    else:
                        print("")
                    if self.IELOW == 1:
                        self.EFINAL = self.EFINAL * math.sqrt(2)
                        print("Calculated the final energy = "+str(self.EFINAL))
                        self.ESTART = self.EFINAL / 50
                print("Calculated the final energy = "+str(self.EFINAL))
            else:
                MIXERT(self)
            if self.BMAG == 0:
                print("MONTE")
                MONTET(self)
            else:
                if self.BTHETA == 0 or self.BTHETA == 180:
                    print("")
                elif self.BTHETA == 90:
                    print("")
                else:
                    print("")
            self.TGAS = 273.15 + self.TEMPC
            self.ALPP = self.ALPHA * 760 * self.TGAS / (self.TORR * 293.15)
            self.ATTP = self.ATT * 760 * self.TGAS / (self.TORR * 293.15)
            self.SSTMIN = 30

            if abs(self.ALPP - self.ATTP) < self.SSTMIN:
                self.end()
                return
            if self.BMAG == 0.0:
                print("ALP")
                ALPCALCT(self)
            elif self.BTHETA == 0.0 or self.BTHETA == 180:
                print("")
            elif self.BTHETA == 90:
                print("")
            else:
                print("")
            self.end()
            return
        else:
            print("")
            if self.EFINAL == 0.0:
                self.EFINAL = 0.5
                EOB = self.EFIELD * (self.TEMPC + 273.15) / (self.TORR * 293.15)
                if EOB > 15:
                    self.EFINAL = 8
                self.ESTART = self.EFINAL / 50
                while self.IELOW == 1:
                    print("")
                    if self.BMAG == 0 or self.BTHETA == 0 or abs(self.BTHETA) == 180:
                        print("")
                    elif self.BTHETA == 90:
                        print("")
                    else:
                        print("")
                    if self.IELOW == 1:
                        self.EFINAL = self.EFINAL * math.sqrt(2)
                        self.ESTART = self.EFINAL / 50
            else:
                print("")

            if self.BMAG == 0:
                print("")
            else:
                if self.BTHETA == 0 or self.BTHETA == 180:
                    print("")
                elif self.BTHETA == 90:
                    print("")
                else:
                    print("")
            self.TGAS = 273.15 + self.TEMPC
            self.ALPP = self.ALPHA * 760 * self.TGAS / (self.TORR * 293.15)
            self.ATTP = self.ATT * 760 * self.TGAS / (self.TORR * 293.15)
            self.SSTMIN = 30

            if abs(self.ALPP - self.ATTP) < self.SSTMIN:
                return
            if self.BMAG == 0.0:
                print("")
            elif self.BTHETA == 0.0 or self.BTHETA == 180:
                print("")
            elif self.BTHETA == 90:
                print("")
            else:
                print("")
