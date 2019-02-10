from SETUPT import SETUPT
import math
from MIXERT import MIXERT
from ELIMITT import ELIMITT
from MONTET import MONTET
from libc.string cimport memset

cdef extern from "C/RM48.h":
    double DRAND48(double dummy)
    void RM48(double lenv)

cdef double drand48(double dummy):
    return DRAND48(dummy)
cdef class Magboltz:
    def __init__(self):
        self.EOVB = 0.0
        self.WB = 0.0
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

    def Start(self):
        cdef double EOB

        if self.ITHRM != 0:
            SETUPT(self)
            if self.EFINAL == 0.0:
                self.EFINAL = 0.5
                EOB = self.EFIELD * (self.TEMPC + 273.15) / (self.TORR * 293.15)
                if EOB > 15:
                    self.EFINAL = 8.0
                self.ESTART = self.EFINAL / 50.0
                while self.IELOW == 1:
                    print(str(self.API))
                    MIXERT(self)
                    if self.BMAG == 0 or self.BTHETA == 0 or abs(self.BTHETA) == 180:
                        ELIMITT(self)
                    elif self.BTHETA == 90:
                        print("")
                    else:
                        print("")
                    if self.IELOW == 1:
                        self.EFINAL = self.EFINAL * math.sqrt(2)
                        self.ESTART = self.EFINAL / 50
            else:
               MIXERT(self)
            if self.BMAG == 0:
                print(self.EFINAL)
                print(self.ESTART)
                MONTET(self)
            else:
                if self.BTHETA == 0 or self.BTHETA == 180:
                    print("")
                elif self.BTHETA == 90:
                    print("")
                else:
                    print("")
            return
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
