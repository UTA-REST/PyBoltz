from Gasmix import Gasmix
from SETUPT import SETUPT
from Gasmix import Gasmix
import numpy as np
import math
from MIXERT import MIXERT
from ELIMITT import ELIMITT
from ELIMITBT import ELIMITBT
from ELIMITCT import ELIMITCT
from MONTET import MONTET
from RAND48 import Rand48
from MONTEAT import MONTEAT
from MONTEBT import MONTEBT
from MONTECT import MONTECT


class Magboltz:
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
        self.TMAX = 0.0
        self.SMALL = 0.0
        self.API = 0.0
        self.ESTART = 0.0
        self.THETA = 0.0
        self.PHI = 0.0
        self.EFIELD = 0.0
        self.DENSY = np.zeros(4000)
        self.SPEC = np.zeros(4000)
        self.TIME = np.zeros(300)
        self.ICOLL = np.zeros(shape=(6, 5))
        self.ICOLNN = np.zeros(shape=(6, 10))
        self.ICOLN = np.zeros(shape=(6, 290))
        self.NMAX = 0.0
        self.ALPHA = 0.0
        self.AMGAS = np.zeros(6)
        self.VTMB = np.zeros(6)
        self.TCFMX = 0.0
        self.TCFMXG = np.zeros(6)
        self.ITHRM = 0.0
        self.NGASN = np.zeros(6)
        self.CORR = 0.0
        self.FRAC = np.zeros(6)
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
        self.ANN = np.zeros(6)
        self.VANN = np.zeros(6)
        self.AN = 0.0
        self.VAN = 0.0
        self.QELM = np.zeros(4000)
        self.QSUM = np.zeros(4000)
        self.QION = np.zeros(shape=(6, 4000))
        self.QIN = np.zeros(shape=(6, 250, 4000))
        self.E = np.zeros(4000)
        self.EROOT = np.zeros(4000)
        self.QTOT = np.zeros(4000)
        self.QREL = np.zeros(4000)
        self.QINEL = np.zeros(4000)
        self.NIN = np.zeros(6)
        self.LION = np.zeros(6)
        self.LIN = np.zeros(shape=(6, 250))
        self.ALION = np.zeros(6)
        self.ALIN = np.zeros(shape=(6, 250))
        self.CF = np.zeros(shape=(6, 4000, 290))
        self.TCF = np.zeros(shape=(6, 4000))
        self.EIN = np.zeros(shape=(6, 290))
        self.IARRY = np.zeros(shape=(6, 290))
        self.RGAS = np.zeros(shape=(6, 290))
        self.IPN = np.zeros(shape=(6, 290))
        self.WPL = np.zeros(shape=(6, 290))
        self.IPLAST = np.zeros(6)
        self.ISIZE = np.zeros(6)
        self.PENFRA = np.zeros(shape=(6, 3, 290))
        self.TCFMAX = np.zeros(6)
        self.CFN = np.zeros(shape=(6, 4000, 10))
        self.TCFN = np.zeros(shape=(6, 4000))
        self.SCLENUL = np.zeros(shape=(6, 10))
        self.NPLAST = np.zeros(6)
        self.PSCT = np.zeros(shape=(6, 4000, 290))
        self.ANGCT = np.zeros(shape=(6, 4000, 290))
        self.INDEX = np.zeros(shape=(6, 290))
        self.NISO = 0
        self.FCION = np.zeros(4000)
        self.FCATT = np.zeros(4000)
        self.NC0 = np.zeros(shape=(6, 290))
        self.EC0 = np.zeros(shape=(6, 290))
        self.NG1 = np.zeros(shape=(6, 290))
        self.EG1 = np.zeros(shape=(6, 290))
        self.NG2 = np.zeros(shape=(6, 290))
        self.EG2 = np.zeros(shape=(6, 290))
        self.WKLM = np.zeros(shape=(6, 290))
        self.EFL = np.zeros(shape=(6, 290))
        self.IFAKET = np.zeros(8)
        self.IFAKED = np.zeros(9)
        self.QEL = np.zeros(4000)
        self.QSATT = np.zeros(4000)
        self.RNMX = np.zeros(6)
        self.ES = np.zeros(4000)
        self.ZTOT = 0
        self.TOTT = 0
        self.ZTOTS = 0
        self.Mixobject = Gasmix()
        self.CONST1 = 0.0
        self.CONST2 = 0.0
        self.CONST3 = 0.0
        self.CONST4 = 0.0
        self.CONST5 = 0.0
        self.LAST = np.zeros(6)
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
        self.RAND48 = Rand48(self.RSTART)
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
        self.ETPL = np.zeros(8)
        self.XTPL = np.zeros(8)
        self.YTPL = np.zeros(8)
        self.ZTPL = np.zeros(8)
        self.TTPL = np.zeros(8)
        self.XXTPL = np.zeros(8)
        self.YYTPL = np.zeros(8)
        self.ZZTPL = np.zeros(8)
        self.VZTPL = np.zeros(8)
        self.NETPL = np.zeros(8)
        self.ATTOINT = 0.0
        self.ATTERT = 0.0
        self.AIOERT = 0.0
        self.ALPHERR = 0.0
        self.ATTSST = 0.0
        self.TTOT = 0.0
        self.ATTERR = 0.0
        self.ZPLANE = np.zeros(8)
        self.IZFINAL = 0.0
        self.RALPHA = 0.0
        self.RALPER = 0.0
        self.TODENE = 0.0
        self.TOFENER = 0.0
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
        self.ZFINAL = 0.0
        self.ITFINAL = 0.0
        self.IPRIM = 0.0
        self.XS = np.zeros(2000)
        self.YS = np.zeros(2000)
        self.ZS = np.zeros(2000)
        self.TS = np.zeros(2000)
        self.ES = np.zeros(2000)
        self.DCX = np.zeros(2000)
        self.DCY = np.zeros(2000)
        self.DCZ = np.zeros(2000)
        self.IPL = np.zeros(2000)
        self.ST = 0.0

    def Start(self):
        if self.ITHRM != 0:
            self = SETUPT(self)
            if self.EFINAL == 0.0:
                self.EFINAL = 0.5
                EOB = self.EFIELD * (self.TEMPC + 273.15) / (self.TORR * 293.15)
                if EOB > 15:
                    self.EFINAL = 8
                self.ESTART = self.EFINAL / 50
                while self.IELOW == 1:
                    self = MIXERT(self)
                    if self.BMAG == 0 or self.BTHETA == 0 or abs(self.BTHETA) == 180:
                        self = ELIMITT(self)
                    elif self.BTHETA == 90:
                        self = ELIMITBT(self)
                    else:
                        self = ELIMITCT(self)
                    if self.IELOW == 1:
                        # TODO: This could be the reason for missing points
                        self.EFINAL = self.EFINAL * math.sqrt(2)
                        self.ESTART = self.EFINAL / 50
            else:
                self = MIXERT(self)
            # TODO: add a printing function "PRINTERT"

            if self.BMAG == 0:
                self = MONTET(self)
            else:
                if self.BTHETA == 0 or Magboltz.BTHETA == 180:
                    self = MONTEAT(self)
                elif self.BTHETA == 90:
                    self = MONTEBT(self)
                else:
                    self = MONTECT(self)
            self.TGAS = 273.15 + self.TEMPC
            self.ALPP = self.ALPHA * 760 * self.TGAS / (self.TORR * 293.15)
            self.ATTP = self.ATT * 760 * self.TGAS / (self.TORR * 293.15)
            self.SSTMIN = 30

            if self.BMAG == 0.0:
                self = ALPCALCT(self)
