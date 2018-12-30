from Gasmix import Gasmix
from SETUPT import SETUPT
from Gasmix import Gasmix
import math
from MIXERT import MIXERT
from ELIMITT import ELIMITT
from ELIMITBT import ELIMITBT
from ELIMITCT import ELIMITCT
from MONTET import MONTET
from RAND48 import Rand48

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
        self.DENSY = [0 for i in range(4000)]
        self.SPEC = [0 for i in range(4000)]
        self.TIME = [0.0 for i in range(300)]
        self.ICOLL = [[0.0 for i in range(5)] for i in range(6)]
        self.ICOLNN = [[0.0 for i in range(10)] for i in range(6)]
        self.ICOLN = [[0.0 for i in range(290)] for i in range(6)]
        self.NMAX = 0.0
        self.ALPHA = 0.0
        self.AMGAS = [0.0 for i in range(6)]
        self.VTMB = [0.0 for i in range(6)]
        self.TCFMX = 0.0
        self.TCFMXG = [0.0 for i in range(6)]
        self.ITHRM = 0.0
        self.NGASN = [0 for i in range(6)]
        self.CORR = 0.0
        self.FRAC = [0.0 for i in range(6)]
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
        self.ANN = [0.0 for i in range(6)]
        self.VANN = [0.0 for i in range(6)]
        self.AN = 0.0
        self.VAN = 0.0
        self.QELM = [0 for i in range(4000)]
        self.QSUM = [0 for i in range(4000)]
        self.QION = [[0 for i in range(4000)] for j in range(6)]
        self.QIN = [[[0 for i in range(4000)] for j in range(250)] for g in range(6)]
        self.E = [0 for i in range(4000)]
        self.EROOT = [0 for i in range(4000)]
        self.QTOT = [0 for i in range(4000)]
        self.QREL = [0 for i in range(4000)]
        self.QINEL = [0 for i in range(4000)]
        self.NIN = [0 for i in range(6)]
        self.LION = [0 for i in range(6)]
        self.LIN = [[0 for i in range(250)] for j in range(6)]
        self.ALION = [0 for i in range(6)]
        self.ALIN = [[0 for i in range(250)] for j in range(6)]
        self.CF = [[[0 for i in range(290)] for j in range(4000)] for g in range(6)]
        self.TCF = [[0 for i in range(4000)] for j in range(6)]
        self.EIN = [[0 for i in range(290)] for j in range(6)]
        self.IARRY = [[0 for i in range(290)] for j in range(6)]
        self.RGAS = [[0 for i in range(290)] for j in range(6)]
        self.IPN = [[0 for i in range(290)] for j in range(6)]
        self.WPL = [[0 for i in range(290)] for j in range(6)]
        self.IPLAST = [0 for i in range(6)]
        self.ISIZE = [0 for i in range(6)]
        self.PENFRA = [[[0 for i in range(290)] for j in range(3)] for g in range(6)]
        self.TCFMAX = [0 for i in range(6)]
        self.CFN = [[[0 for i in range(10)] for j in range(4000)] for g in range(6)]
        self.TCFN = [[0 for i in range((4000))] for j in range(6)]
        self.SCLENUL = [[0 for i in range(10)] for j in range(6)]
        self.NPLAST = [0 for i in range(6)]
        self.PSCT = [[[0 for i in range(290)] for j in range(4000)] for g in range(6)]
        self.ANGCT = [[[0 for i in range(290)] for j in range(4000)] for g in range(6)]
        self.INDEX = [[0 for i in range(290)] for g in range(6)]
        self.NISO = 0
        self.FCION = [0 for i in range(4000)]
        self.FCATT = [0 for i in range(4000)]
        self.NC0 = [[0 for i in range(290)] for g in range(6)]
        self.EC0 = [[0 for i in range(290)] for g in range(6)]
        self.NG1 = [[0 for i in range(290)] for g in range(6)]
        self.EG1 = [[0 for i in range(290)] for g in range(6)]
        self.NG2 = [[0 for i in range(290)] for g in range(6)]
        self.EG2 = [[0 for i in range(290)] for g in range(6)]
        self.WKLM = [[0 for i in range(290)] for g in range(6)]
        self.EFL = [[0 for i in range(290)] for g in range(6)]
        self.IFAKET = [0 for i in range(8)]
        self.IFAKED = [0 for i in range(9)]
        self.QEL = [0 for i in range(4000)]
        self.QSATT = [0 for i in range(4000)]
        self.RNMX = [0 for i in range(6)]
        self.ZTOT = 0
        self.TOTT = 0
        self.ZTOTS = 0
        self.Mixobject = Gasmix()
        self.CONST1 = 0.0
        self.CONST2 = 0.0
        self.CONST3 = 0.0
        self.CONST4 = 0.0
        self.CONST5 = 0.0
        self.LAST = [0 for i in range(6)]
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
        self.RAND48 = Rand48(self.RSTART)

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
