import sys
import warnings
import os
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Python')

from Magboltz import Magboltz
Args = sys.argv
obj = Magboltz()


obj.__init__()
obj.NGAS   = 1
obj.NMAX   = 15
obj.IPEN   = 0
obj.ITHRM  = 1
obj.EFINAL = 0.0
obj.NGASN  = [7,0,0,0,0,0]
obj.FRAC   = [100,0,0,0,0,0]
obj.TEMPC  = 23
obj.TORR   = 750.062
obj.EFIELD = float(Args[1])
obj.A      = float(Args[2])
obj.D      = float(Args[3])
obj.F      = float(Args[4])
obj.A1     = float(Args[5])
obj.Lambda = float(Args[6])
obj.EV0    = float(Args[7])
obj.BMAG   = 0
obj.BTHETA = 0
try:
    obj.Start()
    print("Gas1")
    print(str((obj.FRAC[0]))) #Gas1 Percentage
    print("TEMPERATURE")
    print(str((obj.TEMPC)))   #TEMP
    print("PRESSURE")
    print(str((obj.TORR)))    #press
    print("ELECTRIC FIELD")
    print(str((obj.EFIELD)))  #EFIELD
    print("Z DRIFT VELOCITY")
    print(str(((obj.WZ*1e-5))))      #ZDRIFT
    print(str((obj.DWZ)))     #ZERR
    print("TRANSVERSE DIFFUSION CM**2/SEC")
    print(str((obj.DIFTR)))   #TDIFF
    print(str((obj.DFTER)))  #TERR
    print("LONGITUDINAL DIFFUSION CM**2/SEC")
    print(str((obj.DIFLN)))   #LDIFF
    print(str((obj.DFLER)))  #LERR
    print("TRANSVERSE DIFFUSION MICRONS/CENTIMETER**0.5")
    print(str((obj.DTMN)))
    print(str((obj.DFTER1)))
    print("LONGITUDINAL DIFFUSION MICRONS/CENTIMETER**0.5")
    print(str((obj.DLMN)))
    print(str((obj.DFLER1)))  #LERR
    print("MEAN ELECTRON ENERGY")
    print(str((obj.AVE)))     #MELE
    print(str((obj.DEN)))     #MERR
except ValueError:
    print("didnt work at i =" +str(i))
