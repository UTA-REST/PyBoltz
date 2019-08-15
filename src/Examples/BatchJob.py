import sys
import warnings
import os
import time
import numpy as np
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Python')
#t1 = time.time()
from Magboltz import Magboltz

Args = sys.argv
obj = Magboltz()
this_Gas   = int(Args[1])
frac_Gas   = float(Args[2])
frac_Ar    = round(100-frac_Gas,7)
frac_other = round(frac_Gas,7)
efield     = int(Args[3])

a = str(this_Gas)+"-"
b = '{0:.7f}'.format(frac_other)+"-"
c = str(efield)+".npy"
F_NAME = a+b+c
F_PATH = "/n/holylfs02/LABS/guenette_lab/users/amcdonald/magboltz/Argon_plus_Anything/Outputs/"


obj.__init__()
obj.NGAS   = 2
obj.NMAX   = 10
obj.IPEN   = 0
obj.ITHRM  = 1
obj.EFINAL = 0.0
obj.NGASN  = [2,this_Gas,0,0,0,0]
obj.FRAC   = [frac_Ar,frac_other,0,0,0,0]
obj.TEMPC  = 23
obj.TORR   = 750.062
obj.EFIELD = efield
obj.BMAG   = 0
obj.BTHETA = 0
try:
    obj.Start()
    print("Gas1")
    print(str((obj.FRAC[0]))) #Gas1 Percentage
    print("Gas2")
    print(str((obj.FRAC[1]))) #Gas2 Percentage
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

    INFO = [obj.FRAC[1], obj.TORR, obj.EFIELD, (obj.WZ*1e-5), obj.DWZ, 
             obj.DIFTR, obj.DFTER, obj.DIFLN, obj.DFLER, obj.DTMN, obj.DFTER1, 
             obj.DLMN, obj.DFLER1, obj.AVE, obj.DEN]
    INFO = np.array(INFO)
    print(INFO)
    np.save(F_PATH+F_NAME,INFO)
except ValueError:
    print("didnt work at i =" +str(i))
#t2 = time.time()
#print("time taken ",t2-t1)

