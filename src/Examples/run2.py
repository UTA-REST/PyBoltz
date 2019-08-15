import sys
import warnings
import os
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Python')

from object import object

obj = object()
E = [50,100,150,200,250,300,350,400]
for i in range(8):
    obj.__init__()
    obj.NGAS =1
    obj.NMAX = 10
    obj.IPEN = 0
    obj.ITHRM=1
    obj.EFINAL = 0.0
    obj.NGASN=[7,0,0,0,0,0]
    obj.FRAC=[100,0,0,0,0,0]
    obj.TEMPC = 23
    obj.TORR = 750.062
    obj.EFIELD = float(Args[1])
    obj.A      = float(Args[2])
    obj.B      = float(Args[3])
    obj.C      = float(Args[4])
    obj.Org    = float(Args[5])
    obj.BMAG = 0
    obj.BTHETA = 0
    try:
        obj.Start()
        print("Gas1")
        print(str((obj.FRAC[0]))) #Gas1 Percentage
        print("Tempature")
        print(str((obj.TEMPC)))   #TEMP
        print("Pressure")
        print(str((obj.TORR)))    #press
        print("Efield")
        print(str((obj.EFIELD)))  #EFIELD
        print("Z drift")
        print(str(((obj.WZ*1e-5))))      #ZDRIFT
        print(str((obj.DWZ)))     #ZERR
        print("Transverse")
        print(str((obj.DIFTR)))   #TDIFF
        print(str((obj.DFTER)))  #TERR
        print("Longitudinal")
        print(str((obj.DIFLN)))   #LDIFF
        print(str((obj.DFLER)))  #LERR
        print("Mean electron energy")
        print(str((obj.AVE)))     #MELE
        print(str((obj.DEN)))     #MERR
    except ValueError:
        print("didnt work at i =" +str(i))
