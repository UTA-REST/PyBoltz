import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Python')

from Magboltz import Magboltz

obj = Magboltz()
E = [50,100,150,200,250,300,350,400]
f = open("output.txt", "w")
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
    obj.EFIELD = E[i]
    obj.BMAG = 0
    obj.BTHETA = 90
    try:
        obj.Start()
        f.write(str((obj.FRAC[0]))) #CF4per
        f.write(str(("\n")))
        f.write(str((obj.FRAC[1]))) #ARper
        f.write(str(("\n")))
        f.write(str((obj.TEMPC)))   #TEMP
        f.write(str(("\n")))
        f.write(str((obj.TORR)))    #press
        f.write(str(("\n")))
        f.write(str((obj.EFIELD)))  #EFIELD
        f.write(str(("\n")))
        f.write(str(((obj.WZ*1e-5))))      #ZDRIFT
        f.write(str(("\n")))
        f.write(str((obj.DWZ)))     #ZERR
        f.write(str(("\n")))
        f.write(str((obj.DIFTR)))   #TDIFF
        f.write(str(("\n")))
        f.write(str((obj.DFTER)))  #TERR
        f.write(str(("\n")))
        f.write(str((obj.DIFLN)))   #LDIFF
        f.write(str(("\n")))
        f.write(str((obj.DFLER)))  #LERR
        f.write(str(("\n")))
        f.write(str((obj.AVE)))     #MELE
        f.write(str(("\n")))
        f.write(str((obj.DEN)))     #MERR
        f.write(str(("\n")))
    except ValueError:
        print("didnt work at i =" +str(i))
f.close()

