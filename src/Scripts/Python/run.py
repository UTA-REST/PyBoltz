import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Python')

from Magboltz import Magboltz

obj = Magboltz()

prec =[[10,90,0,0,0,0] for i in range(4)]
prec.append([50,50,0,0,0,0])
prec.append([50,50,0,0,0,0])
prec.append([50,50,0,0,0,0])
prec.append([50,50,0,0,0,0])
prec.append([90,10,0,0,0,0])
prec.append([90,10,0,0,0,0])
prec.append([90,10,0,0,0,0])
prec.append([90,10,0,0,0,0])
TORR = [750.062,7500.62]
E = [50,100,150,200]
f = open("output_drand48.txt", "w")
for i in range(24):
    obj.__init__()
    obj.NGAS =2
    obj.NMAX =1
    obj.IPEN = 0
    obj.ITHRM=1
    obj.EFINAL = 0.0
    obj.NGASN=[1,2,0,0,0,0]
    obj.FRAC=prec[i%12]
    obj.TEMPC = 23
    obj.TORR = TORR[i%2]
    obj.EFIELD = E[i%4]
    obj.BMAG = 0
    obj.BTHETA = 90
    obj.Start()
    f.write(obj.FRAC[0]) #CF4per
    f.write(obj.FRAC[1]) #ARper
    f.write(obj.TEMPC)   #TEMP
    f.write(obj.TORR)    #press
    f.write(obj.EFIELD)  #EFIELD
    f.write((obj.WZ*1e-5))      #ZDRIFT
    f.write(obj.DWZ)     #ZERR
    f.write(obj.DIFTR)   #TDIFF
    f.write(obj.DFTER)  #TERR
    f.write(obj.DIFLN)   #LDIFF
    f.write(obj.DFLER)  #LERR
    f.write(obj.AVE)     #MELE
    f.write(obj.DEN)     #MERR
f.close()

