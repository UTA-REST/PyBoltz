import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Python')

from Magboltz import Magboltz
TORR = input("Torr")

print(TORR)
Magboltz = [Magboltz() for i in range(12)]

prec =[[10,90,0,0,0,0] for i in range(4)]
prec.append([50,50,0,0,0,0])
prec.append([50,50,0,0,0,0])
prec.append([50,50,0,0,0,0])
prec.append([50,50,0,0,0,0])
prec.append([90,10,0,0,0,0])
prec.append([90,10,0,0,0,0])
prec.append([90,10,0,0,0,0])
prec.append([90,10,0,0,0,0])
E = [50,100,150,200]
for i in range(12):
    Magboltz[i].NGAS =2
    Magboltz[i].NMAX =1
    Magboltz[i].IPEN = 0
    Magboltz[i].ITHRM=1
    Magboltz[i].EFINAL = 0.0
    Magboltz[i].NGASN=[1,2,0,0,0,0]
    Magboltz[i].FRAC=prec[i]
    Magboltz[i].TEMPC = 23
    Magboltz[i].TORR = float(TORR)
    Magboltz[i].EFIELD = E[i%4]
    Magboltz[i].BMAG = 0
    Magboltz[i].BTHETA =90

for i in range(12):
	Magboltz[i].Start()
f= open("output_"+str(int(TORR))+".txt","w")
for i in range(12):
    f.write(Magboltz[i].WZ)
    f.write(Magboltz[i].WY)
    f.write(Magboltz[i].WX)
    f.write(Magboltz[i].DWZ)
    f.write(Magboltz[i].DWY)
    f.write(Magboltz[i].DWX)
    f.write(Magboltz[i].AVE)
    f.write(Magboltz[i].DEN)
    f.write(Magboltz[i].TORR)
    f.write(Magboltz[i].EFIELD)

