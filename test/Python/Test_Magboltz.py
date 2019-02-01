import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Python')

from Magboltz import Magboltz

Magboltz = Magboltz()


Magboltz.NGAS =2
Magboltz.NMAX =1
Magboltz.IPEN = 0
Magboltz.ITHRM=1
Magboltz.EFINAL = 0.0

Magboltz.NGASN=[1,2,0,0,0,0]

Magboltz.FRAC=[50,50,0,0,0,0]

Magboltz.TEMPC = 23
Magboltz.TORR = 5250.43

Magboltz.EFIELD =200
Magboltz.BMAG = 0

Magboltz.BTHETA =90

Magboltz.Start()

print(Magboltz.EFINAL)
print(Magboltz.TCFMX)
