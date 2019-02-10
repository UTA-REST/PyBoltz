import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from Magboltz import Magboltz
import numpy as np
Object = Magboltz()

import time
t1 =time.time()
Object.NGAS =1
Object.NMAX =1
Object.IPEN = 0
Object.ITHRM=1
Object.EFINAL = 0.0

Object.NGASN=[1, 0, 0, 0, 0, 0]

Object.FRAC=[100, 0, 0, 0, 0, 0]


Object.TEMPC = float(23)
Object.TORR = 750.062

Object.EFIELD =200
Object.BMAG = 0

Object.BTHETA =90
print("here")
Object.Start()

print(Object.WZ)
print(Object.WY)
print(Object.WX)
print(Object.DWZ)
print(Object.DWY)
print(Object.DWX)
print(Object.AVE)
print(Object.DEN)
print(Object.DIFTR)
print(Object.DIFLN)
t2 = time.time()

print("UNDAAAA DAAA C")
print(t2-t1)