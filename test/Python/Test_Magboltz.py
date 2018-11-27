import sys
import warnings
import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Python')

from Magboltz import Magboltz

TestObj = Magboltz()

TestObj.NGAS =1
TestObj.NMAX =1
TestObj.IPEN = 0
TestObj.ITHRM=1
TestObj.EFINAL = 0.0

TestObj.NGASN=[1,0,0,0,0,0]

TestObj.FRAC=[100,0,0,0,0,0]

TestObj.TEMPC = 20
TestObj.TORR = 750

TestObj.EFIELD =200
TestObj.BMAG = 10

TestObj.BTHETA =90

TestObj.Start()

print(TestObj.EFINAL)
print(TestObj.TCFMX)