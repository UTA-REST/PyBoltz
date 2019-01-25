import sys
import warnings
import time

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
sys.path.append('../../src/Scripts/Cython')
from Gasmix import Gasmix

t1 = time.time()
TestObj = Gasmix()
NGS = []
for i in range(6):
    NGS.append(0)
for i in range(6):
    NGS[i] = 2

EROOT = [0 for i in range(4000)]
QT1 = [0 for i in range(4000)]
QT2 = [0 for i in range(4000)]
QT3 = [0 for i in range(4000)]
QT4 = [0 for i in range(4000)]
DEN = [1 for i in range(4000)]
DENS = 1
NGAS = 1
NSTEP = 4000
NANISO = 2
ESTEP = 1
EG = [6.25e-5 + i * (ESTEP) for i in range(4000)]
EFINAL = 0.5
AKT = 2.6037269846599997e-2
ARY = 1
TEMPC = 23
TORR = 760
IPEN = 0
TestObj.setCommons(NGS, EG, EROOT, QT1, QT2, QT3, QT4, DEN, DENS, NGAS, NSTEP,
                   NANISO, ESTEP, EFINAL, AKT, ARY, TEMPC, TORR, IPEN)
print("now")
if __name__ == '__main__':
    TestObj.Run()

print("hi")
t2 = time.time()
print("time:")
print(t2 - t1)
